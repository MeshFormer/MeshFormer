import os
import sys
import torch as t
import networkx as nx
import networkit as nk
import random
import pickle as pk
import trimesh as tri
import networkx as nx
import torch 
import numpy as np 
import time 

from collections import defaultdict
from vedo import Mesh as vMesh
from copy import deepcopy

from util.util import *
from geometry.Curvature import Curvature
from MeshTransformer.HeteroDireGraph import HeteoDirectedGraph



SHAPE_CLUSTER_LIMIT = 800


def _get_all_pairs_shortest_path(G):
    G.remove_edges_from(nx.selfloop_edges(G))
    _Gk = nk.nxadapter.nx2nk(G, weightAttr='weight')
    t0 = time.time()
    apsp = nk.distance.APSP(_Gk).run().getDistances()
    apsp =  np.array(apsp)
    return apsp



class MeshHeteroDirectedGraph:
    def __init__(self, mesh1, params,  recalcu=True, recalcu_order=3, model=None, assign_cc=None):
        if mesh1 ==None or params==None:
            return
        self.mesh = deepcopy(mesh1)
        if self.mesh.data['coords'].shape[0] != self.mesh.data['mesh'].vertices.shape[0]:
            self.mesh.data['coords'] = deepcopy(self.mesh.data['mesh'].vertices)
            self.mesh.data['faces'] = deepcopy(self.mesh.data['mesh'].faces)

        assert len(self.mesh.data['facelabel'].keys()) == self.mesh.data['mesh'].faces.shape[0]

        pointlabel = defaultdict(list)
        for face_index in range(len(self.mesh.data['mesh'].faces)):
            face = self.mesh.data['mesh'].faces[face_index]
            for i in range(len(face)):
                pointlabel[face[i]].append(self.mesh.data['facelabel'][face_index])
        last_point_label = dict()
        for i in range(self.mesh.data['mesh'].vertices.shape[0]):
            if i in pointlabel.keys():
                last_point_label[i] = max(pointlabel[i])
            else:
                print("label error")
                raise Exception()
        self.mesh.data['pointlabel'] = last_point_label


        self.cc = assign_cc
        self.recalcu = recalcu
        self.recalcu_order = recalcu_order
        self.feature_distance = "diffusion_weight_feature"


        if not hasattr(self.mesh, 'weights') and "weight" in self.feature_distance:
            if model==None:
                raise Exception("not xgboost model!")
            self.mesh.weights =  model.model.predict(mesh1.vertexfeatures)


        values = max(set(self.cc[0][1].values())) + 1
        cormap = []

        for i in range(values + 1):
            r = random.random() * 255
            b = random.random() * 255
            g = random.random() * 255
            color = (r, g, b)
            cormap.append(color)

        self.mesh.data['pointcolor'] = []
        self.mesh.data['riccipointlabel'] = dict()
        label_to_nodes = defaultdict(list)

        for i in range(self.mesh.data['coords'].shape[0]):
            if i in self.cc[0][1].keys():
                self.mesh.data['pointcolor'].append(cormap[self.cc[0][1][i]])
                self.mesh.data['riccipointlabel'][i] = self.cc[0][1][i]
                label_to_nodes[self.cc[0][1][i]].append(i)
            else:
                self.mesh.data['riccipointlabel'][i] = values + 1
                self.mesh.data['pointcolor'].append(cormap[-1])
                label_to_nodes[values + 1].append(i)
                print('this is a huge mistake.')

        self.mesh.data['pointcolor'] = np.array(self.mesh.data['pointcolor'])
        tvertexfeatures = torch.Tensor(self.mesh.vertexfeatures).cuda()
        index = torch.where(tvertexfeatures==-np.inf)
        if len(index[0])>0 and -np.inf in tvertexfeatures[index].tolist():
            tvertexfeatures[index] = np.inf
            minv = torch.min(tvertexfeatures)
            tvertexfeatures[index] = minv - 1
        index = torch.where(tvertexfeatures == np.inf)
        if len(index[0])>0 and np.inf in tvertexfeatures[torch.where(torch.isinf(tvertexfeatures))].tolist():
            tvertexfeatures[index] = -np.inf
            maxv = torch.max(tvertexfeatures)
            tvertexfeatures[index] = maxv + 1
        index = torch.where(torch.isnan(tvertexfeatures))
        if len(index[0]) > 0:
            tvertexfeatures[index] = 0
        self.mesh.vertexfeatures = tvertexfeatures.cpu().numpy()



        cur = Curvature(self.feature_distance)
        Gx = self.mesh.calculateAsWeightGraph(cur)
        self.sourceMesh = Gx

        max_weight = max(dict(Gx.edges).items(), key=lambda x: x[1]['weight'])[1]['weight']
        min_weight = min(dict(Gx.edges).items(), key=lambda x: x[1]['weight'])[1]['weight']

        for i in label_to_nodes[values + 1]:
            neighbors = tri.proximity.nearby_faces(self.mesh.data['mesh'], [self.mesh.data['mesh'].vertices[i]])[0]
            neighbors = self.mesh.data['mesh'].faces[neighbors[0]]
            for j in neighbors:
                Gx.add_edge(i, j, weight=max_weight)
            labels = [self.mesh.data['riccipointlabel'][k] for k in neighbors]
            counts = np.bincount(labels)
            label = np.argmax(counts)
            self.mesh.data['riccipointlabel'][i] = label
            label_to_nodes[label].append(i)

        label_to_nodes.pop(values + 1)
        self.cluster_relationship = label_to_nodes

        cluster_node_weight = dict()
        for i in label_to_nodes.keys():
            first_graph = Gx.subgraph(label_to_nodes[i])
            center, distance, idx = _get_all_pairs_shortest_path(first_graph)
            if max(distance) == sys.float_info.max:
                print('max distance error')
                distance[distance == sys.float_info.max] = -1
                max_dis = np.max(distance)
                distance[distance == -1] = max_dis

            if min(distance) == 0:
                print('min distance error')
                distance[distance <= 0] = sys.float_info.max
                min_dis = np.min(distance)
                distance[distance == sys.float_info.max] = min_dis / 2

            distance = (max(distance) + min(distance) - distance) / np.sum(distance)
            cluster_node_weight[i] = (center, distance, idx)

        cluster_b_node = max(Gx.nodes) + 1
        global_b_node = max(Gx.nodes) + 1 + len(label_to_nodes.keys())

        assert len(self.mesh.data['pointlabel'].keys()) == self.mesh.data['mesh'].vertices.shape[0]
        labels = [ self.mesh.data['pointlabel'][i] for i in range(self.mesh.data['mesh'].vertices.shape[0])]

        self.labels = labels
        self.facelabels = [ self.mesh.data['facelabel'][i] for i in range(self.mesh.data['mesh'].faces.shape[0])]
        self.one_hot_labels = None

        div_value = np.array(self.mesh.vertexfeatures).max(axis=0) - np.array(self.mesh.vertexfeatures).min(axis=0)
        self.mesh.vertexfeatures = (np.array(self.mesh.vertexfeatures) - np.array(
            self.mesh.vertexfeatures).min(axis=0)[None, :]) / div_value

        self.hgraph = HeteoDirectedGraph()

        for i in Gx.nodes:
            self.hgraph.add_node({'id': i, 'type': 'sn', 'feature': self.mesh.vertexfeatures[i],'label':self.labels[i]})

        for i, j in Gx.edges:
            self.hgraph.add_edge({'id': i, 'type': 'sn'}, {'id': j, 'type': 'sn'},
                            max_weight + min_weight - Gx[i][j]['weight'],
                            relation_type='s_s', directed=True)

        for f in range(self.mesh.data['mesh'].faces.shape[0]):
            self.hgraph.add_face(self.mesh.data['mesh'].faces[f], self.facelabels[f])

        for i in label_to_nodes.keys():
            center, distance, idx = cluster_node_weight[i]
            t_feature = np.average(self.mesh.vertexfeatures[idx], axis=0, weights=distance)
            self.hgraph.add_node({'id': i + cluster_b_node, 'type': 'cn', 'feature': t_feature,'label':-1})

            for j in range(len(idx)):
                self.hgraph.add_edge({'id': i + cluster_b_node, 'type': 'cn'}, {'id': idx[j], 'type': 'sn'}, distance[j],
                                relation_type='c_s', directed=True)
                self.hgraph.add_edge({'id': idx[j], 'type': 'sn'}, {'id': i + cluster_b_node, 'type': 'cn'}, distance[j],
                                relation_type='s_c', directed=True)
            print('center distance is ' + str(distance[center]))

        label_to_label_count = defaultdict(lambda: defaultdict(float))
        for i, j in Gx.edges:
            if self.mesh.data['riccipointlabel'][i] != self.mesh.data['riccipointlabel'][j]:
                label_to_label_count[self.mesh.data['riccipointlabel'][i]][
                    self.mesh.data['riccipointlabel'][j]] += max_weight + min_weight - Gx[i][j]['weight']
                label_to_label_count[self.mesh.data['riccipointlabel'][j]][
                    self.mesh.data['riccipointlabel'][i]] += max_weight + min_weight - Gx[i][j]['weight']

        for i in label_to_label_count.keys():
            weights = np.sum(list(label_to_label_count[i].values()))
            for j in label_to_label_count[i].keys():
                label_to_label_count[i][j] = label_to_label_count[i][j] / weights
                ## cluster Mesh to cluster Mesh
        for i in label_to_label_count.keys():
            for j in label_to_label_count[i].keys():
                self.hgraph.add_edge({'id': i + cluster_b_node, 'type': 'cn'}, {'id': j + cluster_b_node, 'type': 'cn'},
                                label_to_label_count[i][j], relation_type='c_c', directed=True)

        # global Mesh
        self.hgraph.add_node({'id': global_b_node, 'type': 'gn', 'feature': np.average(self.mesh.vertexfeatures, axis=0),'label':-2})

        for i in Gx.nodes:
            self.hgraph.add_edge({'id': global_b_node, 'type': 'gn'}, {'id': i, 'type': 'sn'}, 1.0 / len(Gx.nodes),
                            relation_type='g_s', directed=True)
            self.hgraph.add_edge({'id': i, 'type': 'sn'}, {'id': global_b_node, 'type': 'gn'}, 1.0 / len(Gx.nodes),
                            relation_type='s_g', directed=True)

        for i in label_to_nodes.keys():
            self.hgraph.add_edge({'id': global_b_node, 'type': 'gn'}, {'id': i + cluster_b_node, 'type': 'cn'},
                            1.0 / len(label_to_nodes.keys()), relation_type='g_c', directed=True)
            self.hgraph.add_edge({'id': i + cluster_b_node, 'type': 'cn'}, {'id': global_b_node, 'type': 'gn'},
                            1.0 / len(label_to_nodes.keys()), relation_type='c_g', directed=True)




    def set_one_hot_label(self,enc):
        self.one_hot_labels = enc.transform(np.array(self.hgraph.node_label['sn']).reshape(-1, 1)).todense()
        self.one_hot_face_labels = enc.transform(np.array(self.hgraph.face_type).reshape(-1,1)).todense()

    def get_label(self):
        return self.hgraph.node_label['sn']

    def train_data(self):
        return self.hgraph, self.one_hot_labels

    def train_face_data(self):
        return self.hgraph, self.one_hot_face_labels

    def train_all_data(self):
        return self.hgraph, self.one_hot_face_labels, self.one_hot_labels, self.face_area_weight / np.sum(self.face_area_weight)




