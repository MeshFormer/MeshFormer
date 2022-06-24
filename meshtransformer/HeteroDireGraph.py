import json, os
import math, copy, time
import numpy as np
import multiprocessing as mp
import pandas as pd
import torch 
import math
import dill

from tqdm import tqdm
from functools import partial
from collections import defaultdict


class HeteoDirectedGraph():
    def __init__(self):
        super(HeteoDirectedGraph, self).__init__()
        self.node_forward = defaultdict(lambda: {})
        self.node_bacward = defaultdict(lambda: [])
        self.node_feature = defaultdict(lambda: [])
        self.node_label = defaultdict(lambda: [] )


        self.edge_list = defaultdict( #target_type
                            lambda: defaultdict(  #source_type
                                lambda: defaultdict(  #relation_type
                                    lambda: defaultdict(  #target_id
                                        lambda: defaultdict( #source_id(
                                            lambda: float # weight
                                        )))))

        self.face_list = []
        self.face_type = []
        self.times = {}

    def add_node(self, node):
        nfl = self.node_forward[node['type']]
        if node['id'] not in nfl:
            self.node_bacward[node['type']] += [node]
            ser = len(nfl)
            nfl[node['id']] = ser
            if 'feature' in node.keys():
                self.node_feature[node['type']].append(node['feature'].tolist())
            if 'label' in node.keys():
                self.node_label[node['type']].append(node['label'])
            return ser
        return nfl[node['id']]



    def add_edge(self, source_node, target_node, weight = 1.0, relation_type = None, directed = False):
        edge = [self.add_node(source_node), self.add_node(target_node)]


        '''
            Add bi-directional edges with different relation type
        '''
        self.edge_list[target_node['type']][source_node['type']][relation_type][edge[1]][edge[0]] = weight

    def add_face(self, nodes, type):
        nfl = self.node_forward['sn']
        self.face_list.append([nfl[nodes[0]], nfl[nodes[1]], nfl[nodes[2]] ])
        self.face_type.append(type)


    def update_node(self, node):
        nbl = self.node_bacward[node['type']]
        ser = self.add_node(node)
        for k in node:
            if k not in nbl[ser]:
                nbl[ser][k] = node[k]

    def get_meta_graph(self):
        types = self.get_types()
        metas = []
        for target_type in self.edge_list:
            for source_type in self.edge_list[target_type]:
                for r_type in self.edge_list[target_type][source_type]:
                    metas += [(target_type, source_type, r_type)]
        return metas
    
    def add_triangles(self,faces):
        self.faces = []
        nfl = self.node_forward['sn']
        for i in faces:
            t1 = nfl[i[0]]
            t2 = nfl[i[1]]
            t3 = nfl[i[2]]
            self.faces.append([t1,t2,t3])
        

    def get_types(self):
        return list(self.node_feature.keys())

    def get_types(self):
        return list(self.node_feature.keys())






def to_torch(feature, edge_list, graph):
    node_dict = {}
    node_feature = []
    node_type    = []
    edge_index   = []
    edge_type    = []
    edge_weight = []
    faces = graph.faces

    nodefeature_location = dict()

    
    node_num = 0
    types = graph.get_types()
    for t in types:
        node_dict[t] = [node_num, len(node_dict)]
        nodefeature_location[t] = [node_num, len(feature[t])]
        node_num     += len(feature[t])


    num_type = len(types)
    tcount = 0
    for t in types:
        add_feature = np.array(feature[t])
        tm = np.zeros(add_feature.shape[0])
        tm1 = np.zeros(num_type)
        tm1[tcount] = 1
        ttm = np.meshgrid(tm1, tm)[0]
        add_feature = np.hstack([add_feature, ttm])
        
        node_feature += add_feature.tolist()
        node_type    += [node_dict[t][1] for _ in range(len(feature[t]))]
        tcount += 1

    edge_dict = {e[2]: i for i, e in enumerate(graph.get_meta_graph())}
    edge_dict['self'] = len(edge_dict)

    for target_type in edge_list:
        for source_type in edge_list[target_type]:
            for relation_type in edge_list[target_type][source_type]:
                for target_id in edge_list[target_type][source_type][relation_type]:
                    for source_id in edge_list[target_type][source_type][relation_type][target_id]:
                        weight = edge_list[target_type][source_type][relation_type][target_id][source_id]
                        edge_index += [[source_id + node_dict[source_type][0],target_id + node_dict[target_type][0]]]
                        edge_type += [edge_dict[relation_type]]
                        edge_weight += [weight]


    node_feature = torch.FloatTensor(node_feature)
    node_type    = torch.LongTensor(node_type)
    edge_weight    = torch.FloatTensor(edge_weight)
    edge_index   = torch.LongTensor(edge_index).t()
    edge_type    = torch.LongTensor(edge_type)
    faces = torch.LongTensor(faces)
    return node_feature, node_type, edge_weight, edge_index, edge_type, node_dict, edge_dict, nodefeature_location, faces
    

    
class RenameUnpickler(dill.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "GPT_GNN.data" or module == 'data':
            renamed_module = "pyHGT.data"
        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()
