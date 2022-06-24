from geometry.mesh import Mesh

import vtk
import numpy as np
from util import util
import sys
import networkx as nx
alltypes = ['mean','gauss','max','min','diffusion','diffusion_feature','diffusion_weight','diffusion_weight_feature']


class Curvature:
    def __init__(self,type):
        assert type in alltypes
        self.type = type

    def calculate(self,mesh,direction=12):

        if self.type == 'mean':
            res = self.calculateMeanCurv(mesh)
        if self.type == 'gauss':
            res = self.calculateGaussCurv(mesh)
        if self.type == 'max':
            res = self.calculateMaxCurv(mesh)
        if self.type == 'min':
            res = self.calculateMinCurv(mesh)
        if self.type == 'diffusion':
            res = self.calculateDiffCurv(mesh,direction)
        if self.type == 'diffusion_feature':
            res = self.calculateDiffCurv_vertexFeature(mesh, direction)
        if self.type == 'diffusion_weight':
            res = self.calculateDiffCurv_weighted(mesh, direction)
        if self.type == 'diffusion_weight_feature':
            res = self.calculateDiffCurv_weighted_features(mesh, direction)
        return res
        
    def calculateMeanCurv(self,mesh):
        curvature = vtk.vtkCurvatures()
        curvature.SetCurvatureTypeToMean()
        curvature.SetInputData(mesh.convert_to_vtk_mesh())
        curvature.Update()
        curvature = curvature.GetOutputDataObject(0)
        curvs = []
        for i in range(curvature.GetNumberOfPoints()):
            curvs.append(curvature.GetPointData().GetArray(0).GetTuple1(i))
        mesh.setNodeAttribute('mean_cur',np.array(curvs))
        return mesh.data['nodedata']['mean_cur']



    def calculateGaussCurv(self,mesh):
        curvature = vtk.vtkCurvatures()
        curvature.SetCurvatureTypeToGaussian()
        curvature.SetInputData(mesh.convert_to_vtk_mesh())
        curvature.Update()
        curvature = curvature.GetOutputDataObject(0)
        curvs = []
        for i in range(curvature.GetNumberOfPoints()):
            curvs.append(curvature.GetPointData().GetArray(0).GetTuple1(i))
        mesh.setNodeAttribute('gauss_cur',np.array(curvs))
        return mesh.data['nodedata']['gauss_cur']

    def calculateMaxCurv(self,mesh):
        curvature = vtk.vtkCurvatures()
        curvature.SetCurvatureTypeToMaximum()
        curvature.SetInputData(mesh.convert_to_vtk_mesh())
        curvature.Update()
        curvature = curvature.GetOutputDataObject(0)
        curvs = []
        for i in range(curvature.GetNumberOfPoints()):
            curvs.append(curvature.GetPointData().GetArray(0).GetTuple1(i))
        mesh.setNodeAttribute('max_cur',np.array(curvs))
        return mesh.data['nodedata']['max_cur']


    def calculateMinCurv(self,mesh):
        curvature = vtk.vtkCurvatures()
        curvature.SetCurvatureTypeToMinimum()
        curvature.SetInputData(mesh.convert_to_vtk_mesh())
        curvature.Update()
        curvature = curvature.GetOutputDataObject(0)
        curvs = []

        for i in range(curvature.GetNumberOfPoints()):
            curvs.append(curvature.GetPointData().GetArray(0).GetTuple1(i))
        mesh.setNodeAttribute('min_cur',np.array(curvs))
        return mesh.data['nodedata']['min_cur']

    def calculateDiffCurv(self,mesh,direction):
        mm = mesh.data['mesh']
        vnormal = mm.vertex_normals
        fnormal = mm.face_normals
        neighbor = mm.vertex_neighbors

        cur_node = dict()
        graph_weight = dict()

        for i in range(len(vnormal)):
            cur_node[i] = dict()
            positiona = mm.vertices[i]
            dir  = []
            cur  = []
            for j in neighbor[i]:
                t = mm.vertices[j]-positiona
                v = np.dot(t,t)
                dir.append(t/np.sqrt(v))
                cur.append(np.dot(vnormal[i], t/v*2) )
            cur_node[i]['nb'] = neighbor[i]
            cur_node[i]['dir'] = dir
            cur_node[i]['cur'] = cur
        max_v = sys.float_info.min
        min_v = sys.float_info.max

        for i in range(len(vnormal)):
            graph_weight[i] = dict()
            for j in neighbor[i]:
                if np.linalg.norm(vnormal[i]) == 0 or np.linalg.norm(vnormal[j])==0:
                    graph_weight[i][j] = 1e-5
                    continue
                if i > j :
                    i_s_index = cur_node[i]['nb'].index(j)
                    j_s_index = cur_node[j]['nb'].index(i)
                    i_dirs = util.getDirs(cur_node[i]['dir'][i_s_index], vnormal[i], direction)
                    j_dirs = util.getDirs(cur_node[j]['dir'][j_s_index], vnormal[j], direction)

                    i_dirs = util.get_diffusion_Cur_first_order(i_dirs, cur_node[i]['dir'], cur_node[i]['cur'])
                    j_dirs = util.get_diffusion_Cur_first_order(j_dirs, cur_node[j]['dir'], cur_node[j]['cur'])

                    graph_weight[i][j] =  util.getDistance(i_dirs,j_dirs)

                    if graph_weight[i][j] > max_v:
                        max_v = graph_weight[i][j]
                    if graph_weight[i][j] < min_v:
                        min_v = graph_weight[i][j]
                    if graph_weight[i][j] >= 0:
                        pass
                    else:
                        print(graph_weight[i][j])
                        print((graph_weight[i][j] - min_v))
                        print((max_v - min_v))

        for i in range(len(vnormal)):
            for j in neighbor[i]:
                if i > j:
                    if graph_weight[i][j] >= 0:
                        pass
                    else:
                        print(graph_weight[i][j])
                        print((graph_weight[i][j] - min_v))
                        print((max_v - min_v))
                    graph_weight[i][j] = (graph_weight[i][j] - min_v)/(max_v - min_v)
                    if graph_weight[i][j] >= 0:
                        pass
                    else:
                        graph_weight[i][j] = min_v

        return graph_weight

    def calculateDiffCurv_weighted(self,mesh, direction):
        mm = mesh.data['mesh']
        weights = mesh.weights

        vnormal = mm.vertex_normals
        fnormal = mm.face_normals
        neighbor = mm.vertex_neighbors

        cur_node = dict()
        graph_weight = dict()

        for i in range(len(vnormal)):
            cur_node[i] = dict()
            positiona = mm.vertices[i]
            dir  = []
            cur  = []
            for j in neighbor[i]:
                t = mm.vertices[j]-positiona
                v = np.dot(t,t)
                dir.append(t/np.sqrt(v))
                cur.append(np.dot(vnormal[i], t/v*2) )
            cur_node[i]['nb'] = neighbor[i]
            cur_node[i]['dir'] = dir
            cur_node[i]['cur'] = cur

        for i in range(len(vnormal)):
            graph_weight[i] = dict()
            for j in neighbor[i]:
                if i > j :
                    i_s_index = cur_node[i]['nb'].index(j)
                    j_s_index = cur_node[j]['nb'].index(i)
                    i_dirs = util.getDirs(cur_node[i]['dir'][i_s_index], vnormal[i], direction)
                    j_dirs = util.getDirs(cur_node[j]['dir'][j_s_index], vnormal[j], direction)
                    i_dirs = util.get_diffusion_Cur_first_order(i_dirs, cur_node[i]['dir'], cur_node[i]['cur'])
                    j_dirs = util.get_diffusion_Cur_first_order(j_dirs, cur_node[j]['dir'], cur_node[j]['cur'])

                    i_weight = weights[i]
                    j_weight = weights[j]
                    graph_weight[i][j] =  util.getDistance(i_dirs,j_dirs) * max(i_weight,j_weight)

        return graph_weight

    def calculateDiffCurv_vertexFeature(self,mesh,direction):
        mm = mesh.data['mesh']
        vnormal = mm.vertex_normals
        fnormal = mm.face_normals
        neighbor = mm.vertex_neighbors

        cur_node = dict()
        graph_weight = dict()
        # mesh.vertexfeature = np.delete(mesh.vertexfeature, mesh.delete_point, axis=0)
        print(np.shape(mesh.vertexfeatures)[0],mesh.data['mesh'].vertices.shape[0])
        assert(np.shape(mesh.vertexfeatures)[0] == mesh.data['mesh'].vertices.shape[0])

        for i in range(len(vnormal)):
            cur_node[i] = dict()
            positiona = mm.vertices[i]
            dir  = []
            cur  = []
            for j in neighbor[i]:
                t = mm.vertices[j]-positiona
                v = np.dot(t,t)
                dir.append(t/np.sqrt(v))
                cur.append(np.dot(vnormal[i], t/v*2) )
            cur_node[i]['nb'] = neighbor[i]
            cur_node[i]['dir'] = dir
            cur_node[i]['cur'] = cur

        for i in range(len(vnormal)):
            graph_weight[i] = dict()
            for j in neighbor[i]:
                if i > j :
                    i_s_index = cur_node[i]['nb'].index(j)
                    j_s_index = cur_node[j]['nb'].index(i)
                    i_dirs = util.getDirs(cur_node[i]['dir'][i_s_index], vnormal[i], direction)
                    j_dirs = util.getDirs(cur_node[j]['dir'][j_s_index], vnormal[j], direction)
                    i_dirs = util.get_diffusion_Cur_first_order(i_dirs, cur_node[i]['dir'], cur_node[i]['cur'])
                    j_dirs = util.get_diffusion_Cur_first_order(j_dirs, cur_node[j]['dir'], cur_node[j]['cur'])

                    graph_weight[i][j] =  util.getDistance(i_dirs,j_dirs)

        for i in range(len(vnormal)):
            for j  in neighbor[i]:
                if i > j:
                    graph_weight[i][j] += util.cosineDistance(mesh.vertexfeatures[i][3:], mesh.vertexfeatures[j][3:])
                    # graph_weight[i][j] += util.euildDistance(mesh.vertexfeature[i][3:], mesh.vertexfeature[j][3:])
                    # graph_weight[i][j] /= 2

        return graph_weight

    def calculateDiffCurv_weighted_features(self,mesh, direction):
        mm = mesh.data['mesh']
        weights_label = mesh.weights
        scale = mesh.scale

        vnormal = mm.vertex_normals
        fnormal = mm.face_normals
        neighbor = mm.vertex_neighbors

        cur_node = dict()
        graph_weight = dict()
        # mesh.vertexfeature = np.delete(mesh.vertexfeature, mesh.delete_point, axis=0)
        print(np.shape(mesh.vertexfeatures)[0], mesh.data['mesh'].vertices.shape[0])
        assert (np.shape(mesh.vertexfeatures)[0] == mesh.data['mesh'].vertices.shape[0])

        max_v = sys.float_info.min
        min_v = sys.float_info.max

        for i in range(len(vnormal)):
            cur_node[i] = dict()
            positiona = mm.vertices[i]
            dir = []
            cur = []
            for j in neighbor[i]:
                t = mm.vertices[j] - positiona
                v = np.dot(t, t)
                dir.append(t / np.sqrt(v))
                cur.append(np.dot(vnormal[i], t / v * 2))
            cur_node[i]['nb'] = neighbor[i]
            cur_node[i]['dir'] = dir
            cur_node[i]['cur'] = cur

        for i in range(len(vnormal)):
            graph_weight[i] = dict()
            for j in neighbor[i]:
                if np.linalg.norm(vnormal[i]) == 0 or np.linalg.norm(vnormal[j]) == 0:
                    graph_weight[i][j] = 1e-5
                    continue
                if i > j:
                    i_s_index = cur_node[i]['nb'].index(j)
                    j_s_index = cur_node[j]['nb'].index(i)
                    i_dirs = util.getDirs(cur_node[i]['dir'][i_s_index], vnormal[i], direction)
                    j_dirs = util.getDirs(cur_node[j]['dir'][j_s_index], vnormal[j], direction)
                    i_dirs = util.get_diffusion_Cur_first_order(i_dirs, cur_node[i]['dir'], cur_node[i]['cur'])
                    j_dirs = util.get_diffusion_Cur_first_order(j_dirs, cur_node[j]['dir'], cur_node[j]['cur'])

                    i_weight = weights_label[i]
                    j_weight = weights_label[j]

                    graph_weight[i][j] = util.getDistance(i_dirs, j_dirs)
                    if graph_weight[i][j] > max_v:
                        max_v = graph_weight[i][j]
                    if graph_weight[i][j] < min_v:
                        min_v = graph_weight[i][j]

                    if i_weight == j_weight and i_weight ==1 :
                        graph_weight[i][j] *= scale
                    assert not np.isnan(graph_weight[i][j]) and not np.isinf(graph_weight[i][j])

        for i in range(len(vnormal)):
            for j in neighbor[i]:
                if i > j:
                    graph_weight[i][j] = (graph_weight[i][j] - min_v)/(max_v - min_v)

                    i_weight = weights_label[i]
                    j_weight = weights_label[j]
                    cosdis = util.cosineDistance(mesh.vertexfeatures[i][3:], mesh.vertexfeatures[j][3:])
                    if cosdis <= -0.0000001:
                        print(cosdis)
                    if cosdis <0 :
                        cosdis = abs(cosdis)
                    assert not  np.isnan(graph_weight[i][j]) and  not np.isinf(graph_weight[i][j])
                    if i_weight == j_weight and i_weight == 1:
                        graph_weight[i][j] += cosdis * scale
                    else:
                        graph_weight[i][j] += cosdis
                    # assert not  np.isnan(graph_weight[i][j]) and  not np.isinf(graph_weight[i][j])
                    # graph_weight[i][j] += util.euildDistance(mesh.vertexfeature[i][3:], mesh.vertexfeature[j][3:])
                    # graph_weight[i][j] /= 2
                    assert not np.isnan(graph_weight[i][j]) and not np.isinf(graph_weight[i][j])

        return graph_weight



    def calculateDiffCurv_weighted_features_balance(self,mesh, direction):
        mm = mesh.data['mesh']
        weights_label = mesh.weights
        scale = mesh.scale
        alpha = mesh.alpha

        vnormal = mm.vertex_normals
        fnormal = mm.face_normals
        neighbor = mm.vertex_neighbors

        cur_node = dict()
        graph_weight = dict()
        # mesh.vertexfeature = np.delete(mesh.vertexfeature, mesh.delete_point, axis=0)
        print(np.shape(mesh.vertexfeatures)[0], mesh.data['mesh'].vertices.shape[0])
        assert (np.shape(mesh.vertexfeatures)[0] == mesh.data['mesh'].vertices.shape[0])

        for i in range(len(vnormal)):
            cur_node[i] = dict()
            positiona = mm.vertices[i]
            dir = []
            cur = []
            for j in neighbor[i]:
                t = mm.vertices[j] - positiona
                v = np.dot(t, t)
                dir.append(t / np.sqrt(v))
                cur.append(np.dot(vnormal[i], t / v * 2))
            cur_node[i]['nb'] = neighbor[i]
            cur_node[i]['dir'] = dir
            cur_node[i]['cur'] = cur

        min_v = sys.float_info.max
        max_v = sys.float_info.min

        for i in range(len(vnormal)):
            graph_weight[i] = dict()
            for j in neighbor[i]:
                if i > j:
                    i_s_index = cur_node[i]['nb'].index(j)
                    j_s_index = cur_node[j]['nb'].index(i)
                    i_dirs = util.getDirs(cur_node[i]['dir'][i_s_index], vnormal[i], direction)
                    j_dirs = util.getDirs(cur_node[j]['dir'][j_s_index], vnormal[j], direction)
                    i_dirs = util.get_diffusion_Cur_first_order(i_dirs, cur_node[i]['dir'], cur_node[i]['cur'])
                    j_dirs = util.get_diffusion_Cur_first_order(j_dirs, cur_node[j]['dir'], cur_node[j]['cur'])

                    i_weight = weights_label[i]
                    j_weight = weights_label[j]

                    graph_weight[i][j] = util.getDistance(i_dirs, j_dirs)
                    if graph_weight[i][j] > max_v:
                        max_v = graph_weight[i][j]
                    if graph_weight[i][j] < min_v:
                        min_v = graph_weight[i][j]

                    if i_weight == j_weight and i_weight ==1 :
                        graph_weight[i][j] *= scale
                    graph_weight[i][j] += graph_weight[i][j] * alpha[0]

        for i in range(len(vnormal)):
            for j in neighbor[i]:
                if i > j:
                    i_weight = weights_label[i]
                    j_weight = weights_label[j]

                    graph_weight[i][j] = (graph_weight[i][j]-min_v) /(max_v-min_v)
                    if i_weight == j_weight and i_weight == 1:
                        graph_weight[i][j] +=  util.cosineDistance(mesh.vertexfeatures[i][3:], mesh.vertexfeatures[j][3:]) * scale * alpha[1]
                    else:
                        graph_weight[i][j] += util.cosineDistance(mesh.vertexfeatures[i][3:],
                                                                  mesh.vertexfeatures[j][3:]) * alpha[1]
                    # graph_weight[i][j] += util.euildDistance(mesh.vertexfeature[i][3:], mesh.vertexfeature[j][3:])
                    # graph_weight[i][j] /= 2

        return graph_weight




    def __call__(self, mesh,directions=2):
        data = self.calculate(mesh,directions)
        return data





if __name__ == "__main__":
    pass
    
    
