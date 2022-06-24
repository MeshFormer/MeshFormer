import vtk
import nibabel as nb
import numpy as np
import trimesh
import meshio
import matplotlib.pyplot as plt
import networkx as nx
from util import util
import random
from tqdm import tqdm

def build_gemm(mesh, faces, face_areas):
    mesh.ve = [[] for _ in mesh.vs]
    edge_nb = []
    sides = []
    edge2key = dict()
    edges = []
    edges_count = 0
    nb_count = []
    for face_id, face in enumerate(faces):
        faces_edges = []
        for i in range(3):
            cur_edge = (face[i], face[(i + 1) % 3])
            faces_edges.append(cur_edge)
        for idx, edge in enumerate(faces_edges):
            edge = tuple(sorted(list(edge)))
            faces_edges[idx] = edge
            if edge not in edge2key:
                edge2key[edge] = edges_count
                edges.append(list(edge))
                edge_nb.append([-1, -1, -1, -1])
                sides.append([-1, -1, -1, -1])
                mesh.ve[edge[0]].append(edges_count)
                mesh.ve[edge[1]].append(edges_count)

                nb_count.append(0)
                edges_count += 1

        for idx, edge in enumerate(faces_edges):
            edge_key = edge2key[edge]
            edge_nb[edge_key][nb_count[edge_key]] = edge2key[faces_edges[(idx + 1) % 3]]
            edge_nb[edge_key][nb_count[edge_key] + 1] = edge2key[faces_edges[(idx + 2) % 3]]
            nb_count[edge_key] += 2
        for idx, edge in enumerate(faces_edges):
            edge_key = edge2key[edge]
            sides[edge_key][nb_count[edge_key] - 2] = nb_count[edge2key[faces_edges[(idx + 1) % 3]]] - 1
            sides[edge_key][nb_count[edge_key] - 1] = nb_count[edge2key[faces_edges[(idx + 2) % 3]]] - 2
    mesh.edges = np.array(edges, dtype=np.int32)
    mesh.gemm_edges = np.array(edge_nb, dtype=np.int64)
    mesh.sides = np.array(sides, dtype=np.int64)
    mesh.edges_count = edges_count


def fill_from_file(mesh, file):
    mesh.filename = ntpath.split(file)[1]
    mesh.fullfilename = file
    vs, faces = [], []
    f = open(file)
    for line in f:
        line = line.strip()
        splitted_line = line.split()
        if not splitted_line:
            continue
        elif splitted_line[0] == 'v':
            vs.append([float(v) for v in splitted_line[1:4]])
        elif splitted_line[0] == 'f':
            face_vertex_ids = [int(c.split('/')[0]) for c in splitted_line[1:]]
            assert len(face_vertex_ids) == 3
            face_vertex_ids = [(ind - 1) if (ind >= 0) else (len(vs) + ind)
                               for ind in face_vertex_ids]
            faces.append(face_vertex_ids)
    f.close()
    vs = np.asarray(vs)
    faces = np.asarray(faces, dtype=int)
    assert np.logical_and(faces >= 0, faces < len(vs)).all()
    return vs, faces

def compute_face_normals_and_areas(mesh, faces):
    face_normals = np.cross(mesh.vs[faces[:, 1]] - mesh.vs[faces[:, 0]],
                            mesh.vs[faces[:, 2]] - mesh.vs[faces[:, 1]])
    face_areas = np.sqrt((face_normals ** 2).sum(axis=1))
    face_normals /= face_areas[:, np.newaxis]
    assert (not np.any(face_areas[:, np.newaxis] == 0)), 'has zero area face: %s' % mesh.filename
    face_areas *= 0.5
    return face_normals, face_areas


def remove_non_manifolds(mesh, faces):
    mesh.ve = [[] for _ in mesh.vs]
    edges_set = set()
    mask = np.ones(len(faces), dtype=bool)
    _, face_areas = compute_face_normals_and_areas(mesh, faces)
    for face_id, face in enumerate(faces):
        if face_areas[face_id] == 0:
            mask[face_id] = False
            continue
        faces_edges = []
        is_manifold = False
        for i in range(3):
            cur_edge = (face[i], face[(i + 1) % 3])
            if cur_edge in edges_set:
                is_manifold = True
                break
            else:
                faces_edges.append(cur_edge)
        if is_manifold:
            mask[face_id] = False
        else:
            for idx, edge in enumerate(faces_edges):
                edges_set.add(edge)
    return faces[mask], face_areas[mask]
def get_side_points(mesh, edge_id):
    # if mesh.gemm_edges[edge_id, side] == -1:
    #     return mesh.get_side_points(edge_id, ((side + 2) % 4))
    # else:
    edge_a = mesh.edges[edge_id]

    if mesh.gemm_edges[edge_id, 0] == -1:
        edge_b = mesh.edges[mesh.gemm_edges[edge_id, 2]]
        edge_c = mesh.edges[mesh.gemm_edges[edge_id, 3]]
    else:
        edge_b = mesh.edges[mesh.gemm_edges[edge_id, 0]]
        edge_c = mesh.edges[mesh.gemm_edges[edge_id, 1]]
    if mesh.gemm_edges[edge_id, 2] == -1:
        edge_d = mesh.edges[mesh.gemm_edges[edge_id, 0]]
        edge_e = mesh.edges[mesh.gemm_edges[edge_id, 1]]
    else:
        edge_d = mesh.edges[mesh.gemm_edges[edge_id, 2]]
        edge_e = mesh.edges[mesh.gemm_edges[edge_id, 3]]
    first_vertex = 0
    second_vertex = 0
    third_vertex = 0
    if edge_a[1] in edge_b:
        first_vertex = 1
    if edge_b[1] in edge_c:
        second_vertex = 1
    if edge_d[1] in edge_e:
        third_vertex = 1
    return [edge_a[first_vertex], edge_a[1 - first_vertex], edge_b[second_vertex], edge_d[third_vertex]]


def get_edge_points(mesh):
    """ returns: edge_points (#E x 4) tensor, with four vertex ids per edge
        for example: edge_points[edge_id, 0] and edge_points[edge_id, 1] are the two vertices which define edge_id
        each adjacent face to edge_id has another vertex, which is edge_points[edge_id, 2] or edge_points[edge_id, 3]
    """
    edge_points = np.zeros([mesh.edges_count, 4], dtype=np.int32)
    for edge_id, edge in enumerate(mesh.edges):
        edge_points[edge_id] = get_side_points(mesh, edge_id)
        # edge_points[edge_id, 3:] = mesh.get_side_points(edge_id, 2)
    return edge_points

def set_edge_lengths(mesh, edge_points=None):
    if edge_points is not None:
        edge_points = get_edge_points(mesh)
    edge_lengths = np.linalg.norm(mesh.vs[edge_points[:, 0]] - mesh.vs[edge_points[:, 1]], ord=2, axis=1)
    mesh.edge_lengths = edge_lengths

def fixed_division(to_div, epsilon):
    if epsilon == 0:
        to_div[to_div == 0] = 0.1
    else:
        to_div += epsilon
    return to_div


def get_normals(mesh, edge_points, side):
    edge_a = mesh.vs[edge_points[:, side // 2 + 2]] - mesh.vs[edge_points[:, side // 2]]
    edge_b = mesh.vs[edge_points[:, 1 - side // 2]] - mesh.vs[edge_points[:, side // 2]]
    normals = np.cross(edge_a, edge_b)
    div = fixed_division(np.linalg.norm(normals, ord=2, axis=1), epsilon=0.1)
    normals /= div[:, np.newaxis]
    return normals

def get_opposite_angles(mesh, edge_points, side):
    edges_a = mesh.vs[edge_points[:, side // 2]] - mesh.vs[edge_points[:, side // 2 + 2]]
    edges_b = mesh.vs[edge_points[:, 1 - side // 2]] - mesh.vs[edge_points[:, side // 2 + 2]]

    edges_a /= fixed_division(np.linalg.norm(edges_a, ord=2, axis=1), epsilon=0.1)[:, np.newaxis]
    edges_b /= fixed_division(np.linalg.norm(edges_b, ord=2, axis=1), epsilon=0.1)[:, np.newaxis]
    dot = np.sum(edges_a * edges_b, axis=1).clip(-1, 1)
    return np.arccos(dot)


def get_ratios(mesh, edge_points, side):
    edges_lengths = np.linalg.norm(mesh.vs[edge_points[:, side // 2]] - mesh.vs[edge_points[:, 1 - side // 2]],
                                   ord=2, axis=1)
    point_o = mesh.vs[edge_points[:, side // 2 + 2]]
    point_a = mesh.vs[edge_points[:, side // 2]]
    point_b = mesh.vs[edge_points[:, 1 - side // 2]]
    line_ab = point_b - point_a
    projection_length = np.sum(line_ab * (point_o - point_a), axis=1) / fixed_division(
        np.linalg.norm(line_ab, ord=2, axis=1), epsilon=0.1)
    closest_point = point_a + (projection_length / edges_lengths)[:, np.newaxis] * line_ab
    d = np.linalg.norm(point_o - closest_point, ord=2, axis=1)
    return d / edges_lengths


def dihedral_angle(mesh, edge_points):
    normals_a = get_normals(mesh, edge_points, 0)
    normals_b = get_normals(mesh, edge_points, 3)
    dot = np.sum(normals_a * normals_b, axis=1).clip(-1, 1)
    angles = np.expand_dims(np.pi - np.arccos(dot), axis=0)
    return angles


def symmetric_opposite_angles(mesh, edge_points):
    """ computes two angles: one for each face shared between the edge
        the angle is in each face opposite the edge
        sort handles order ambiguity
    """
    angles_a = get_opposite_angles(mesh, edge_points, 0)
    angles_b = get_opposite_angles(mesh, edge_points, 3)
    angles = np.concatenate((np.expand_dims(angles_a, 0), np.expand_dims(angles_b, 0)), axis=0)
    angles = np.sort(angles, axis=0)
    return angles


def symmetric_ratios(mesh, edge_points):
    """ computes two ratios: one for each face shared between the edge
        the ratio is between the height / base (edge) of each triangle
        sort handles order ambiguity
    """
    ratios_a = get_ratios(mesh, edge_points, 0)
    ratios_b = get_ratios(mesh, edge_points, 3)
    ratios = np.concatenate((np.expand_dims(ratios_a, 0), np.expand_dims(ratios_b, 0)), axis=0)
    return np.sort(ratios, axis=0)

def extract_features(mesh):
    features = []
    edge_points = get_edge_points(mesh)
    set_edge_lengths(mesh, edge_points)
    with np.errstate(divide='raise'):
        try:
            for extractor in [dihedral_angle, symmetric_opposite_angles, symmetric_ratios]:
                feature = extractor(mesh, edge_points)
                features.append(feature)
            return np.concatenate(features, axis=0)
        except Exception as e:
            print(e)
            raise ValueError(mesh.filename, 'bad features')

def from_scratch(file):

    class MeshPrep:
        def __getitem__(self, item):
            return eval('self.' + item)

    mesh_data = MeshPrep()
    mesh_data.vs = mesh_data.edges = None
    mesh_data.gemm_edges = mesh_data.sides = None
    mesh_data.edges_count = None
    mesh_data.ve = None
    mesh_data.v_mask = None
    mesh_data.filename = 'unknown'
    mesh_data.edge_lengths = None
    mesh_data.edge_areas = []
    mesh_data.vs, faces = fill_from_file(mesh_data, file)
    mesh_data.v_mask = np.ones(len(mesh_data.vs), dtype=bool)
    faces, face_areas = remove_non_manifolds(mesh_data, faces)
    build_gemm(mesh_data, faces, face_areas)
    mesh_data.features = extract_features(mesh_data)
    return mesh_data


class Mesh:
    def __init__(self,path, labelpath=None,subdivide=True, isload=True):
        self.datapath = path
        self.subdivide = subdivide
        if  isload==True:
            self.data = self.load_mesh_geometry(path)
        else:
            self.data = dict() 
            
        self.data['nodedata'] = {}
        self.data['facedata'] = {}
        self.labelpath = labelpath


    def setNodeAttribute(self,name,data):
        assert len(data) == len(self.data['coords'])
        self.data['nodedata'][name] = data


    def getBox(self):
        return [
            np.min(self.data['coords'][:,0]),
            np.max(self.data['coords'][:,0]),
            np.max(self.data['coords'][:,1]),
            np.max(self.data['coords'][:,1]),
            np.max(self.data['coords'][:,2]),
            np.max(self.data['coords'][:,2])
        ]

    def setFaceAttribute(self,name,data):
        assert len(data) == len(self.data['coords'])
        self.data['facedata'][name] = data


    def convert_to_vtk_mesh(self):
        points = vtk.vtkPoints()
        triangles = vtk.vtkCellArray()
        for p in self.data['coords']:
            points.InsertNextPoint(p[0], p[1], p[2])

        for p in self.data['faces']:
            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0, p[0])
            triangle.GetPointIds().SetId(1, p[1])
            triangle.GetPointIds().SetId(2, p[2])
            triangles.InsertNextCell(triangle)

        trianglePolyData = vtk.vtkPolyData()
        trianglePolyData.SetPoints(points)
        trianglePolyData.SetPolys(triangles)

        return trianglePolyData



    # function to load mesh geometry
    def load_mesh_geometry(self,surf_mesh):
        mesh = None ;
        # if input is a filename, try to load it with nibabel
        if isinstance(surf_mesh, str):
            if (surf_mesh.endswith('orig') or surf_mesh.endswith('pial') or
                    surf_mesh.endswith('white') or surf_mesh.endswith('sphere') or
                    surf_mesh.endswith('inflated')):
                coords, faces = nb.freesurfer.io.read_geometry(surf_mesh)
            elif surf_mesh.endswith('gii'):
                coords, faces = \
                    nb.gifti.read(surf_mesh).getArraysFromIntent(nb.nifti1.intent_codes['NIFTI_INTENT_POINTSET'])[
                        0].data, \
                    nb.gifti.read(surf_mesh).getArraysFromIntent(nb.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE'])[
                        0].data
            elif surf_mesh.endswith('vtk'):
                coords, faces, mesh, _ = read_vtk(surf_mesh)
            elif surf_mesh.endswith('ply'):
                coords, faces, mesh = read_ply(surf_mesh)
            elif surf_mesh.endswith('obj'):
                coords, faces, mesh= read_obj(surf_mesh)
            elif isinstance(surf_mesh, dict):
                if ('faces' in surf_mesh and 'coords' in surf_mesh):
                    coords, faces = surf_mesh['coords'], surf_mesh['faces']
                else:
                    raise ValueError('If surf_mesh is given as a dictionary it must '
                                     'contain items with keys "coords" and "faces"')
            else:
                raise ValueError('surf_mesh must be a either filename or a dictionary '
                                 'containing items with keys "coords" and "faces"')
        return {'coords': np.array(coords), 'faces': np.array(faces),'mesh':mesh}


    # function to save mesh geometry
    def save_mesh_geometry(self,fname):
        surf_dict = self.data
        # if input is a filename, try to load it with nibabel
        if isinstance(fname, str) and isinstance(surf_dict, dict):
            if (fname.endswith('orig') or fname.endswith('pial') or
                    fname.endswith('white') or fname.endswith('sphere') or
                    fname.endswith('inflated')):
                nb.freesurfer.io.write_geometry(fname, surf_dict['coords'], surf_dict['faces'])
            #            save_freesurfer(fname,surf_dict['coords'],surf_dict['faces'])

            elif fname.endswith('vtk'):
                if 'data' in surf_dict.keys():
                    write_vtk(fname, surf_dict['mesh'],  surf_dict['data'])
                else:
                    write_vtk(fname, surf_dict['mesh'])
            elif fname.endswith('ply'):
                write_ply(fname, surf_dict['mesh'])
            elif fname.endswith('obj'):
                write_obj(fname, surf_dict['mesh'])
                print('to view mesh in brainview, run the command:\n')
                print('average_objects ' + fname + ' ' + fname)
        else:
            raise ValueError('fname must be a filename and surf_dict must be a dictionary')


    def show(self):
        self.data['mesh'].show()

    def calculateAsFaceColor(self,funcs):
        colors = funcs(self)
        assert colors.shape[0] == self.data['faces'].shape[0]
        if len(colors.shape) == 1 or colors.shape[1] == 1:
            Reds = plt.get_cmap('Reds')
            colors1 = np.zeros((colors.shape[0],3))
            for i in range(colors.shape[0]):
                colors1[i] = (np.array(Reds(colors[i]))*255)[:3]
        self.data['facecolor'] = colors1

    def calculateAsPointColor(self,funcs):
        colors = funcs(self)
        assert colors.shape[0] == self.data['coords'].shape[0]
        if len(colors.shape)==1  or colors.shape[1] == 1:
            Reds = plt.get_cmap('Reds')
            colors1 = np.zeros((colors.shape[0],3))
            for i in range(colors.shape[0]):
                colors1[i] = (np.array(Reds(colors[i]))*255)[:3]
        self.data['pointcolor'] = colors1


    def addLabelAsEdgeColor(self, path):
        seg_labels = np.loadtxt(open(path, 'r'), dtype='float64')
        meshdata = from_scratch(self.datapath)
        edgedata = dict()
        count = 0

        meshdataIdx2Trimesh = dict()
        for i in range(len(meshdata.vs)):
            index = self.data['mesh'].nearest.vertex(meshdata.vs[i])[1]
            meshdataIdx2Trimesh[i] = index

        for edge in meshdata.edges:
            edgedata[(meshdataIdx2Trimesh[int(edge[0])],meshdataIdx2Trimesh[int(edge[1])])] = seg_labels[count]
            edgedata[(meshdataIdx2Trimesh[int(edge[1])],meshdataIdx2Trimesh[int(edge[0])])] = seg_labels[count]
            count += 1

        colorsize = set(seg_labels)
        cormap = []
        for i in range(len(colorsize)+1):
            r = random.random()*255
            b = random.random()*255
            g = random.random()*255
            color = (r, g, b)
            cormap.append(color)

        self.data['facecolor'] = []
        self.data['edgelabel'] = edgedata
        for p in self.data['faces']:
            label1 = edgedata[(p[0],p[1])]
            label2 = edgedata[(p[1],p[2])]
            label3 = edgedata[(p[2],p[0])]

            facelabel = max([label1,label2,label3], key= [label1,label2,label3].count)
            # print(facelabel, ' is facelabel')
            # assert  [label1,label2,label3].count(facelabel) > 1
            if [label1,label2,label3].count(facelabel) <= 1 :
                print("error",[label1,label2,label3])
            self.data['facecolor'].append(cormap[int(facelabel)])
        self.data['facecolor'] = np.array(self.data['facecolor'])


    def translate(self, length):
        for i in range(len(self.data['coords'])):
            self.data['coords'][i][2] += length

    def showFaceLabel(self,facelabeldict):
        values = len(set(facelabeldict.values()))
        cormap = []
        for i in range(values+1):
            r = random.random()*255
            b = random.random()*255
            g = random.random()*255
            color = (r, g, b)
            cormap.append(color)

        self.data['facecolor'] = []
        for i in range(self.data['faces'].shape[0]):
            self.data['facecolor'].append(cormap[facelabeldict[i]])
        self.data['facecolor'] = np.array(self.data['facecolor'])


    def addLabelAsPointColor(self,path):
        label_for_nodes = util.read_node_label(path)
        dd = dict()
        for i in range(len(label_for_nodes)):
            index = self.data['mesh'].nearest.vertex(label_for_nodes[i][1])[1]
            dd[index] = label_for_nodes[i][0]
        colors = []
        for i in range(self.data['mesh'].vertices.shape[0]):
            colors.append(dd[i])

        cormap = []
        for i in range(10):
            r = random.random()*255
            b = random.random()*255
            g = random.random()*255
            color = (r, g, b)
            cormap.append(color)

        colors = np.array(colors)
        if len(colors.shape)==1  or colors.shape[1] == 1:
            colors1 = np.zeros((colors.shape[0],3))
            for i in range(colors.shape[0]):
                colors1[i] = cormap[int(colors[i])]
        self.data['pointcolor'] = colors1
        self.data['pointlabel'] = dd


    def calculateAsWeightGraph(self,funcs):
        gg = self.data['mesh'].vertex_adjacency_graph
        gg.remove_edges_from(nx.selfloop_edges(gg))
        graph_weight = funcs(self,12)
        for i in gg.nodes:
            for j in gg.neighbors(i):
                if i > j :
                    if graph_weight[i][j] >= 0:
                        gg[i][j]['weight'] = graph_weight[i][j]
                    else:
                        print("error ", graph_weight[i][j])


                    assert graph_weight[i][j] >= 0
                    # gg[i][j]['weight'] = graph_weight[i][j]
        return gg

    def subdivided(self,iter):
        self.newdata = dict()
        if self.subdivide:
            # trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max(sorted(mesh.edges_unique_length)) / 2)
            vert = self.data['mesh'].vertices
            face =  self.data['mesh'].faces
            for i in range(iter):
                vert, face = trimesh.remesh.subdivide(vert,face)
            self.newdata['mesh'] = trimesh.Trimesh(vert, face)
            mm = self.newdata['mesh']
            f_mm = self.data['mesh']
            from_p_label = self.data['pointlabel']
            # from_f_label = self.data['facelabel']
            from_e_label = self.data['edgelabel']

            new_p_label = dict()
            # new_f_label = dict()
            new_e_label = dict()

            for i in range(mm.vertices.shape[0]):
                ll = f_mm.nearest.vertex(mm.vertices[i])[1]
                new_p_label[i] = from_p_label[ll]

            for i in tqdm(range(mm.edges.shape[0])):
                firstlabel = f_mm.nearest.vertex(mm.vertices[mm.edges[i][0]])[1]
                secondlabel = f_mm.nearest.vertex(mm.vertices[mm.edges[i][1]])[1]
                if firstlabel == secondlabel:
                    new_e_label[(mm.edges[i][0],mm.edges[i][1])] = from_p_label[firstlabel]
                    new_e_label[(mm.edges[i][1],mm.edges[i][0])] = from_p_label[firstlabel]
                else:
                    if (firstlabel,secondlabel) in from_e_label.keys():
                        new_e_label[(mm.edges[i][0], mm.edges[i][1])] = from_e_label[(firstlabel,secondlabel)]
                        new_e_label[(mm.edges[i][1], mm.edges[i][0])] = from_e_label[(firstlabel,secondlabel)]
                    else:
                        new_e_label[(mm.edges[i][0], mm.edges[i][1])] = from_p_label[firstlabel]
                        new_e_label[(mm.edges[i][1], mm.edges[i][0])] = from_p_label[firstlabel]
            cormap = []
            for i in range(10):
                r = random.random() * 255
                b = random.random() * 255
                g = random.random() * 255
                color = (r, g, b)
                cormap.append(color)

            colors = []
            for i in range(self.newdata['mesh'].vertices.shape[0]):
                colors.append(new_p_label[i])

            colors = np.array(colors)
            if len(colors.shape) == 1 or colors.shape[1] == 1:
                colors1 = np.zeros((colors.shape[0], 3))
                for i in range(colors.shape[0]):
                    colors1[i] = cormap[int(colors[i])]
            self.newdata['pointcolor'] = colors1
            self.newdata['pointlabel'] = new_p_label

            self.newdata['facecolor'] = []
            self.newdata['edgelabel'] = new_e_label
            for p in mm.faces:
                label1 = new_e_label[(p[0], p[1])]
                label2 = new_e_label[(p[1], p[2])]
                label3 = new_e_label[(p[2], p[0])]

                facelabel = max([label1, label2, label3], key=[label1, label2, label3].count)
                # print(facelabel, ' is facelabel')
                # assert  [label1,label2,label3].count(facelabel) > 1
                if [label1, label2, label3].count(facelabel) <= 1:
                    print("error", [label1, label2, label3])
                self.newdata['facecolor'].append(cormap[int(facelabel)])
            self.newdata['facecolor'] = np.array(self.newdata['facecolor'])

            self.newdata['coords'] = mm.vertices
            self.newdata['faces'] = mm.faces
            self.newdata['nodedata'] = self.data['nodedata']
            self.newdata['facedata'] = self.data['facedata']
            self.data = None
            self.data = self.newdata


def read_obj(path):
    trimesh.util.attach_to_log()
    mesh = trimesh.load(path, force='mesh')
    return np.array(mesh.vertices), np.array(mesh.faces) , mesh

def write_obj(path,meshdata):
    vertices, cells = np.array(meshdata.vertices), np.array(meshdata.faces)
    faces = [("triangle", cells)]

    meshio.Mesh(
        vertices,
        faces
        # Optionally provide extra data on points, cells, etc.
        # point_data=point_data,
        # cell_data=cell_data,
        # field_data=field_data
    ).write(
        path,  # str, os.PathLike, or buffer/open file
        file_format="obj",  # optional if first argument is a path; inferred from extension
    )


def read_vtk(path):
    mesh = meshio.read(path)
    vertices = mesh.points
    faces = []
    for i in range(len(mesh.cells)):
        data = mesh.cells[i]
        faces += data.data.tolist()

    mesh = trimesh.Trimesh(vertices=vertices,
                           faces=faces,
                           process=False)
    return np.array(mesh.vertices), np.array(mesh.faces) , mesh


def write_vtk(path, trimesh_Data,data=None):
    vertices,cells = np.array(trimesh_Data.vertices), np.array(trimesh_Data.faces)
    faces = [("triangle", cells)]

    meshio.Mesh(
        vertices,
        faces
        # Optionally provide extra data on points, cells, etc.
        # point_data=point_data,
        # cell_data=cell_data,
        # field_data=field_data
    ).write(
        path,  # str, os.PathLike, or buffer/open file
        file_format="vtk",  # optional if first argument is a path; inferred from extension
    )




def read_ply(path):
    mesh = meshio.read(path)
    vertices = mesh.points
    faces = []
    for i in range(len(mesh.cells)):
        data = mesh.cells[i]
        faces += data.data.tolist()

    mesh = trimesh.Trimesh(vertices=vertices,
                           faces=faces,
                           process=False)
    return np.array(mesh.vertices), np.array(mesh.faces), mesh

def write_ply(path,trimesh_data):
    vertices,cells = np.array(trimesh_data.vertices), np.array(trimesh_data.faces)
    faces = [("triangle", cells)]

    meshio.Mesh(
        vertices,
        faces
        # Optionally provide extra data on points, cells, etc.
        # point_data=point_data,
        # cell_data=cell_data,
        # field_data=field_data
    ).write(
        path,  # str, os.PathLike, or buffer/open file
        file_format="ply",  # optional if first argument is a path; inferred from extension
    )



if __name__ == '__main__':
    pass
