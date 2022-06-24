import os
import numpy as np
import math
# import pickle
import pickle
import gc
import argparse
import dill
import trimesh as tri
import scipy.spatial.distance as dis 


from visual.visual import *


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def normlength(v):
    return np.dot(v,v)


def normalize(v, axis=None):
    if axis == None:
        norm = np.linalg.norm(v)
    elif axis==1:
        norm = np.linalg.norm(v, axis=axis).reshape(-1, 1)
    elif axis==0:
        norm = np.linalg.norm(v, axis=axis).reshape(1, -1)
    if axis != None and 0 in norm:
        return v
    if(type(norm)==np.float64 and norm==0.0):
        norm = norm + 1e-8
    if (type(norm)!=np.float64 and 0.0 in norm):
        index = np.where(norm==0.0)
        norm[index] = 1e-8
    return v / norm


def getPathName(path):
    aa = path.rfind('/')
    if aa == -1:
        aa = path.rfind('\\')
        if aa == -1 :
            return ""

    name = path[aa+1:]
    name = name.split('.')[0]
    return name

def getFileEnd(path):
    aa = path.rfind('.')
    if aa == -1 :
        return ""

    name = path[aa+1:]
    return name

def getStartDir(dir,norm):
    k = np.cross(dir,norm)
    start = np.cross(norm,k)
    start = normalize(start)
    return start

def getDirs(dir, norm,i):
    assert i%4 == 0 and 360 % i ==0 
    start = getStartDir(dir,norm)
    # print(start)
    norm = normalize(norm)
    clock_0 = start
    clock_90 = np.cross(start,norm)
    clock_180 = np.cross(clock_90,norm)
    clock_270 = np.cross(clock_180,norm)

    dirs = []

    interval = 360 / i
    for k in range(i):
        xu = np.sin(math.radians(k*interval))
        yv = np.cos(math.radians(k*interval))
        dirs.append(xu*clock_90 + yv*clock_0)

    return dirs


def get_diffusion_Cur(normal_dirs, curdirs, cur):
    normal_dirs = np.array(normal_dirs)
    curdirs = np.array(curdirs)
    cur = np.array(cur)
    # N * 3
    normal_dirs = normalize(normal_dirs, axis=1)
    # M * 3
    curdirs = normalize(curdirs, axis = 1 )
    # N * M
    dotres = np.matmul(normal_dirs, curdirs.T)
    dotres = dotres*dotres
    dotres = normalize(dotres, axis= 0 )
    # N * 1
    newcur = np.matmul(dotres, cur.reshape(-1, 1))
    return newcur


def get_diffusion_Cur_first_order(normal_dirs, curdirs, cur):
    normal_dirs = np.array(normal_dirs)
    curdirs = np.array(curdirs)
    cur = np.array(cur)

    # N * 3
    normal_dirs = normalize(normal_dirs, axis=1)
    # M * 3
    curdirs = normalize(curdirs, axis = 1 )
    # N * M
    dotres = np.matmul(normal_dirs, curdirs.T)
    # dotres = dotres*dotres
    dotres = np.clip(dotres,0,None)
    dotres = normalize(dotres, axis= 0 )
    # N * 1
    newcur = np.matmul(dotres, cur.reshape(-1, 1))
    return newcur

def getDistance(dirs1, dirs2):
    assert len(dirs1) == len(dirs2)
    dirs1_t = dirs1[:int(len(dirs1)/2)]
    dirs1_tt = dirs1[int(len(dirs1)/2):]
    newdirs1 = dirs1_tt + dirs1_t 
    
    dis = 0.0 
    for i in range(len(newdirs1)):
        dis += normlength(newdirs1[i]-dirs2[i])
    return dis/len(dirs1)

def render_mesh_color(points,faces,ylabels):
    mm = tri.Trimesh(points, faces, process=False)
    label_dict = {}
    labels = ylabels.argsort()[:, -1].reshape(-1).tolist()[0]
    for i,j in enumerate(labels):
        label_dict[i] = j
    render_face_color(mm, label_dict)

def euildDistance(a, b):
    return dis.euclidean(a, b)

def cosineDistance(a, b):
    return dis.cosine(a,b)

def read_node_label(path):
    seg_labels = np.loadtxt(open(path, 'r'),dtype='float64')
    result = dict()
    for i in seg_labels:
        result[i[0]] = [i[-1],[i[1],i[2],i[3]]]
    return result

def save_cache_dill(obj, path):
    with open(path,'wb') as f:
        dill.dump(obj,f)

def load_cache_dill( path):
    with open(path, 'rb') as f:
        return dill.load(f)

class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "pyHGT.data":
            renamed_module = "MeshTransformer.HeteroDireGraph"
        if name == 'HeteoGraph':
            name = 'HeteoDirectedGraph'
        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


def save_cache(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_cache(path):
    with open(path, "rb") as f:
        return renamed_load(f)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


    
if __name__ == '__main__':
    getDirs(np.array([1,0,0]),np.array([0,0,1]),12)
