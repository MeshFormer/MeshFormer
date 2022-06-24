from visual.visual import  *
from util.util import *

def render_ricci_flow_result(mesh_path):
    hgt = load_cache(mesh_path)
    mesh = hgt.mesh
    clustering_result = [hgt.cc[0][1][i] for i in range(len(mesh.data['mesh'].vertices))]
    render_vertice_color(mesh.data['mesh'], clustering_result)

if __name__=='__main__':
    render_ricci_flow_result("")