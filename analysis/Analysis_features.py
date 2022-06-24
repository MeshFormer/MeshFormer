import sys
import argparse
import dash
import random 

import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import plotly.graph_objects as go

from util.util import *
from util.logger import Logger
from dash.dependencies import Input, Output


def export_meshs_feature_color(mesh_dir, size=None):
    mesh_paths = [os.path.join(mesh_dir, i)  for i in os.listdir(mesh_dir) if int(i) > -1]
    meshs = [] 
    features = [] 
    names = [] 
    
    for i in range(len(mesh_paths)):
        mp2 = load_cache(mesh_paths[i])
        meshs.append(mp2.mesh)
        features.append(mp2.mesh.vertexfeatures )
        names.append(mesh_paths[i].split('/')[-1])
        if len(meshs) > size:
            break
        
    vs = []
    fs = []
    ids = []
    count = 0 
    for i in range(len(meshs)):
        mesh = meshs[i]
        mm = mesh.data['mesh']
        vertices = mm.vertices
        faces = mm.faces
        
        
        vs.append(vertices)
        fs.append(faces)
        ids.append(count)
        count +=1

    
    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.P("Choose an feature:"),
        dcc.Dropdown(
            id='drop_id',
            options=[{'value': x, 'label': x}
                     for x in range(features[0].shape[1])],
            value=0,
            clearable=False
        ),
        html.P("Choose an object:"),
        dcc.Dropdown(
            id='dropdown',
            options=[{'value': ids[x], 'label': names[x]}
                     for x in range(len(ids))],
            value=ids[0],
            clearable=False
        ),
        dcc.Graph(id="graph"),
    ],{'width': '100%','height':'1024', 'vertical-align': 'middle'})
    
    
    
    @app.callback(
        Output("graph", "figure"),
        [Input("dropdown", "value"), Input("drop_id", "value")]
    )
    def display_mesh(name,feature_location):
        fig = go.Figure(go.Mesh3d(
            x=vs[name].T[0], y=vs[name].T[1], z=vs[name].T[2],
            i=fs[name].T[0], j=fs[name].T[1], k=fs[name].T[2],
            colorbar_title='z',
            colorscale=[[0, 'gold'],
                    [1, 'magenta']],
            intensity=features[name][:,feature_location].reshape(-1),
            showscale=True
        
        ))
        print('display mesh ', vs[name].shape[0], ' ', fs[name].shape[0])
        return fig

    app.run_server(debug=True,port=8051)
    

    
    
        


def parseParam():
    parser = argparse.ArgumentParser(description='Mesh simplify')
    parser.add_argument('--input', type=str, help='Please input the datadir. ' )
    parser.add_argument('--size', type=int, default=10, help='mesh size. ' )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parseParam()
    input_path = args.input
    export_meshs_feature_color(input_path, size = args.size)
