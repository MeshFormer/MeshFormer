from torch.nn.parameter import Parameter
from torch.nn import init
from torch_geometric.utils import softmax
from torch.nn.init import *
from torch import nn
import torch as t
import torch.nn.functional as F
from .util import ScaledDotProductAttention
from .MBlock import *



        
class MBLocks(nn.Module):
    def __init__(self, in_dim, n_hid, num_types, num_relations, n_heads, n_layers, dropout = 0.2,prev_norm = True, last_norm = True):
        super(MBLocks, self).__init__()
        self.mblocks = nn.ModuleList()
        self.num_types = num_types
        self.in_dim    = in_dim
        self.n_hid     = n_hid
        self.adapt_ws  = nn.ModuleList()
        self.drop      = nn.Dropout(dropout)
        for t in range(num_types):
            self.adapt_ws.append(nn.Linear(in_dim, n_hid))
        for l in range(n_layers - 1):
            self.mblocks.append(MBlock(n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm = prev_norm))
        self.mblocks.append(MBlock(n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm = last_norm))

    def forward(self, node_feature, node_type, edge_weight, edge_index, edge_type):
        res = torch.zeros(node_feature.size(0), self.n_hid).to(node_feature.device)
        for t_id in range(self.num_types):
            idx = (node_type == int(t_id))
            if idx.sum() == 0:
                continue
            res[idx] = torch.tanh(self.adapt_ws[t_id](node_feature[idx]))
        meta_xs = self.drop(res)
        attention_weight = None
        for mb in self.mblocks:
            meta_xs, attention_weight = mb(meta_xs, node_type, edge_index, edge_type, edge_weight)
        return meta_xs, attention_weight






class Point2FaceLayer(nn.Module):
    def __init__(self, node_feature_dim, num_graph_layer, outdim, attention_head, dropout=0.1):
        super(Point2FaceLayer, self).__init__()
        assert outdim > attention_head
        assert outdim % attention_head==0
        node_feature_dim = node_feature_dim * num_graph_layer
        self.kqv = Parameter(t.Tensor(3, node_feature_dim, 3, outdim))
        self.kqv_bias = Parameter(t.Tensor(3, node_feature_dim,1,outdim ))
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(outdim, eps=1e-6)
        self.head = attention_head
        self.outdim = outdim
        self.attention = ScaledDotProductAttention(temperature=node_feature_dim ** 0.5)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        xavier_uniform_(self.kqv[0])
        xavier_uniform_(self.kqv[1])
        xavier_uniform_(self.kqv[2])
        xavier_normal_(self.kqv_bias[0])
        xavier_normal_(self.kqv_bias[1])
        xavier_normal_(self.kqv_bias[2])

    def forward(self, node_feature: t.Tensor,faces : t.Tensor) -> t.Tensor:
        face_tensors = node_feature[faces].transpose(1,2).unsqueeze(-2)

        k = self.kqv[0]
        k_bias = self.kqv_bias[0]
        key = torch.einsum('bkij,kjt->bkit', face_tensors, k) + k_bias
        key = key.squeeze().view(len(face_tensors), node_feature.shape[1], self.head, self.outdim//self.head)


        q = self.kqv[1]
        q_bias = self.kqv_bias[1]
        query = torch.einsum('bkij,kjt->bkit', face_tensors, q) + q_bias
        query = query.squeeze().view(len(face_tensors), node_feature.shape[1], self.head, self.outdim // self.head)


        v = self.kqv[2]
        v_bias = self.kqv_bias[2]
        value = torch.einsum('bkij,kjt->bkit', face_tensors, v) + v_bias
        value = value.squeeze().view(len(face_tensors), node_feature.shape[1], self.head, self.outdim // self.head)
 
        q, k, v = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        q, attn = self.attention(q, k, v)
        q = q.transpose(1, 2).contiguous().view(len(face_tensors), node_feature.shape[1], -1)
        q = t.sum(q, dim=2)
        return q




class HierachyClassify(nn.Module):
    def __init__(self, n_hid, n_out, hierachy_layer):
        super(HierachyClassify, self).__init__()
        self.n_hid = n_hid*hierachy_layer
        self.n_out = n_out
        mid = (self.n_hid + self.n_out) // 2
        self.linear_1 = nn.Linear(self.n_hid, mid)
        self.linear_2 = nn.Linear(mid, n_out)
    def forward(self, x):
        tx = self.linear_1(x)
        ty = self.linear_2(tx)
        return torch.log_softmax(ty.squeeze(), dim=-1)

    def __repr__(self):
        return '{}(n_hid={}, n_out={})'.format(
            self.__class__.__name__, self.n_hid, self.n_out)




