from torch.nn.parameter import Parameter
from torch.nn.init import *
from torch_geometric.utils import softmax
from torch import nn
import torch as t
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # attn = t.matmul(q / self.temperature, k.transpose(2, 3))
        attn = t.matmul(k.transpose(2, 3), q / self.temperature)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = t.matmul( v , attn).transpose(2,3)
        return output, attn
