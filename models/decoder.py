import torch.nn as nn
import torch.nn.functional as F
from models.RGT import RGTLayer



class GNNDecoder(nn.Module):
    def __init__(self, num_hidden, out_dim, feat_drop):
        super(GNNDecoder, self).__init__()
        self.feat_drop = feat_drop
        self.activation = nn.PReLU()
        self.linear_input = nn.Linear(num_hidden, num_hidden, bias=False)
        self.linear_output = nn.Linear(num_hidden, out_dim, bias=False)
        self.head = nn.Identity()

    def forward(self, x):
        x = F.dropout(x, p=self.feat_drop, training=self.training)
        x = self.linear_output(x)
        return x

class RGTDecoder(nn.Module):
    def __init__(self,
                 num_hidden,
                 nhead,
                 feat_drop,
                 attn_drop,
                 out_dim
                ):
        super(RGTDecoder, self).__init__()
        self.feat_drop = feat_drop
        self.rgt_layer = RGTLayer(num_edge_type=2, in_channel=num_hidden, out_channel=num_hidden, trans_heads=nhead,
                               semantic_head=nhead, dropout=attn_drop)
        self.activation = nn.PReLU()
        self.norm = nn.BatchNorm1d(num_hidden)
        self.linear = nn.Linear(num_hidden, out_dim, bias=False)
    
    def forward(self, h, edge_index, edge_type=None, return_hidden=False):
        residual = h
        h = self.rgt_layer(h, edge_index, edge_type)
        h = self.norm(h)
        h = h + residual
        h = self.activation(h)
        h = self.linear(h)
        return h


class MLPDecoder(nn.Module):
    def __init__(self, num_hidden, out_dim):
        super(MLPDecoder, self).__init__()
        self.linear = nn.Linear(num_hidden, out_dim)

    def forward(self, x):
        x = self.linear(x)
        return x

class EdgeDecoder(nn.Module):
    def __init__(self, num_hidden):
        super(EdgeDecoder, self).__init__()
        self.edge_prob = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.PReLU(),
            nn.Linear(num_hidden, 1),
            nn.Sigmoid()
        )
        self.edge_type_pred = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.PReLU(),
            nn.Linear(num_hidden, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x_pairs):
        # x_pairs由节点对逐元素相乘得到，长度等同于x
        edge_prob = self.edge_prob(x_pairs)
        edge_type_pred = self.edge_type_pred(x_pairs)
        return edge_prob, edge_type_pred
