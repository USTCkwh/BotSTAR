import torch
from torch_geometric.nn import TransformerConv
from models.utils import SemanticAttention


def masked_edge_index(edge_index, edge_mask):
    return edge_index[:, edge_mask]


class RGTLayer(torch.nn.Module):
    def __init__(self, num_edge_type, in_channel, out_channel, trans_heads, semantic_head, dropout):
        super(RGTLayer, self).__init__()
        self.activation = torch.nn.PReLU()
        self.transformer_list = torch.nn.ModuleList()
        for i in range(int(num_edge_type)):
            self.transformer_list.append(
                TransformerConv(in_channels=in_channel, out_channels=out_channel//trans_heads, heads=trans_heads,
                                dropout=dropout,
                                concat=True
                                )
            )

        self.num_edge_type = num_edge_type
        self.semantic_attention = SemanticAttention(in_channel=out_channel, num_head=semantic_head)

    def forward(self, features, edge_index, edge_type):
        edge_index_list = []
        for i in range(self.num_edge_type):
            tmp = masked_edge_index(edge_index, edge_type == i)
            edge_index_list.append(tmp)

        u = self.transformer_list[0](features, edge_index_list[0].squeeze(0)).unsqueeze(1)
        for i in range(1, len(edge_index_list)):
            temp = self.transformer_list[i](features, edge_index_list[i].squeeze(0)).unsqueeze(1)
            u = torch.cat((u, temp), dim=1)
        v = self.semantic_attention(u)
        return v
