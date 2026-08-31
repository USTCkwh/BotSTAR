import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import SemanticAttention
import copy


class Adaptation(nn.Module):
    def __init__(self, gnn_encoder, num_hidden, dropout, dataset, basevector_num, bvlen_factor=1,
                 edge_drop_ratio=0, edge_add_ratio=0.5):
        super(Adaptation, self).__init__()
        self.gnn_encoder_pretrained = copy.deepcopy(gnn_encoder)
        self.gnn_encoder = copy.deepcopy(gnn_encoder)
        self.dropout = dropout
        self.attention = SemanticAttention(num_hidden, 4)
        self.linear1 = nn.Linear(num_hidden, num_hidden // 4)
        self.linear2 = nn.Linear(num_hidden // 4, 2)
        self.activation = nn.PReLU()

        # 冻结 gnn_encoder_pretrained 的参数
        for param in self.gnn_encoder_pretrained.parameters():
            param.requires_grad = False

        # feature size
        if 'mgtab' in dataset.lower():
            self.feature_size = 788
        else:
            self.feature_size = 1544

        self.project = nn.Linear(self.feature_size, num_hidden)

        self.singleprompt = nn.Parameter(torch.randn(num_hidden))
        self.prompt_attention = SemanticAttention(num_hidden, 4)

        # 边扰动对齐相关：drop_ratio 删除原始边比例，add_ratio 加入虚构边比例
        self.edge_drop_ratio = edge_drop_ratio
        self.edge_add_ratio = edge_add_ratio

    def perturb_edges(self, edge_index, edge_type, num_nodes):
        """随机删除 drop_ratio 比例的原始边，再随机加入 add_ratio 比例的虚构边。"""
        num_edges = edge_index.size(1)
        # 删除：保留 (1 - drop_ratio) 比例的边
        keep = max(int(num_edges * (1 - self.edge_drop_ratio)), 1)
        perm = torch.randperm(num_edges, device=edge_index.device)[:keep]
        ei = edge_index[:, perm]
        et = edge_type[perm]
        # 加入：随机生成虚构边
        num_add = int(num_edges * self.edge_add_ratio)
        if num_add > 0:
            num_types = 2
            src = torch.randint(0, num_nodes, (num_add,), device=edge_index.device)
            dst = torch.randint(0, num_nodes, (num_add,), device=edge_index.device)
            ei_add = torch.stack([src, dst], dim=0)
            et_add = torch.randint(0, num_types, (num_add,), device=edge_index.device)
            ei = torch.cat([ei, ei_add], dim=1)
            et = torch.cat([et, et_add], dim=0)
        return ei, et

    def forward(self, g):
        x = g.x
        edge_index = g.edge_index
        edge_type = g.edge_type

        # 不要线性编码，保留注意力聚合
        a1 = self.gnn_encoder_pretrained(x, edge_index, edge_type)
        a2 = self.gnn_encoder(x, edge_index, edge_type)

        # 边扰动对齐：扰动图编码对齐到完整图编码（仅在训练时）
        if self.training and (self.edge_drop_ratio > 0 or self.edge_add_ratio > 0):
            ei_p, et_p = self.perturb_edges(edge_index, edge_type, x.size(0))
            a2_perturb = self.gnn_encoder(x, ei_p, et_p)
            align_loss = F.mse_loss(a2_perturb, a2.detach())
        else:
            align_loss = a2.new_zeros(())

        prompt = self.singleprompt.unsqueeze(0).expand(a1.shape[0], -1)  # (N, num_hidden)
        a1 = self.prompt_attention(torch.stack((a1, prompt), dim=1))  # (N, num_hidden)

        a1 = self.activation(a1)
        a2 = self.activation(a2)
        c = self.attention(torch.cat((a1.unsqueeze(1), a2.unsqueeze(1)), dim=1))
        c = self.linear1(c)
        c = self.activation(c)
        c = self.linear2(c)

        return c, align_loss