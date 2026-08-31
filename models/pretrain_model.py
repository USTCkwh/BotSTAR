import math
from typing import Union
import torch
import torch.nn as nn
import faiss
from .decoder import GNNDecoder, MLPDecoder, EdgeDecoder, RGTDecoder
from .encoder import MLPTwibot, MLPMGTAB, GNNTwibot, GNNMGTAB
from .utils import set_zeros, sce_loss
import torch.nn.functional as F


class PreModel(nn.Module):
    def __init__(self, dataset: str, input_dim: int, hidden_dim: int, num_heads_encoder: int, feat_drop: float,
                 attn_drop: float, feat_mask_rate: Union[int, str], replace_rate: float, leave_unchanged_rate: float,
                 cluster_num: int, interval: int, advanced_loss_rate:float, instance_prototype_loss_rate: float, align_loss_rate: float,
                 max_epoch: int, use_align_loss: bool = True):
        super(PreModel, self).__init__()
        self.output_hidden_size = hidden_dim
        self.replace_rate = replace_rate
        self.leave_unchanged = leave_unchanged_rate
        self.feat_mask_rate = feat_mask_rate
        assert hidden_dim % num_heads_encoder == 0
        # build gnn encoder
        if 'mgtab' in dataset.lower():
            self.gnn_encoder = GNNMGTAB(
                num_hidden=hidden_dim,
                nhead=num_heads_encoder,
                feat_drop=feat_drop,
                attn_drop=attn_drop,
            )
        else:
            self.gnn_encoder = GNNTwibot(
                num_hidden=hidden_dim,
                nhead=num_heads_encoder,
                feat_drop=feat_drop,
                attn_drop=attn_drop
            )
        # build gnn decoder
        self.gnn_decoder = GNNDecoder(
            num_hidden=hidden_dim,
            out_dim=input_dim,
            feat_drop=feat_drop
        )
        # build mlp encoder
        if 'mgtab' in dataset.lower():
            self.mlp_encoder = MLPMGTAB(hidden_dim=hidden_dim,
                                        numerical_feature_size=10,
                                        categorical_feature_size=10,
                                        tweet_feature_size=768)
        else:
            self.mlp_encoder = MLPTwibot(hidden_dim=hidden_dim,
                                         numerical_feature_size=5,
                                         categorical_feature_size=3,
                                         des_feature_size=768,
                                         tweet_feature_size=768)

        self.edge_homo_prob = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.color_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, 15)
        )
        # build edge decoder
        self.edge_decoder = EdgeDecoder(hidden_dim)
        self.rgt_decoder = RGTDecoder(hidden_dim, num_heads_encoder, feat_drop, attn_drop, input_dim)
        # build mlp decoder
        self.mlp_decoder = MLPDecoder(hidden_dim, input_dim)
        self.enc_mask_token = nn.Parameter(torch.zeros(1, input_dim))
        self.gnn_encoder_to_gnn_decoder = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mlp_encoder_to_mlp_decoder = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attr_criterion = nn.MSELoss()
        self.edge_criterion = nn.BCEWithLogitsLoss()

        self.cluster_num = cluster_num
        self.interval = interval
        self.advanced_loss_rate = advanced_loss_rate
        self.instance_prototype_loss_rate = instance_prototype_loss_rate
        self.max_iter_cluster = 200
        self.max_epoch = max_epoch
        self.v = 1.0
        self.project_head = nn.Linear(hidden_dim * 2, hidden_dim // 4)
        self.prototypes = nn.Parameter(torch.Tensor(self.cluster_num, hidden_dim // 4))
        torch.nn.init.xavier_normal_(self.prototypes.data)
        self.flag = True
        self.use_align_loss = use_align_loss
        self.align_loss_rate = align_loss_rate

    def target_distribution(self, q):
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def instance_prototype_loss(self, x, edge_index, edge_type, nclusters, max_iter, epoch, start_epoch=-1):
        if start_epoch == -1:
            start_epoch = self.max_epoch // 2
        epoch -= start_epoch
        z1 = self.gnn_encoder(x, edge_index, edge_type)
        z2 = self.mlp_encoder(x)
        z = torch.concat([z1, z2], dim=1)
        z = self.project_head(z)
        if self.flag:
            z_np = z.detach().cpu().numpy()
            kmeans = faiss.Kmeans(z.shape[1], nclusters, niter=max_iter, seed=0)
            kmeans.train(z_np)  # 训练 KMeans
            self.prototypes.data = torch.tensor(kmeans.centroids, dtype=torch.float32, device=z.device)
            self.flag = False
        if not hasattr(self, 'target') or epoch % self.interval == 0:
            q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.prototypes, 2), 2) / self.v)
            q = q.pow((self.v + 1.0) / 2.0)
            q = (q.t() / torch.sum(q, 1)).t()
            self.target = self.target_distribution(q.data)
        predicts = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.prototypes, 2), 2) / self.v)
        predicts = predicts.pow((self.v + 1.0) / 2.0)
        predicts = (predicts.t() / torch.sum(predicts, 1)).t()
        loss = F.kl_div(torch.log(predicts), self.target, reduction='batchmean')
        return loss

    def linear_attr_prediction(self, x):
        x_dropout = F.dropout(x, p=0.2, training=self.training)
        rep = self.mlp_encoder(x_dropout)
        rep = self.mlp_encoder_to_mlp_decoder(rep)
        recon = self.mlp_decoder(rep)
        loss = self.attr_criterion(recon, x)
        return loss

    def mask_attr_prediction(self, x, edge_index, edge_type=None, epoch=None):
        cur_feat_mask_rate = self.get_mask_rate(self.feat_mask_rate, epoch=epoch)
        use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(x, cur_feat_mask_rate)
        rep = self.gnn_encoder(use_x, edge_index, edge_type=edge_type, return_hidden=False)
        rep = self.gnn_encoder_to_gnn_decoder(rep)
        recon = self.gnn_decoder(rep)
        x_init = x[mask_nodes]
        x_recon = recon[mask_nodes]
        loss = self.attr_criterion(x_recon, x_init)

        return loss, rep

    def align_loss(self, x, edge_index, rep, edge_type=None, epoch=None):
        """
        # rep: 传入的通常是重度掩码视图 H_m 的表征，需要保留梯度进行更新。
        """

        # 1. L_msi: 重度掩码 vs 轻度掩码
        # 锚点是轻度掩码视图 rep1，需要 detach 截断梯度
        high_mask_rate = 0.8
        low_mask_rate = 0.4
        use_x1, (mask_nodes1, keep_nodes1) = self.encoding_mask_noise(x, high_mask_rate)
        use_x2, (mask_nodes2, keep_nodes2) = self.encoding_mask_noise(x, low_mask_rate)
        
        rep_high_mask = self.gnn_encoder(use_x1, edge_index, edge_type=edge_type, return_hidden=False)
        rep_low_mask = self.gnn_encoder(use_x2, edge_index, edge_type=edge_type, return_hidden=False)
        loss_al1 = -F.cosine_similarity(rep_high_mask, rep_low_mask.detach(), dim=-1).mean()

        # 2. L_usi: 当前掩码 vs 完整图
        # 锚点是完整图视图 rep2，需要 detach 截断梯度
        rep2 = self.gnn_encoder(x, edge_index, edge_type=edge_type, return_hidden=False)
        loss_al2 = -F.cosine_similarity(rep, rep2.detach(), dim=-1).mean()

        # 3. L_uti: 边丢弃 vs 完整图
        # 锚点是完整图视图 rep2，需要 detach 截断梯度
        edge_drop_rate = 0.7
        edge_perm = torch.randperm(edge_index.size(1), device=edge_index.device)
        drop_num = int(edge_drop_rate * edge_index.size(1))
        keep_edges = edge_perm[drop_num:]
        edge_index_kept = edge_index[:, keep_edges]
        if edge_type is not None:
            edge_type_kept = edge_type[keep_edges]
        else:
            edge_type_kept = None
        rep3 = self.gnn_encoder(x, edge_index_kept, edge_type=edge_type_kept, return_hidden=False)
        loss_al3 = -F.cosine_similarity(rep3, rep2.detach(), dim=-1).mean()

        loss_al = loss_al1 + loss_al2 #+ loss_al3
        return loss_al

    @torch.no_grad()
    def embed(self, x, edge_index, edge_type=None):
        gnn_rep = self.gnn_encoder(x, edge_index, edge_type)
        linear_rep = self.mlp_encoder(x)
        return gnn_rep, linear_rep

    def advanced_loss(self, x, edge_index, edge_type=None, beta=0.5):
        z1 = self.gnn_encoder(x, edge_index, edge_type)
        z2 = self.mlp_encoder(x)
        device = z1.device
        N = z1.size(0)
        D = z1.size(1)
        z1_norm = ((z1 - z1.mean(0)) / z1.std(0)) / math.sqrt(N)
        z2_norm = ((z2 - z2.mean(0)) / z2.std(0)) / math.sqrt(N)
        c1 = torch.mm(z1_norm.T, z1_norm)
        c2 = torch.mm(z2_norm.T, z2_norm)
        iden = torch.eye(D, device=device)
        loss_inv = (z1_norm - z2_norm).pow(2).sum()
        loss_dec_1 = (c1 - iden).pow(2).sum()
        loss_dec_2 = (c2 - iden).pow(2).sum()
        loss_dec = loss_dec_1 + loss_dec_2
        # # 微调阶段不使用线性编码了,这里就有必要对比学习
        # loss = loss_inv + beta * loss_dec
        # 不再要求两个编码器的输出是相似的
        loss = beta * loss_dec
        return loss

    def get_mask_rate(self, input_mask_rate, get_min=False, epoch=None):
        try:
            return float(input_mask_rate)
        except ValueError:
            if "~" in input_mask_rate:  # 0.6~0.8 Uniform sample
                mask_rate = [float(i) for i in input_mask_rate.split('~')]
                assert len(mask_rate) == 2
                if get_min:
                    return mask_rate[0]
                else:
                    return torch.empty(1).uniform_(mask_rate[0], mask_rate[1]).item()
            elif "," in input_mask_rate:  # 0.6,-0.1,0.4 stepwise increment/decrement
                mask_rate = [float(i) for i in input_mask_rate.split(',')]
                assert len(mask_rate) == 3
                start = mask_rate[0]
                step = mask_rate[1]
                end = mask_rate[2]
                if get_min:
                    return min(start, end)
                else:
                    cur_mask_rate = start + epoch * step
                    if cur_mask_rate < min(start, end) or cur_mask_rate > max(start, end):
                        return end
                    return cur_mask_rate
            else:
                raise NotImplementedError

    def encoding_mask_noise(self, x, mask_rate=0.3):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]
        perm_mask = torch.randperm(num_mask_nodes, device=x.device)
        num_leave_nodes = int(self.leave_unchanged * num_mask_nodes)
        num_noise_nodes = int(self.replace_rate * num_mask_nodes)
        num_real_mask_nodes = num_mask_nodes - num_leave_nodes - num_noise_nodes
        token_nodes = mask_nodes[perm_mask[: num_real_mask_nodes]]
        noise_nodes = mask_nodes[perm_mask[-num_noise_nodes:]]
        noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]
        out_x = x.clone()
        out_x = set_zeros(out_x, token_nodes)
        out_x[token_nodes] += self.enc_mask_token
        if num_noise_nodes > 0:
            out_x[noise_nodes] = x[noise_to_be_chosen]
        return out_x, (mask_nodes, keep_nodes)

    def forward(self, x, edge_index, edge_type=None, epoch=None):
        loss1, rep = self.mask_attr_prediction(x, edge_index, edge_type, epoch)
        loss = loss1

        loss_align = self.align_loss(x, edge_index, rep, edge_type, epoch)
        
        if epoch >= self.max_epoch // 2 and self.use_align_loss:
            loss = loss + loss_align * self.align_loss_rate
                

        loss2 = self.linear_attr_prediction(x)
        # loss = loss + loss2 

        start_epoch = 0
        loss4 = self.instance_prototype_loss(x, edge_index, edge_type, self.cluster_num, self.max_iter_cluster,
                                            epoch, start_epoch) * self.instance_prototype_loss_rate
        # if epoch >= start_epoch:
        #     loss = loss + loss4

        return loss
