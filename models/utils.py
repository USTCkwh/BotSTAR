import torch
import torch.nn.functional as F

class SemanticAttention(torch.nn.Module):
    def __init__(self, in_channel, num_head):
        super(SemanticAttention, self).__init__()

        self.in_channel = in_channel
        self.num_head = num_head
        self.multi_head_att_layer = torch.nn.Sequential(
            torch.nn.Linear(in_channel, num_head, bias=False),
            torch.nn.PReLU()
        )

    def forward(self, z):
        w = self.multi_head_att_layer(z)
        beta = torch.softmax(w, dim=1)
        beta = beta.unsqueeze(-1)
        z = z.unsqueeze(2)
        output = (beta * z).sum(dim=1)
        output = torch.mean(output, dim=1)
        return output


def set_zeros(tensor, index):
    coefficient = torch.ones(tensor.shape[0], device=tensor.device)
    coefficient[index] = 0
    tensor = tensor * coefficient.unsqueeze(-1)
    return tensor

def create_identity_edge_index_and_type(x):
    # 构造identity稀疏单位邻接矩阵，每条边出现2次
    # (0,0), (0,0), (1,1), (1,1), (2,2), (2,2),...,(N-1,N-1),(N-1,N-1)
    # iden_edge_index: tensor([[    0,     0,     1,  ..., 12544, 12545, 12545],
    #                          [    0,     0,     1,  ..., 12544, 12545, 12545]])
    iden_edge_index = torch.stack([torch.arange(x.size(0), device=x.device).repeat_interleave(2), 
                                torch.arange(x.size(0), device=x.device).repeat_interleave(2)], dim=0)

    # iden_edge_type: tensor([1, 0, 1,  ..., 0, 1, 0])
    iden_edge_type = torch.zeros(iden_edge_index.shape[1], dtype=torch.long, device=x.device)
    iden_edge_type[::2] = 1.0    # 这句的作用是将所有偶数索引的元素设为1
    return iden_edge_index, iden_edge_type

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss