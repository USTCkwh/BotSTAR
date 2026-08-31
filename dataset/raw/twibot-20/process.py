import os

import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph


def sample_mask(idx, l):
    mask = torch.zeros(l)
    mask[idx] = 1
    return torch.as_tensor(mask, dtype=torch.bool)


def create_graph():
    cat_properties_tensor = torch.load('cat_properties_tensor.pt')
    des_tensor = torch.load('des_tensor.pt')
    num_properties_tensor = torch.load('num_properties_tensor.pt')
    tweets_tensor = torch.load('tweets_tensor.pt')
    x = torch.cat([cat_properties_tensor,num_properties_tensor,tweets_tensor,des_tensor], dim=1)
    edge_index = torch.load('edge_index.pt')
    edge_type = torch.load('edge_type.pt')
    label = torch.load('label.pt')
    train_idx = torch.load('train_idx.pt')
    test_idx = torch.load('test_idx.pt')
    val_idx = torch.load('val_idx.pt')
    num_hops = 100
    sub_node_idx, sub_edge_index, _, edge_mask = k_hop_subgraph(torch.cat([train_idx, test_idx, val_idx], dim=0),
                                                                num_hops=num_hops, edge_index=edge_index,
                                                                relabel_nodes=True)
    sample_number = len(sub_node_idx)
    train_mask = sample_mask(train_idx, sample_number)
    val_mask = sample_mask(val_idx, sample_number)
    test_mask = sample_mask(test_idx, sample_number)
    labels = torch.cat([label, 3 * torch.ones(sample_number - len(label)).long()], dim=0)
    graph = Data(x=x[sub_node_idx],
                 edge_index=sub_edge_index,
                 edge_type=edge_type[edge_mask],
                 y=labels,
                 train_mask=train_mask,
                 test_mask=test_mask,
                 val_mask=val_mask)
    processed_dir = '../../processed/'
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    torch.save(graph, '../../processed/Twibot-20-n-hops-subgraph.pt')

if __name__ == '__main__':
    create_graph()
