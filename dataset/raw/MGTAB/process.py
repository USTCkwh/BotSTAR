import os.path
import numpy as np
import torch
from sklearn.utils import shuffle
from torch_geometric.data import Data


def sample_mask(idx, l):
    mask = torch.zeros(l)
    mask[idx] = 1
    return torch.as_tensor(mask, dtype=torch.bool)


def create_graph():
    # relation_dict = {
    #     0: 'followers',
    #     1: 'friends',
    #     2: 'mention',
    #     3: 'reply',
    #     4: 'quoted',
    #     5: 'url',
    #     6: 'hashtag'
    # }
    features = torch.load('features.pt')
    edge_index = torch.load('edge_index.pt')
    edge_type = torch.load('edge_type.pt')
    edge_index_followers = edge_index[:, edge_type == 0]
    edge_type_followers = edge_type[edge_type == 0]
    edge_index_friends = edge_index[:, edge_type == 1]
    edge_type_friends = edge_type[edge_type == 1]
    edge_index = torch.cat([edge_index_followers, edge_index_friends], dim=1)
    edge_type = torch.cat([edge_type_followers, edge_type_friends], dim=0)
    label = torch.load('label.pt')
    sample_number = len(label)
    seed = 0
    shuffled_idx = shuffle(np.array(range(sample_number)), random_state=seed)
    train_idx = shuffled_idx[:int(0.7 * sample_number)]
    val_idx = shuffled_idx[int(0.7 * sample_number):int(0.9 * sample_number)]
    test_idx = shuffled_idx[int(0.9 * sample_number):]
    train_mask = sample_mask(train_idx, sample_number)
    val_mask = sample_mask(val_idx, sample_number)
    test_mask = sample_mask(test_idx, sample_number)
    graph = Data(x=features,
                 edge_index=edge_index,
                 edge_type=edge_type,
                 y=label,
                 train_mask=train_mask,
                 test_mask=val_mask,
                 val_mask=test_mask)
    processed_dir = '../../processed/'
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    torch.save(graph, '../../processed/MGTAB-followers-friends.pt')


if __name__ == '__main__':
    create_graph()
