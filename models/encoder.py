import torch
import torch.nn as nn
import torch.nn.functional as F

from models.RGT import RGTLayer


class MLPTwibot(nn.Module):
    def __init__(self, hidden_dim, numerical_feature_size=5, categorical_feature_size=3, des_feature_size=768,
                 tweet_feature_size=768):
        super(MLPTwibot, self).__init__()
        self.numerical_feature_linear = nn.Sequential(
            nn.Linear(numerical_feature_size, hidden_dim // 4),
            nn.PReLU()
        )
        self.categorical_feature_linear = nn.Sequential(
            nn.Linear(categorical_feature_size, hidden_dim // 4),
            nn.PReLU()
        )
        self.des_feature_linear = nn.Sequential(
            nn.Linear(des_feature_size, hidden_dim // 4),
            nn.PReLU()
        )
        self.tweet_feature_linear = nn.Sequential(
            nn.Linear(tweet_feature_size, hidden_dim // 4),
            nn.PReLU()
        )
        self.total_feature_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU()
        )
        self.init_weights()

    def forward(self, x):
        category_prop = x[:, :3]
        num_prop = x[:, 3:8]
        des_tensor = x[:, 8:776]
        tweet_tensor = x[:, 776:]
        num_prop = self.numerical_feature_linear(num_prop)
        category_prop = self.categorical_feature_linear(category_prop)
        des_tensor = self.des_feature_linear(des_tensor)
        tweet_tensor = self.tweet_feature_linear(tweet_tensor)
        x = torch.cat((num_prop, category_prop, des_tensor, tweet_tensor), dim=1)
        x = self.total_feature_linear(x)
        return x

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()


class MLPMGTAB(nn.Module):
    def __init__(self, hidden_dim, numerical_feature_size, categorical_feature_size,
                 tweet_feature_size=768):
        super(MLPMGTAB, self).__init__()
        self.numerical_feature_linear = nn.Sequential(
            nn.Linear(numerical_feature_size, hidden_dim // 8),
            nn.PReLU()
        )
        self.categorical_feature_linear = nn.Sequential(
            nn.Linear(categorical_feature_size, hidden_dim // 8),
            nn.PReLU()
        )
        self.des_feature_linear = nn.Sequential(
            nn.Linear(tweet_feature_size, 3 * hidden_dim // 4),
            nn.PReLU()
        )
        self.total_feature_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU()
        )
        self.init_weights()

    def forward(self, x):
        d = self.des_feature_linear(x[:, -768:])
        n = self.numerical_feature_linear(x[:, [3, 5, 6, 7, 9, 10, 11, 12, 13, 14]])
        c = self.categorical_feature_linear(x[:, [0, 1, 2, 4, 8, 15, 16, 17, 18, 19]])
        h = torch.cat([d, n, c], dim=1)
        h = self.total_feature_linear(h)
        return h

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()


class GNNTwibot(nn.Module):
    def __init__(self,
                 num_hidden,
                 nhead,
                 feat_drop,
                 attn_drop
                 ):
        super(GNNTwibot, self).__init__()
        self.feat_drop = feat_drop
        self.activation1 = nn.PReLU()
        self.activation2 = nn.PReLU()
        self.norm = nn.BatchNorm1d(num_hidden)
        self.layer1 = RGTLayer(num_edge_type=2, in_channel=num_hidden, out_channel=num_hidden, trans_heads=nhead,
                               semantic_head=nhead, dropout=attn_drop)
        self.layer2 = RGTLayer(num_edge_type=2, in_channel=num_hidden, out_channel=num_hidden, trans_heads=nhead,
                               semantic_head=nhead, dropout=attn_drop)
        self.numerical_feature_linear = nn.Sequential(
            nn.Linear(5, num_hidden // 4),
            nn.PReLU()
        )
        self.categorical_feature_linear = nn.Sequential(
            nn.Linear(3, num_hidden // 4),
            nn.PReLU()
        )

        self.des_feature_linear = nn.Sequential(
            nn.Linear(768, num_hidden // 4),
            nn.PReLU()
        )

        self.tweet_feature_linear = nn.Sequential(
            nn.Linear(768, num_hidden // 4),
            nn.PReLU()
        )

        self.total_feature_linear = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.PReLU()
        )

    def forward(self, x, edge_index, edge_type=None, return_hidden=False):
        category_prop = x[:, :3]
        num_prop = x[:, 3:8]
        des_tensor = x[:, 8:776]
        tweet_tensor = x[:, 776:]
        num_prop = self.numerical_feature_linear(num_prop)
        category_prop = self.categorical_feature_linear(category_prop)
        des_tensor = self.des_feature_linear(des_tensor)
        tweet_tensor = self.tweet_feature_linear(tweet_tensor)
        h = torch.cat([num_prop, category_prop, des_tensor, tweet_tensor], dim=1)
        h = self.total_feature_linear(h)
        h = F.dropout(h, p=self.feat_drop, training=self.training)
        residual = h
        h = self.layer1(h, edge_index, edge_type)
        h = self.activation1(h)
        h = self.norm(h)
        h = self.layer2(h, edge_index, edge_type)
        h = h + residual
        h = self.activation2(h)
        return h


class GNNMGTAB(nn.Module):
    def __init__(self,
                 num_hidden,
                 nhead,
                 feat_drop,
                 attn_drop,
                 num_edge_type=2
                 ):
        super(GNNMGTAB, self).__init__()
        self.feat_drop = feat_drop
        self.activation1 = nn.PReLU()
        self.activation2 = nn.PReLU()
        self.norm = nn.BatchNorm1d(num_hidden)
        self.layer1 = RGTLayer(num_edge_type=num_edge_type, in_channel=num_hidden, out_channel=num_hidden,
                               trans_heads=nhead,
                               semantic_head=nhead, dropout=attn_drop)
        self.layer2 = RGTLayer(num_edge_type=num_edge_type, in_channel=num_hidden, out_channel=num_hidden,
                               trans_heads=nhead,
                               semantic_head=nhead, dropout=attn_drop)
        self.numerical_feature_linear = nn.Sequential(
            nn.Linear(10, num_hidden // 8),
            nn.PReLU()
        )
        self.categorical_feature_linear = nn.Sequential(
            nn.Linear(10, num_hidden // 8),
            nn.PReLU()
        )
        self.des_feature_linear = nn.Sequential(
            nn.Linear(768, 3 * num_hidden // 4),
            nn.PReLU()
        )
        self.total_feature_linear = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.PReLU()
        )

    def forward(self, x, edge_index, edge_type=None, return_hidden=False):
        d = self.des_feature_linear(x[:, -768:])
        n = self.numerical_feature_linear(x[:, [3, 5, 6, 7, 9, 10, 11, 12, 13, 14]])
        c = self.categorical_feature_linear(x[:, [0, 1, 2, 4, 8, 15, 16, 17, 18, 19]])
        h = torch.cat([d, n, c], dim=1)
        h = self.total_feature_linear(h)
        h = F.dropout(h, p=self.feat_drop, training=self.training)
        residual = h
        h = self.layer1(h, edge_index, edge_type)
        h = self.activation1(h)
        h = self.norm(h)
        h = self.layer2(h, edge_index, edge_type)
        h = h + residual
        h = self.activation2(h)
        return h