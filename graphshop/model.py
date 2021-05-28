import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn


class MLP(nn.Module):

    def __init__(self, feature_dim, hidden_sizes, dropout=0.5, out_dim=3):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(feature_dim, hidden_sizes[0]))
        for i, hidden in enumerate(hidden_sizes):
            if i == 0:
                continue
            self.fcs.append(nn.Linear(hidden_sizes[i - 1], hidden))

    def forward(self, x):
        for layer in self.fcs:
            x = F.relu(self.dropout(layer(x)))

        return x


class GNN(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 is_final=False,
                 output_size=3,
                 num_layers=2,
                 model="GraphSage",
                 aggr="mean"):
        super().__init__()
        self.num_layers = num_layers
        self.is_final = is_final

        conv_model = self._build_conv_model(model)
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_size, hidden_size, aggr=aggr))
        for _ in range(self.num_layers - 1):
            self.convs.append(conv_model(hidden_size, hidden_size, aggr=aggr))

        if is_final:
            self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.leaky_relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        if self.is_final:
            return self.fc(x)

        return x

    def _build_conv_model(self, model_type="GraphSage"):
        if model_type == 'GCN':
            return pyg_nn.GCNConv
        if model_type == 'GAT':
            return pyg_nn.GATConv
        if model_type == "GraphSage":
            return pyg_nn.SAGEConv
        if model_type == "TransformerConv":
            return pyg_nn.TransformerConv

        raise ValueError(
            f"Model {model_type} unavailable, please add it to GNN.build_conv_model.")


class DistanceModule(nn.Module):

    def __init__(self,
                 user_feature_dim,
                 cat_vocab_len,
                 state_vocab_len,
                 city_vocab_len,
                 embedding_dim,
                 feature_dim,
                 hidden_sizes,
                 dropout=0.5,
                 out_dim=3):
        super().__init__()
        # self.cat_embedding = nn.Embedding(num_embeddings=cat_vocab_len,
        #                                   embedding_dim=embedding_dim)
        # self.state_embedding = nn.Embedding(num_embeddings=state_vocab_len,
        #                                     embedding_dim=embedding_dim)
        # self.city_embedding = nn.Embedding(num_embeddings=city_vocab_len,
        #                                    embedding_dim=embedding_dim)

        self.user_GNN = GNN(user_feature_dim, hidden_sizes[-1], aggr='max')
        self.distance_embedding = nn.Embedding(num_embeddings=10,
                                               embedding_dim=embedding_dim)

        self.mlp = MLP(feature_dim + embedding_dim, hidden_sizes)
        self.rnn = nn.LSTM(hidden_sizes[-1], hidden_sizes[-1], num_layers=3)

        self.target_mlp = MLP(feature_dim, hidden_sizes)

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_sizes[-1], hidden_sizes[-1])

        self.out = nn.Linear(hidden_sizes[-1] * 2, out_dim)

        self.distances = nn.Parameter(torch.arange(10), requires_grad=False)

    def forward(self,
                node_features,
                target_bus,
                bids,
                distances,
                lengths,
                dataset,
                device,
                user_x=None,
                u_adjs=None,
                inverse_idx=None,
                num_users_per_res=None):
        # bids = shape of [bs, 10, k]  k is the padded number
        # lengths = [bs, 10,]
        # all_bid = bids.reshape([-1])
        # all_bid = torch.unqiue(all_bid).to(device)

        # bid2idx = {int(bid): idx for idx, bid in enumerate(all_bid)}

        # cat_id, state_id, city_id = dataset.get_embs(all_bid)
        # # to device
        # cat_id = cat_id.to(device)
        # state_id = state_id.to(device)
        # city_id = city_id.to(device)

        # cat_emb = self.cat_embding(cat_id)  # [all_bid, emb dim]
        # state_emb = self.state_embedding(state_id)
        # city_emb = self.city_embedding(city_id)

        # distances = distances.to(device)
        # dist_emb = self.distance_embedding(distances)
        # dist_emb = self.distance_embedding(self.distances)  # [10, emb dim]
        # dist_emb = dist_emb.unsqueeze(1).repeat(1, bids.shape[2], 1)  # [10, k, emb dim]
        # # [bs, 10, k, emb dim]
        # dist_emb = dist_emb.unsqueeze(0).repeat(bids.shape[0], 1, 1, 1)

        # all_node_feat = node_features[all_bid]  # [all_bid, node feature dim]

        # all_feature = torch.cat([cat_emb, state_emb, city_emb, all_node_feat],
        #                         dim=-1)

        # feat = all_feature[bids]  # [bs, 10, k, feature dim]

        #########################################################################
        # lengths = lengths.to(device)
        # feat = node_features[bids]  # [bs, 10, k, feature_dim]
        # feat = torch.cat([feat, dist_emb], dim=-1)
        # feat = torch.flip(feat, dims=(1,))
        # feat = self.mlp(feat)
        # feat = feat.sum(dim=2)  # [bs, 10, feature dim]
        # feat = feat / lengths.unsqueeze(-1)

        # Finally, add the target business features
        target_features = node_features[target_bus]
        target_feat = self.target_mlp(target_features)
        # feat = torch.cat([feat, target_feat.unsqueeze(1)], dim=1)

        # seq_out, _ = self.rnn(feat.permute(1, 0, 2))  # [11, bs, feature dim]

        # far_embedding = seq_out[-2, :, :]
        # near_embedding = self.fc(self.dropout(seq_out[-1, :, :]))

        # return self.out(torch.cat([far_embedding, near_embedding], -1))
        user_x = self.user_GNN(user_x, u_adjs)  # [all_unique_users, hidden size]
        user_x = user_x[inverse_idx]  # [all users per network, hidden size]
        user_x = torch.split(
            user_x,
            num_users_per_res)  # batch size * [# of user per restaruant, hidden size]
        user_x = torch.stack([x.mean(dim=0) for x in user_x],
                             0)  # [batch size, hidden size]
        user_x = torch.nan_to_num(user_x)

        # return self.out(torch.cat([far_embedding, near_embedding, user_x], dim=-1))
        return self.out(torch.cat([target_feat, user_x], dim=-1))
