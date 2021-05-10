from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import networkx as nx
import numpy as np
import pandas as pd


def from_networkx(G):
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
        
    This code is forked from official torch_geometric.utils.convert.from_networkx,
    version 1.7.0
    """

    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    G = G.to_directed() if not nx.is_directed(G) else G
    edge_index = torch.LongTensor(list(G.edges)).t().contiguous()

    data = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        for key, value in feat_dict.items():
            if i == 0:
                data[str(key)] = [value]
            else:
                data[str(key)].append(value)

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        for key, value in feat_dict.items():
            if i == 0:
                data[str(key)] = [value]
            else:
                data[str(key)].append(value)
    
    for key, item in data.items():
        try:
            data[key] = torch.tensor(item)
        except ValueError:
            pass

    data['edge_index'] = edge_index.view(2, -1)
    data = Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()

    return data


class HybridDataset:
    def __init__(self, res_graph_path, user_graph_path, user_per_res_path, split=[0.8, 0.1, 0.1]):
        """
        res_graph_path = "../graphs/restaurants_with_categories.gpickle"
        user_graph_path = "../graphs/2017-2018_user_network.gpickle"
        user_per_res_path = "../datasets/2017-2018_visited_users.csv"
        """
        # 1. restaurant graphs
        self.res_G = nx.read_gpickle(res_graph_path)

        print(f"Number of restaurants: {self.res_G.number_of_nodes()}")
        print(f"Number of neighbors: {self.res_G.number_of_edges()}")

        self.res_idx2node = dict(enumerate(sorted(self.res_G.nodes())))
        self.res_node2idx = {node: idx for idx, node in self.res_idx2node.items()}
        
        print("converting restaurant graph to pyg graph...", end=" ")
        self.res_pyg_graph = from_networkx(self.res_G)  # to pytorch geometric format
        print("done!")

        # 2. user graph
        self.user_G = nx.read_gpickle(user_graph_path)

        print(f"Number of users: {self.user_G.number_of_nodes()}")
        print(f"Number of friends: {self.user_G.number_of_edges()}")

        self.user_idx2node = dict(enumerate(sorted(self.user_G.nodes())))
        self.user_node2idx = {node: idx for idx, node in self.user_idx2node.items()}
        
        print("converting restaurant graph to pyg graph...", end=" ")
        self.user_pyg_graph = from_networkx(self.user_G)  # to pytorch geometric format
        print("done!")

        # 3. visited users per restaurant
        self.visited_user_df = pd.read_csv(user_per_res_path)
        self.visited_user_df.set_index("business_id", inplace=True)
        self.visited_user_df["user_ids"] = self.visited_user_df["user_ids"].apply(eval)

        # split
        self.train_index, self.val_index, self.test_index = self.train_test_split(split)
        
        # features and labels
        self.res_x = torch.stack(self.res_pyg_graph.node_feature, 0)
        self.user_x = torch.stack(self.user_pyg_graph.node_features, 0)
        self.labels = self.res_pyg_graph.node_label.long()


    def train_test_split(self, split=[0.8, 0.1, 0.1]):
        train_nodes, test_nodes = train_test_split(
            np.arange(self.res_pyg_graph.node_label.shape[0]), test_size=split[1] + split[2], shuffle=True)
        val_nodes, test_nodes = train_test_split(test_nodes, test_size=split[2] / (split[1] + split[2]), shuffle=True)

        return [torch.LongTensor(x) for x in (train_nodes, val_nodes, test_nodes)]


    def get_visited_users(self, target_res, k=5):
        target_res = [self.res_idx2node[int(res)] for res in target_res]
        users_per_res = self.visited_user_df.loc[target_res]["user_ids"].tolist()  # list of list
        num_users_per_res = []
        visited_users = []
        for users in users_per_res:
            users = users[:k]  # sample first k users
            num_users_per_res.append(len(users))
            visited_users.extend(users)
        
        visited_users = [self.user_node2idx[user] for user in visited_users]

        return num_users_per_res, np.array(visited_users)
    
    def to(self, device):
        self.res_x = self.res_x.to(device)
        self.user_x = self.user_x.to(device)
        self.labels = self.labels.to(device)

    @property
    def num_res_features(self):
        return self.res_pyg_graph.node_feature[0].shape[0]
    
    @property
    def num_user_features(self):
        return self.user_pyg_graph.node_features[0].shape[0]
    
    @property
    def num_class(self):
        return self.res_pyg_graph.node_label.unique().shape[0]
