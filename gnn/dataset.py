import torch
from torch_geometric.data import Data
import networkx as nx
import numpy as np
import pandas as pd
from deepsnap.graph import Graph
from deepsnap.dataset import GraphDataset


class HybridDataset:

    def __init__(self,
                 res_graph_path,
                 user_graph_path,
                 user_per_res_path,
                 split=[0.8, 0.1, 0.1]):
        """
        res_graph_path = "../graphs/restaurants_with_categories.gpickle"
        user_graph_path = "../graphs/2017-2018_user_network.gpickle"
        user_per_res_path = "../datasets/2017-2018_visited_users.csv"
        """
        # 1. restaurant graphs
        self.res_G = nx.read_gpickle(res_graph_path)

        print(f"Number of restaurants: {self.res_G.number_of_nodes()}")
        print(f"Number of neighbors: {self.res_G.number_of_edges()}")

        self.res_idx2node = dict(enumerate(self.res_G.nodes()))
        self.res_node2idx = {node: idx for idx, node in self.res_idx2node.items()}

        print("converting restaurant graph to pyg graph...", end=" ")
        self.res_pyg_graph = Graph(self.res_G)
        self.res_pyg_graph.node_label = torch.LongTensor(self.res_pyg_graph.node_label)
        print("done!")

        # 2. user graph
        self.user_G = nx.read_gpickle(user_graph_path)

        print(f"Number of users: {self.user_G.number_of_nodes()}")
        print(f"Number of friends: {self.user_G.number_of_edges()}")

        self.user_idx2node = dict(enumerate(self.user_G.nodes()))
        self.user_node2idx = {node: idx for idx, node in self.user_idx2node.items()}

        print("converting restaurant graph to pyg graph...", end=" ")
        self.user_pyg_graph = Graph(self.user_G)
        print("done!")

        # 3. visited users per restaurant
        self.visited_user_df = pd.read_csv(user_per_res_path)
        self.visited_user_df.set_index("business_id", inplace=True)
        self.visited_user_df["user_ids"] = self.visited_user_df["user_ids"].apply(eval)

        self.max_k = self.visited_user_df["user_ids"].apply(len).max()
        print(self.max_k)

        # split
        dataset = GraphDataset(graphs=[self.res_pyg_graph], task='node')
        dataset_train, dataset_val, dataset_test = dataset.split(transductive=True,
                                                                 split_ratio=split,
                                                                 shuffle=True)
        self.train_index = dataset_train.graphs[0].node_label_index
        self.val_index = dataset_val.graphs[0].node_label_index
        self.test_index = dataset_test.graphs[0].node_label_index

        self.res_x = self.res_pyg_graph.node_feature
        self.user_x = self.user_pyg_graph.node_feature
        self.labels = self.res_pyg_graph.node_label

    def get_visited_users(self, target_res, k=5):
        target_res = [self.res_idx2node[int(res)] for res in target_res]
        users_per_res = self.visited_user_df.loc[target_res]["user_ids"].tolist(
        )  # list of list
        num_users_per_res = []
        visited_users = []
        for users in users_per_res:
            if k is not None:
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
        return self.res_pyg_graph.num_node_features

    @property
    def num_user_features(self):
        return self.user_pyg_graph.num_node_features

    @property
    def num_class(self):
        return self.res_pyg_graph.num_node_labels
