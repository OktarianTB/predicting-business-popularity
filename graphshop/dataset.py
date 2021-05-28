import torch
from torch.nn.utils.rnn import pad_sequence
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
                 processed_res_path,
                 graphshop_path,
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

        # 4. embedding features
        res_df = pd.read_csv(processed_res_path)
        res_df["top_categories_vector"] = res_df["top_categories_vector"].apply(
            lambda x: x.replace(".", ",")).apply(eval)

        cat_vec = np.array(res_df["top_categories_vector"].tolist())
        first_cat = []
        for row in cat_vec[:, 1:]:
            for i, item in enumerate(row):
                if item == 1:
                    first_cat.append(i)
                    break
            else:
                first_cat.append(1)

        res_df["category_idx"] = first_cat
        self.embedding_df = res_df.set_index("business_id")

        self.cat_vocab_len = 50
        self.city_vocab_len = self.embedding_df["city_idx"].max() + 1
        self.state_vocab_len = self.embedding_df["state_idx"].max() + 1

        # 5. graphshop neighborhood
        df = pd.read_csv(graphshop_path)
        df = df.set_index("Unnamed: 0")
        df.index = df.index.rename("business_id")
        for col in df.columns:
            df[col] = df[col].apply(eval)

        business_ids = []  # [num_business, 10, max_neighbors, 1]
        business_lens = []  # [num_business, 10,]
        distances = []
        for bus in df.values.tolist():
            business = []
            lengths = []
            dist = []
            for radius in bus:
                if len(radius) == 0:
                    neighbors = torch.LongTensor([[-1]])
                    _dist = torch.LongTensor([[0]])
                    lengths.append(1)
                else:
                    neighbors = torch.LongTensor(
                        [[self.res_node2idx[neigh[0]]] for neigh in radius])
                    _dist = torch.LongTensor([[neigh[1]] for neigh in radius])
                    lengths.append(len(radius))
                business.append(neighbors)
                dist.append(_dist)
            business_lens.append(lengths)
            business = pad_sequence(business, batch_first=True,
                                    padding_value=-1)  # [10, max_neighbors, 1]
            business = business.squeeze(-1).permute(1, 0)
            business_ids.append(business)  # [max_neighbors, 10]
            dist = pad_sequence(dist, batch_first=True,
                                padding_value=0)  # [10, max_neighbors, 1]
            dist = dist.squeeze(-1).permute(1, 0)  # [max_neighbors, 10]
            distances.append(dist)

        self.business_ids = pad_sequence(business_ids, batch_first=True, padding_value=-1)
        self.business_ids = self.business_ids.permute(0, 2, 1)  # [all, 10, max_neighbors]
        self.distances = pad_sequence(distances, batch_first=True, padding_value=0)
        self.distances = self.distances.permute(0, 2, 1)
        self.business_lens = torch.LongTensor(business_lens)

        # split
        dataset = GraphDataset(graphs=[self.res_pyg_graph], task='node')
        dataset_train, dataset_val, dataset_test = dataset.split(transductive=True,
                                                                 split_ratio=split,
                                                                 shuffle=True)
        self.train_index = dataset_train.graphs[0].node_label_index
        self.val_index = dataset_val.graphs[0].node_label_index
        self.test_index = dataset_test.graphs[0].node_label_index

        self.res_x = self.res_pyg_graph.node_feature
        self.res_x = torch.cat(  # last index as padding
            [self.res_x, torch.zeros(1, self.res_pyg_graph.num_node_features)],
            dim=0)
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

    def get_embs(self, target_res):
        target_res = [self.res_idx2node[int(res)] for res in target_res]

        cat_idx = torch.LongTensor(
            self.embedding_df.loc[target_res]["category_idx"].values)
        city_idx = torch.LongTensor(self.embedding_df.loc[target_res]["city_idx"].values)
        state_idx = torch.LongTensor(
            self.embedding_df.loc[target_res]["state_idx"].values)

        return cat_idx, city_idx, state_idx

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

    def get_batch(self, batch_size, mode="train"):
        if mode == "train":
            for i in range(0, self.train_index.shape[0], batch_size):
                max_idx = min(i + batch_size, self.train_index.shape[0])
                yield (self.train_index[i:max_idx],
                       self.business_ids[self.train_index[i:max_idx]],
                       self.distances[self.train_index[i:max_idx]],
                       self.business_lens[self.train_index[i:max_idx]],
                       self.labels[self.train_index[i:max_idx]])

        elif mode == "val":
            for i in range(0, self.val_index.shape[0], batch_size):
                max_idx = min(i + batch_size, self.val_index.shape[0])
                yield (self.val_index[i:max_idx],
                       self.business_ids[self.val_index[i:max_idx]],
                       self.distances[self.train_index[i:max_idx]],
                       self.business_lens[self.val_index[i:max_idx]],
                       self.labels[self.val_index[i:max_idx]])

        else:
            for i in range(0, self.test_index.shape[0], batch_size):
                max_idx = min(i + batch_size, self.test_index.shape[0])
                yield (self.test_index[i:max_idx],
                       self.business_ids[self.test_index[i:max_idx]],
                       self.distances[self.train_index[i:max_idx]],
                       self.business_lens[self.test_index[i:max_idx]],
                       self.labels[self.test_index[i:max_idx]])
