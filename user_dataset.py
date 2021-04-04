import pandas as pd
import pickle
import random

class UserDataset:
    def __init__(self, feature_file_path, graph_path):
        self.user_features = pd.read_csv(feature_file_path, index_col="user_id")

        with open(graph_path, "rb") as f:
            self.user_graph = pickle.load(f)

    def get_feats(self, user_ids):
        """user_ids: user id or list of user ids."""
        return self.user_features.loc[user_ids].values

    def sample_neighbors(self, user_id, num_neighbor_samples=5, return_feats=False):
        """
        user_id: specific user id in string.
        num_neighbor_samples: number of neighbors to sample for the user.
        return_feats: whether to return neighboring nodes as their features.
            Default = False, meaning it returns the ids of the neighbor nodes.
        """
        sampled_neighbors = random.choices(
            list(self.user_graph[user_id]), k=num_neighbor_samples)

        return self.get_feats(sampled_neighbors) if return_feats else sampled_neighbors
