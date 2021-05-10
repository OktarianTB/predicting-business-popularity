import pandas as pd
import networkx as nx
import torch
from sklearn.preprocessing import MinMaxScaler


def get_all_edges(df):
    user_friend = df[["user_id", "friends"]].values.tolist()
    edges = [
        (user, friend) for user, friends in user_friend for friend in friends.split(", ")
    ]

    print("number of edges before truncating:", len(edges))

    return edges


def get_user_features(df, columns):
    user_df = df[["user_id"] + columns]
    user_df = user_df.set_index("user_id")
    x = user_df.values
    scaled = MinMaxScaler().fit_transform(x)
    user_features = pd.DataFrame(scaled, columns=user_df.columns, index=user_df.index)
    print("number of nodes (users):", len(user_features))

    return user_features


def create_user_graph(user_features, edges):
    G = nx.Graph()

    for uid, feat in user_features.iterrows():
        G.add_node(uid, node_features=torch.Tensor(feat.values), node_type="user")

    G.add_edges_from(edges, edge_type="uu")  # uu means user to user

    # Remove friends that are not users anymore
    remove_nodes = set(list(G.nodes)) - set(user_features.index)

    G.remove_nodes_from(list(remove_nodes))

    print("number of edges after truncation:", len(G.edges))
    return G