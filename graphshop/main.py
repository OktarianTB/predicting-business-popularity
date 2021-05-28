import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch_geometric.data import NeighborSampler, Data
import torch_geometric.nn as pyg_nn
from tqdm.notebook import tqdm
import numpy as np
from dataset import HybridDataset
from model import DistanceModule


def train(model, num_epochs, dataset, optimizer, criterion, device="cpu", k=5):
    for epoch in range(num_epochs):
        model.train()

        for target_bus, business_ids, distances, business_lens, labels in dataset.get_batch(
                2048, "train"):
            num_users_per_res, users = dataset.get_visited_users(target_bus, k)
            unique_users, inverse_idx = np.unique(users, return_inverse=True)
            unique_users = torch.LongTensor(unique_users)

            user_loader = NeighborSampler(dataset.user_pyg_graph.edge_index,
                                          node_idx=unique_users,
                                          sizes=[-1, -1],
                                          batch_size=unique_users.shape[0],
                                          shuffle=False)

            for _, u_id, u_adjs in user_loader:
                # this is actually just one loop: using for loop since
                # next(iter(user_loader)) appears to be buggy
                u_adjs = [adj.to(device) for adj in u_adjs]

            del user_loader

            business_ids = business_ids.to(device)
            business_lens = business_lens.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            out = model(dataset.res_x,
                        target_bus,
                        business_ids,
                        distances,
                        business_lens,
                        dataset,
                        device,
                        user_x=dataset.user_x[u_id],
                        u_adjs=u_adjs,
                        inverse_idx=inverse_idx,
                        num_users_per_res=num_users_per_res)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            train_acc = test(model, dataset, device, "train", k=k)
            val_acc = test(model, dataset, device, "val", k=k)
            test_acc = test(model, dataset, device, "test", k=k)
            print(f'epoch: {epoch}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                  f'Test: {test_acc:.4f}')


@torch.no_grad()
def test(model, dataset, device, mode, k=5):
    outs = []
    model.eval()
    y_true = []
    for target_bus, business_ids, distances, business_lens, labels in dataset.get_batch(
            2048, mode):

        num_users_per_res, users = dataset.get_visited_users(target_bus, k)
        unique_users, inverse_idx = np.unique(users, return_inverse=True)
        unique_users = torch.LongTensor(unique_users)

        user_loader = NeighborSampler(dataset.user_pyg_graph.edge_index,
                                      node_idx=unique_users,
                                      sizes=[-1, -1],
                                      batch_size=unique_users.shape[0],
                                      shuffle=False)

        for _, u_id, u_adjs in user_loader:
            # this is actually just one loop: using for loop since
            # next(iter(user_loader)) appears to be buggy
            u_adjs = [adj.to(device) for adj in u_adjs]

        del user_loader

        business_ids = business_ids.to(device)
        business_lens = business_lens.to(device)

        out = model(dataset.res_x,
                    target_bus,
                    business_ids,
                    distances,
                    business_lens,
                    dataset,
                    device,
                    user_x=dataset.user_x[u_id],
                    u_adjs=u_adjs,
                    inverse_idx=inverse_idx,
                    num_users_per_res=num_users_per_res)
        outs.append(out.cpu())
        y_true.append(labels.cpu())

    outs = torch.cat(outs, dim=0)
    y_true = torch.cat(y_true, dim=0)
    y_pred = outs.argmax(dim=-1, keepdim=False)

    results = int(y_pred.eq(y_true).sum()) / int(y_true.shape[0])

    return results


def main():
    basedir = "/data/schoiaj/repos/predicting-business-popularity"
    dataset = HybridDataset(f"{basedir}/graphs/restaurants_user_influence2.gpickle",
                            f"{basedir}/graphs/2017-2018_user_network.gpickle",
                            f"{basedir}/datasets/2017-2018_visited_users.csv",
                            f"restaurant_processed.csv",
                            f"graphshop_neighborhood.csv",
                            split=[0.8, 0.1, 0.1])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset.to(device)
    model = DistanceModule(dataset.num_user_features,
                           dataset.cat_vocab_len,
                           dataset.state_vocab_len,
                           dataset.city_vocab_len,
                           16,
                           dataset.num_res_features, [48, 64, 64, 16],
                           dropout=0.3,
                           out_dim=dataset.num_class).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    train(model, 100, dataset, optimizer, loss_fn, device=device, k=5)


if __name__ == "__main__":
    main()