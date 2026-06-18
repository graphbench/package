# Source: https://github.com/WenkelF/copt/blob/main/graphgym/loss/copt_loss.py

import torch
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.utils import unbatch


def mis_loss(x: Tensor, batch: Batch, beta: float = 1.0) -> Tensor:
    x = torch.sigmoid(x)
    data_list = batch.to_data_list()
    x_list = unbatch(x, batch.batch)

    loss = 0.0
    for data, x_data in zip(data_list, x_list):
        src, dst = data.edge_index[0], data.edge_index[1]

        loss1 = torch.sum(x_data[src] * x_data[dst])
        loss2 = x_data.sum() ** 2 - loss1 - torch.sum(x_data ** 2)
        loss += (- loss2 + beta * loss1) * data.num_nodes

    return loss / batch.size(0)


def max_cut_loss(x: Tensor, batch: Batch) -> Tensor:
    x = torch.sigmoid(x)
    x = (x - 0.5) * 2
    src, dst = batch.edge_index[0], batch.edge_index[1]
    return torch.sum(x[src] * x[dst]) / len(batch.batch.unique())


# Adapted from GCON. GCON implements this for a node feature matrix X of size [num_nodes, num_colors] and an adjacency
# matrix A of size [num_nodes, num_nodes]. It basically calculates sum(diag(X^T A X)) - 4 * sum(abs(X)).
# The implementation here replaces the adjacency matrix with a pytorch geometric graph.
def graph_coloring_loss(x: Tensor, batch: Batch) -> Tensor:
    x = torch.sigmoid(x)
    x = (x - 0.5) * 2
    src, dst = batch.edge_index
    edge_loss = torch.sum(x[src] * x[dst])
    node_loss = 4 * torch.abs(x).sum()
    return edge_loss - node_loss
