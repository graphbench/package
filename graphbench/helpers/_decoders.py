# Source for mis and max cut: https://github.com/WenkelF/copt/blob/main/utils/metrics.py

import torch
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.utils import unbatch, unbatch_edge_index, remove_self_loops


def mis_size(x: Tensor, batch: Batch, dec_length: int = 300, num_seeds: int = 1) -> Tensor:
    batch = mis_decoder(x, batch, dec_length, num_seeds)

    data_list = batch.to_data_list()

    size_list = [data.is_size for data in data_list]

    return Tensor(size_list).mean()


def mis_decoder(x: Tensor, batch: Batch, dec_length: int = 300, num_seeds: int = 1) -> Batch:
    x = torch.sigmoid(x)
    data_list = batch.to_data_list()
    x_list = unbatch(x, batch.batch)

    for data, x_data in zip(data_list, x_list):
        is_size_list = []

        for seed in range(num_seeds):

            order = torch.argsort(x_data, dim=0, descending=True)
            c = torch.zeros_like(x_data)

            edge_index = remove_self_loops(data.edge_index)[0]
            src, dst = edge_index[0], edge_index[1]

            c[order[seed]] = 1
            for idx in range(seed, min(dec_length, data.num_nodes)):
                c[order[idx]] = 1

                cTWc = torch.sum(c[src] * c[dst])
                if cTWc != 0:
                    c[order[idx]] = 0

            is_size_list.append(c.sum())

        data.is_size = max(is_size_list)

    return Batch.from_data_list(data_list)


def max_cut_size(x: Tensor, data: Batch) -> Tensor:
    x = (x > 0).float()
    x = (x - 0.5) * 2

    x_list = unbatch(x, data.batch)
    edge_index_list = unbatch_edge_index(data.edge_index, data.batch)

    cut_list = []
    for x, edge_index in zip(x_list, edge_index_list):
        cut_list.append(torch.sum(x[edge_index[0]] * x[edge_index[1]] == -1.0) / 2)

    return Tensor(cut_list).mean()


def num_colors_used(x: Tensor, batch: Batch, num_seeds: int = 1) -> Tensor:
    batch = graph_coloring_decoder(x, batch, num_seeds)

    data_list = batch.to_data_list()

    num_colors_used_list = []
    for data in data_list:
        num_colors_used = data.colors.unique().size(0)
        num_colors_used_list.append(num_colors_used)

    return torch.tensor(num_colors_used_list).mean(dtype=torch.float)


def graph_coloring_decoder(x: Tensor, batch: Batch, num_seeds: int = 1) -> Batch:
    max_num_colors = x.size(1)
    x = torch.sigmoid(x)
    data_list = batch.to_data_list()
    x_list = unbatch(x, batch.batch)

    for data, x_data in zip(data_list, x_list):
        edge_index = remove_self_loops(data.edge_index)[0]
        src, dst = edge_index[0], edge_index[1]

        best_colors = None
        min_colors_used = max_num_colors + 1  # upper bound

        for seed in range(num_seeds):
            order = torch.argsort(x_data.max(dim=1).values, descending=True)
            colors = torch.full((data.num_nodes,), -1, dtype=torch.long, device=x_data.device)

            for idx in range(data.num_nodes):
                node = order[(seed + idx) % data.num_nodes]
                # Find available colors for this node
                used = torch.zeros(max_num_colors, dtype=torch.bool, device=x_data.device)
                neighbors = dst[src == node]
                for neighbor in neighbors:
                    c = colors[neighbor]
                    if c >= 0:
                        used[c] = True

                # Assign the available color with the highest score in x_data for this node
                available_color_indices = (~used).nonzero(as_tuple=True)[0]
                max_idx = torch.argmax(x_data[node, available_color_indices])
                colors[node] = available_color_indices[max_idx]

            num_colors_used = colors.unique().size(0)
            if num_colors_used < min_colors_used:
                min_colors_used = num_colors_used
                best_colors = colors

        data.colors = best_colors

    return Batch.from_data_list(data_list)
