import torch
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.utils import unbatch, unbatch_edge_index

from ._decoders import graph_coloring_decoder, mis_decoder


# Source: https://github.com/WenkelF/copt/blob/main/utils/metrics.py
def mis_size(x: Tensor, batch: Batch, dec_length: int = 300, num_seeds: int = 1) -> Tensor:
    batch = mis_decoder(x, batch, dec_length, num_seeds)

    data_list = batch.to_data_list()

    size_list = [data.is_size for data in data_list]

    return Tensor(size_list).mean()


# Source: https://github.com/WenkelF/copt/blob/main/utils/metrics.py
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
