import torch
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.utils import unbatch, unbatch_edge_index

from ._decoders import graph_coloring_decoder, mis_decoder


# Source: https://github.com/WenkelF/copt/blob/main/utils/metrics.py
def mis_size(x: Tensor, batch: Batch, dec_length: int = 300, num_seeds: int = 1) -> Tensor:
    decoded_solutions = mis_decoder(x, batch, dec_length, num_seeds)
    size_list = [solution.sum() for solution in decoded_solutions]
    return torch.stack(size_list).mean()


# Source: https://github.com/WenkelF/copt/blob/main/utils/metrics.py
def max_cut_size(x: Tensor, data: Batch) -> Tensor:
    x = (x > 0).float()
    x = (x - 0.5) * 2

    x_list = unbatch(x, data.batch)
    edge_index_list = unbatch_edge_index(data.edge_index, data.batch)

    cut_list: list[Tensor] = []
    for x, edge_index in zip(x_list, edge_index_list):
        cut_list.append(torch.sum(x[edge_index[0]] * x[edge_index[1]] == -1.0) / 2)

    return torch.stack(cut_list).mean()


def num_colors_used(x: Tensor, batch: Batch, num_seeds: int = 1) -> Tensor:
    decoded_colorings = graph_coloring_decoder(x, batch, num_seeds)
    num_colors_used_list = [colors.unique().size(0) for colors in decoded_colorings]
    return torch.tensor(num_colors_used_list).mean(dtype=torch.float)
