from typing import Optional

import torch
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.utils import unbatch_edge_index


# Source: https://github.com/WenkelF/copt/blob/main/utils/metrics.py
def mis_size(solutions: list[Tensor], batch: Optional[Batch] = None) -> Tensor:
    size_list = [solution.sum() for solution in solutions]
    return torch.stack(size_list).mean()


# Source: https://github.com/WenkelF/copt/blob/main/utils/metrics.py
def max_cut_size(solutions: list[Tensor], batch: Batch) -> Tensor:
    edge_index_list = unbatch_edge_index(batch.edge_index, batch.batch)

    cut_list: list[Tensor] = []
    for assignments, edge_index in zip(solutions, edge_index_list):
        signed_assignments = assignments.float() * 2 - 1
        cut_list.append(torch.sum(signed_assignments[edge_index[0]] * signed_assignments[edge_index[1]] == -1.0) / 2)

    return torch.stack(cut_list).mean()


def num_colors_used(solutions: list[Tensor], batch: Optional[Batch] = None) -> Tensor:
    num_colors_used_list = [colors.unique().size(0) for colors in solutions]
    return torch.tensor(num_colors_used_list).mean(dtype=torch.float)
