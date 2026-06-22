from typing import Optional

import torch
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.utils import unbatch_edge_index


# Source: https://github.com/WenkelF/copt/blob/main/utils/metrics.py
def mis_size(solutions: list[Tensor], batch: Optional[Batch] = None) -> Tensor:
    """
    Computes the average size of the given independent sets.

    :func:`~graphbench.helpers.combinatorial_optimization.mis_decoder` can be used to obtain maximum independent set
    (MIS) solutions from a model's soft output.
    Alternatively, :func:`~graphbench.helpers.combinatorial_optimization.validate_mis_solution` can be used to check if
    a given node subset is an independent set.

    Args:
        solutions: List of (valid) solutions to the MIS problem.
                   Each tensor of size ``[num_nodes]`` is a binary vector where a 1 indicates that the node is in the
                   independent set.
        batch: Ignored, argument is present for consistency (default ``None``).
    """
    size_list = [solution.sum() for solution in solutions]
    return torch.stack(size_list).mean()


# Source: https://github.com/WenkelF/copt/blob/main/utils/metrics.py
def max_cut_size(solutions: list[Tensor], batch: Batch) -> Tensor:
    """
    Computes the average size of the given cuts.

    :func:`~graphbench.helpers.combinatorial_optimization.max_cut_decoder` can be used to obtain max-cut solutions from
    a model's soft output.

    Args:
        solutions: List of (valid) solutions to the max-cut problem.
                   Each tensor of size ``[num_nodes]`` is a binary vector where a 1 indicates that the node is in one
                   partition and a 0 indicates that it is in the other partition.
        batch: Problem graphs.
    """
    # the docstring is purposefully not mentioning validate_max_cut_solution, since any binary partition is a valid
    # max cut solution

    edge_index_list = unbatch_edge_index(batch.edge_index, batch.batch)

    cut_list: list[Tensor] = []
    for assignments, edge_index in zip(solutions, edge_index_list):
        signed_assignments = assignments.float() * 2 - 1
        cut_list.append(torch.sum(signed_assignments[edge_index[0]] * signed_assignments[edge_index[1]] == -1.0) / 2)

    return torch.stack(cut_list).mean()


def num_colors_used(solutions: list[Tensor], batch: Optional[Batch] = None) -> Tensor:
    """
    Computes the average number of colors used by the given graph colorings.

    :func:`~graphbench.helpers.combinatorial_optimization.graph_coloring_decoder` can be used to obtain graph coloring
    solutions from a model's soft output.
    Alternatively, :func:`~graphbench.helpers.combinatorial_optimization.validate_graph_coloring_solution` can be used
    to check if a given coloring is valid.

    Args:
        solutions: List of (valid) solutions to the graph coloring problem.
                   Each tensor of size ``[num_nodes]`` contains integer color assignments for each node.
        batch: Ignored, argument is present for consistency (default ``None``).
    """
    num_colors_used_list = [colors.unique().size(0) for colors in solutions]
    return torch.tensor(num_colors_used_list).mean(dtype=torch.float)
