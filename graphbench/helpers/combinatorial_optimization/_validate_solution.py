from typing import Optional

from torch import Tensor
from torch_geometric.data import Data


def validate_mis_solution(graph: Data, solution: Tensor) -> bool:
    """
    Checks whether the given solution is a valid independent set for the provided graph.
    That is, no two nodes in the set are adjacent to each other.

    Note that solutions created by :func:`~graphbench.helpers.combinatorial_optimization.mis_decoder` are guaranteed to
    be valid, so validating them again isn't necessary.

    Args:
        graph: The problem graph.
        solution: The independent set, as a binary vector where a 1 indicates that the node is in the set.
                  Size ``[graph.num_nodes]``.
    """
    # check if solution is a binary vector of correct length
    if solution.size() != (graph.num_nodes,) or not ((solution == 0) | (solution == 1)).all():
        return False

    # for each edge, see if both the source node and the destination node are in the set
    src = graph.edge_index[0]
    dst = graph.edge_index[1]
    # this is non-zero if both nodes are in the set
    both_in_set = solution[src] * solution[dst]

    # if any edge connects two nodes in the set, it is not a valid independent set
    return not both_in_set.any()


def validate_max_cut_solution(graph: Optional[Data] = None, solution: Optional[Tensor] = None) -> bool:
    """
    Always returns ``True``, since any node subset defines a valid cut.

    Args:
        graph: Ignored, argument is present for consistency.
        solution: Ignored, argument is present for consistency.
    """
    return True


def validate_chrom_solution(graph: Data, solution: Tensor) -> bool:
    """
    Checks whether the given solution is a valid graph coloring for the provided graph.
    That is, no two adjacent nodes are assigned the same color.

    Note that solutions created by :func:`~graphbench.helpers.combinatorial_optimization.graph_coloring_decoder` are
    guaranteed to be valid, so validating them again isn't necessary.

    Args:
        graph: The problem graph.
        solution: The graph coloring, as a vector where each entry indicates the color assigned to the corresponding
                  node. Size ``[graph.num_nodes]``.
    """
    if solution.size() != (graph.num_nodes,):
        return False

    # for each edge, see if both nodes were assigned the same color
    src = graph.edge_index[0]
    dst = graph.edge_index[1]
    same_color = solution[src] == solution[dst]

    # if any edge connects two nodes with the same color, then this is not a valid graph coloring
    return not same_color.any()
