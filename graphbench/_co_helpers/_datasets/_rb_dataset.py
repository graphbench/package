import itertools
import random
from typing import Optional, Union

import networkx as nx
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx

from ._synthetic_dataset import SyntheticDataset


class RBDataset(SyntheticDataset):
    """
    A dataset of RB graphs.

    See
    Xu et al., "A simple model to generate hard satisfiable instances", 2005
    https://www.ijcai.org/Proceedings/05/Papers/0989.pdf

    Dataset parameters:
    - `num_cliques`: The number of cliques (integer, n >= 1, or tuple of two integers defining a range)
    - `k`: The number of nodes in each clique (integer, k >= 2, or tuple of two integers defining a range)
    - `p`: The tightness of each constraint.
      Regulates the interconnectedness between cliques, with a lower value corresponding to more connections between
      cliques
      (float, 1 >= p >= 0, or tuple of two floats defining a range)

    Optional parameters:
    - `num_nodes`: The number of nodes in the graph (tuple of two integers defining a range).
      Graphs will be re-sampled until the number of nodes falls into this range.
      Warning: Choosing this range carelessly can cause graph generation to get stuck in an endless loop of re-sampling.
    - `alpha`: Determines the domain size `d = n^alpha` of each variable (alpha > 0).
      Default: `log(k) / log(num_cliques)`
    - `r`: Determines the number `m = r * n * ln(n)` of constraints (r > 0). Default: `- alpha / log(1 - p)`
    """

    def __init__(
        self,
        root: str,
        num_samples: Optional[int] = None,
        num_cliques: Union[int, tuple[int, int]] = (20, 25),
        k: Union[int, tuple[int, int]] = (5, 12),
        p: Union[float, tuple[float, float]] = (0.3, 1),
        num_nodes: Optional[tuple[int, int]] = (200, 300),
        alpha: Optional[float] = None,
        r: Optional[float] = None,
        **kwargs
    ):
        self.num_cliques = num_cliques
        self.k = k
        self.p = p
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.r = r

        super().__init__(root, num_samples=num_samples, **kwargs)

    def create_graph(self, _index) -> tuple[Data, nx.Graph]:
        while True:
            num_cliques, k, r, p = self._prepare_parameters()

            sat_instance = generate_instance(num_cliques, k, r, p)
            graph_nx = nx.Graph()
            graph_nx.add_edges_from(sat_instance.clauses['NAND'])
            graph_nx.remove_nodes_from(list(nx.isolates(graph_nx)))

            if self.num_nodes is None \
                or self.num_nodes[0] <= graph_nx.number_of_nodes() <= self.num_nodes[1]:
                break

        graph_pyg = from_networkx(graph_nx)
        return graph_pyg, graph_nx

    def _prepare_parameters(self) -> tuple[int, int, float, float]:
        """
        Prepares the parameters `num_cliques`, `k`, `r`, and `p` necessary for sampling RB graphs.
        """
        # num_cliques
        if isinstance(self.num_cliques, int):
            num_cliques = self.num_cliques
        else:
            num_cliques = np.random.randint(self.num_cliques[0], self.num_cliques[1] + 1)

        # k
        if isinstance(self.k, int):
            k = self.k
        else:
            k = np.random.randint(self.k[0], self.k[1] + 1)

        # p
        if isinstance(self.p, float) or isinstance(self.p, int):
            p = self.p
        else:
            p = np.random.uniform(self.p[0], self.p[1])

        # alpha
        if self.alpha is not None:
            alpha = self.alpha
        else:
            alpha = np.log(k) / np.log(num_cliques)

        # r
        if self.r is not None:
            r = self.r
        else:
            r = - alpha / np.log(1 - p)

        return num_cliques, k, r, p



# Code below this point is adapted from
# https://github.com/WenkelF/copt/blob/f61ed0376bd3b74e15b1ddd2afd4bd5e78570e35/graphgym/loader/dataset/rb_dataset.py
def generate_instance(n: int, k: int, r: float, p: float) -> np.ndarray[tuple[int, int], np.dtype[np.int64]]:
    v = k * n
    a = np.log(k) / np.log(n)
    s = int(p * (n ** (2 * a)))
    iterations = int(r * n * np.log(n) - 1)

    parts = np.reshape(np.array(range(v)), (n, k))
    nand_clauses: list[tuple[np.int64, np.int64]] = []

    for i in parts:
        nand_clauses += itertools.combinations(i, 2)

    edges: set[tuple[np.int64, np.int64]] = set()
    for _ in range(iterations):
        i, j = np.random.choice(n, 2, replace=False)
        all_ = set(itertools.product(parts[i, :], parts[j, :]))
        all_ -= edges
        edges |= set(random.sample(tuple(all_), k=min(s, len(all_))))

    nand_clauses += list(edges)
    return np.array(nand_clauses)
