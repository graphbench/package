import random
from typing import Optional, Union

import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx

from .synthetic_dataset import SyntheticDataset


class ERDataset(SyntheticDataset):
    """
    A dataset of Erdős-Rényi graphs.

    See https://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93R%C3%A9nyi_model

    Dataset parameters:
    - num_nodes: The number of nodes in the graph, or a tuple specifying the range (min, max).
    - p: The probability that any given pair of nodes is connected by an edge.

    Graphs that are not connected are resampled.

    Also see the parameters and documentation of `SyntheticDataset`.
    """

    def __init__(
        self,
        root: str,
        num_samples: Optional[int] = None,
        num_nodes: Union[int, tuple[int, int]] = (700, 800),
        p: int = 0.15,
        **kwargs
    ):
        self.num_nodes = num_nodes
        self.p = p

        super().__init__(root, num_samples=num_samples, **kwargs)

    def create_graph(self, _index) -> tuple[Data, nx.Graph]:
        if isinstance(self.num_nodes, tuple) or isinstance(self.num_nodes, list):
            num_nodes = random.randint(*self.num_nodes)
        else:
            num_nodes = self.num_nodes

        graph_nx = nx.fast_gnp_random_graph(num_nodes, self.p)
        while not nx.is_connected(graph_nx):
            graph_nx = nx.fast_gnp_random_graph(num_nodes, self.p)

        graph_pyg = from_networkx(graph_nx)
        return graph_pyg, graph_nx
