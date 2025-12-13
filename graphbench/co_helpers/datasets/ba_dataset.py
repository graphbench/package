import random
from typing import Optional, Union

import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx

from .synthetic_dataset import SyntheticDataset


class BADataset(SyntheticDataset):
    """
    A dataset of Barabási-Albert graphs.

    See https://en.wikipedia.org/wiki/Barab%C3%A1si%E2%80%93Albert_model

    Dataset parameters:
    - `num_nodes`: The number of nodes in the graph, or a tuple specifying the range (min, max).
    - `m`: The number of edges to attach from a new node to existing nodes.

    Also see the parameters and documentation of `SyntheticDataset`.
    """

    def __init__(
        self,
        root: str,
        num_samples: Optional[int] = None,
        num_nodes: Union[int, tuple[int, int]] = (700, 800),
        m: int = 2,
        **kwargs
    ):
        self.num_nodes = num_nodes
        self.m = m

        super().__init__(root, num_samples=num_samples, **kwargs)

    def create_graph(self, _index) -> tuple[Data, nx.Graph]:
        if isinstance(self.num_nodes, tuple) or isinstance(self.num_nodes, list):
            num_nodes = random.randint(*self.num_nodes)
        else:
            num_nodes = self.num_nodes

        graph_nx = nx.barabasi_albert_graph(num_nodes, self.m)

        graph_pyg = from_networkx(graph_nx)
        return graph_pyg, graph_nx
