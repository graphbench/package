import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import networkx as nx
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from ..parallelize_with_progress_bar import parallelize_with_progress_bar


class SyntheticDataset(InMemoryDataset, ABC):
    """
    An abstract base class for a dataset of synthetically generated graphs.

    Parameters:
    - `root`: The path to the directory that contains the dataset.
    - `num_samples`: The number of graphs in the dataset.
    - `pre_filter`: A function to filter graphs before saving.
    - `pre_transform`: A function to transform graphs before saving (applied after `pre_filter`).
    - `multiprocessing`: If True, the graphs will be generated in parallel.
    - `num_workers`: The number of subprocesses to use in case `multiprocessing == True`.
                     If `num_workers <= 0`, then the number of workers is set to the number of CPUs.
    """

    def __init__(
        self,
        root: str,
        num_samples: Optional[int] = None,
        pre_filter: Optional[Callable[[Data], bool]] = None,
        pre_transform: Optional[Callable[[Data], Data]] = None,
        multiprocessing: bool = False,
        num_workers: int = 0,
        **kwargs
    ):
        self.num_samples = num_samples
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform
        self.multiprocessing = multiprocessing
        self.num_workers = num_workers

        super().__init__(root=root, pre_filter=pre_filter, pre_transform=pre_transform, log=False, **kwargs)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list[str]:
        return []

    @property
    def processed_file_names(self) -> list[str]:
        return ['data.pt']

    def process(self):
        if self.num_samples is None:
            raise ValueError("num_samples cannot be None when generating a new dataset")

        print("Generating Graphs...")
        if self.multiprocessing:
            graphs = parallelize_with_progress_bar(self.create_graph, range(self.num_samples), self.num_workers)
        else:
            graphs = [self.create_graph(i) for i in tqdm(range(self.num_samples))]

        # separate PyG and NetworkX graphs
        graphs_pyg, graphs_nx = zip(*graphs)

        if self.pre_filter is not None:
            graphs_pyg = [data for data in graphs_pyg if self.pre_filter(data)]

        if self.pre_transform is not None:
            graphs_pyg = [self.pre_transform(data) for data in graphs_pyg]

        # save PyG dataset and NetworkX graphs separately
        self.save(graphs_pyg, self.processed_paths[0])
        with open(Path(self.processed_dir) / "graphs_nx.pickle", "wb") as file:
            pickle.dump(graphs_nx, file)

        print(f"Done generating, dataset saved. Root: {self.root}")

    @abstractmethod
    def create_graph(self, _index) -> tuple[Data, nx.Graph]:
        """Generates a single graph."""
        pass
