import os
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Union

from torch_geometric.data import Data, InMemoryDataset

from graphbench._algoreas_helpers import generate_algoreas_data
from graphbench._helpers import download_and_unpack, SourceSpec, get_logger


# (i) helper functions

# -----------------------------------------------------------------------------#
# (a) Utilities
# -----------------------------------------------------------------------------#

_logger = get_logger(__name__)


class AlgoReasDataset(InMemoryDataset):
    """
    Algorithmic reasoning (AlgoReas) datasets.

    Note:
        This class should only be used directly when generating new datasets.
        To access provided datasets, please consider using :class:`graphbench.Loader`.
        The sections below give details on the data available through the :class:`graphbench.Loader` interface.


    Overview:
        On the algorithmic reasoning datasets, the goal is to predict the solution of a classical graph problem ("task")
        given an input graph.

        We provide datasets for 7 different tasks:

        - Minimum spanning tree
        - Maximum clique
        - Topological sort
        - Maximum flow
        - Bipartite matching
        - Bridge finding
        - Steiner tree

        For each of these, we provide graphs on three difficulty levels, easy, medium, and hard.

        The provided datasets include a mixture of graphs from different graph generators, depending on the difficulty
        setting.
        In total, we used six graph generators:

        - Erdős-Rényi
        - Newman-Watts-Strogatz
        - Barabási-Albert
        - Dual Barabási-Albert
        - Powerlaw Cluster
        - Stochastic Block Model

        The parameters used for each generator depend on the algorithmic reasoning task;
        please refer to the `GraphBench paper <https://arxiv.org/abs/2512.04475>`__ for the exact values used and for
        further graph generation details.
        Our datasets combine the generators as follows.

        .. list-table::
           :header-rows: 1

           * - Difficulty
             - Training
             - Validation/Testing
           * - Easy
             - [All 6 generators]
             - [All 6 generators]
           * - Medium
             - Erdős-Rényi, Barabási-Albert, Stochastic Block Model
             - [All 6 generators]
           * - Hard
             - Erdős-Rényi
             - [All 6 generators]

        Training and validation graphs contain 16 nodes each, while test graphs contain 128 nodes.

        We also provide additional size generalization datasets for each task, with 192, 256, 384, and 512 nodes each.
        Note that these only contain test graphs, and the training and validation graphs from the main datasets should
        be used for training.


    Graph Attributes:
        The features and additional information available with each graph depend on the algorithmic reasoning task.
        Note that if available, node features (``data.x``) are always of size ``[num_nodes, 1]`` and edge features
        (``data.edge_attr``) are always of size ``[num_edges, 1]``.
        The size of the target (``data.y``) depends on the task.

        .. list-table::
           :header-rows: 1

           * - Task
             - Node features
             - Edge features
             - Target
           * - Minimum spanning tree
             - None
             - Edge weights
             - Edges contained in the minimum spanning tree, size ``[num_edges, 1]``
           * - Maximum clique
             - None
             - None
             - Nodes belonging to the maximum clique, size ``[num_nodes, 1]``
           * - Topological sort
             - None
             - None
             - Topological rank, size ``[num_nodes, 1]``
           * - Maximum flow
             - 1 for the source node, 2 for the sink node, and 0 for all other nodes
             - Edge weights
             - Maximum flow computation, size ``[1]``
           * - Bipartite matching
             - None
             - Edge weights
             - Edges contained in the maximum matching, size ``[num_edges, 1]``
           * - Bridge finding
             - None
             - None
             - Edges that are bridges, size ``[num_edges, 1]``
           * - Steiner Tree
             - 0, except for terminal nodes
             - Edge weights
             - Edges contained in the Steiner tree, size ``[num_edges, 1]``


    List of Available Datasets:
        We provide one dataset for each combination of task and difficulty level.
        These can be loaded with ``{task}_{difficulty}``, where task is one of
        ``mst``,
        ``maxclique``,
        ``topologicalorder``,
        ``flow``,
        ``bipartite_matching``,
        ``bridges``,
        or ``steinertree``,
        and difficulty is one of ``easy``, ``medium``, or ``hard``.

        For example:

        .. code:: python

           from graphbench import Loader
           # loads the minimum spanning tree dataset with easy difficulty
           dataset = Loader("data", "mst_easy").load()

        In addition to this, we provide convenience identifiers for loading all datasets of a given difficulty level,
        using ``algorithmic_reasoning_easy``, ``algorithmic_reasoning_medium``, and ``algorithmic_reasoning_hard``.

        The size generalization datasets can be loaded with ``TODO``.
        Note that these only include test graphs.


    Usage Notes:
        The dataset class supports two modes:

        1. Generate synthetic graphs using NetworkX random graph generators
        2. Download and load pre-generated graphs.
           We recommend against using this directly; use the interface provided by :class:`graphbench.Loader` instead

        The class writes a processed file named ``{dataset_name}_{num_nodes}_{difficulty}_{split}.pt``
        into the dataset folder under ``root/algoreas/<raw_folder>/processed/``.
    """

    def __init__(
        self,
        name: str,
        split: Literal["train", "val", "test"],
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        generate: bool = False,
        # TODO these are currently unused. should either be implemented or removed from the public API
        num_nodes: Optional[int] = 16,
        difficulty: Optional[str] = "easy",
        follower_subgraph: bool = False,
        cleanup_raw: bool = True,
        # TODO: This should be removed in the future -- the user will download these files
        load_preprocessed=False,
    ):
        """
        Args:
            name: Dataset identifier in the form ``{task}_{difficulty}_{num_nodes}``,
                  e.g. ``bipartitematching_easy_16``.
                  Note that this differs from the names provided to :class:`graphbench.Loader`.
            split: Whether to load the train, validation, or test split of the dataset.
            root: Root directory where the ``algoreas`` dataset folder lives.
            transform: Optional PyG transform applied to data objects before every access.
            pre_transform: Optional PyG transform applied before saving data objects to disk.
            pre_filter: A function that indicates whether a data object should be included in the final dataset.
            generate: If True, generate synthetic graphs instead of downloading.
            num_nodes: Number of nodes (used when generating datasets).
            difficulty: Difficulty level used for sampling configuration.
            follower_subgraph: (Unused) placeholder for follower-subgraph extraction.
            cleanup_raw: If True, remove raw files after processing.
            load_preprocessed: If True, load existing processed objects instead of regenerating.
        """

        #currently downloads everything at once for a single dataset. Up to the user to manually unpack it so far
        self.SOURCES: Dict[str, SourceSpec] = {
            "topologicalorder": SourceSpec(
                url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_Algoreas/resolve/main/topologicalorder.tar.gz",
                raw_folder="topological_order",
            ),
            "bipartitematching": SourceSpec(
                url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_Algoreas/resolve/main/bipartitematching.tar.gz",
                raw_folder="bipartite_matching",
            ),
            "mst": SourceSpec(
                url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_Algoreas/resolve/main/mst.tar.gz",
                raw_folder="mst",
            ),
            "steinertree": SourceSpec(
                url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_Algoreas/resolve/main/steinertree.tar.gz",
                raw_folder="steiner_tree",
            ),
            "bridges": SourceSpec(
                url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_Algoreas/resolve/main/bridges.tar.gz",
                raw_folder="bridges",
            ),
            "maxclique": SourceSpec(
                url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_Algoreas/resolve/main/maxclique.tar.gz",
                raw_folder="max_clique",
            ),
            "flow": SourceSpec(
                url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_Algoreas/resolve/main/flow.tar.gz",
                raw_folder="flow",
            )
        }
        self.name_temp = name.replace("_"," ").lower()
        self.dataset_name = self.name_temp.split(" ")[1]
        self.num_nodes = self.name_temp.split(" ")[3]
        self.difficulty = self.name_temp.split(" ")[2]
        
        #self.name = name.lower()
        if self.dataset_name not in self.SOURCES:
            raise ValueError(f"Unsupported dataset name: {self.dataset_name}")
        assert split in ["train", "val", "test"], "Only 'train', 'val', 'test' splits are supported."


        self.generate = generate
        self.split = split
        self.source = self.SOURCES[self.dataset_name]
        self.cleanup_raw = cleanup_raw
        self.load_preprocessed = load_preprocessed

        # paths
        self.algoreas_dir = Path(root) / "algoreas"
        self._raw_dir = (self.algoreas_dir  / self.SOURCES[self.dataset_name].raw_folder / "raw")
        self.processed_path = self.algoreas_dir / self.SOURCES[self.dataset_name].raw_folder / "processed" / f"{self.dataset_name}_{self.num_nodes}_{self.difficulty}_{split}.pt"
        super().__init__(str(self.algoreas_dir), transform, pre_transform, pre_filter)

        # process data if needed
        if self.processed_path.exists():
            _logger.info(f"Loading cached processed data: {self.processed_path}")
            self.load(self.processed_path)
            return

        self._prepare()  # (i) downloads, unpacks, load data + (ii) timestep handle + (e) subgraph + collate
        self.load(self.processed_path)
        if self.cleanup_raw:
            self._cleanup()


    def _generate(self) -> None:
        """
        Creates the algorithmic reasoning datasets with the underlying generation methods used in the original creation.
        Returns
        - list[Data]: Generated dataset as a list of PyG Data objects.
        """
        data_list = generate_algoreas_data(
            name=self.dataset_name,
            split=self.split,
            num_nodes=self.num_nodes,
            difficulty=self.difficulty,
        )
        return data_list

    def _prepare(self) -> None:
        # (b) Download & unpack helpers
        if self.generate:
            data_list = self._generate()
        else:
            # Download and unpack into the raw directory, and then load the
            # first matching processed file using `_load_algoreas_graphs`.
            download_and_unpack(
                source=self.source,
                raw_dir=self._raw_dir,
                processed_dir=self.processed_path,
                logger=_logger,
            )

            # The loader places the data into this InMemoryDataset instance
            self._load_algoreas_graphs()

            # After loading into `self`, expose all elements as a list
            data_list = [self.get(i) for i in range(len(self))]

        # Apply pre_filter if provided
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        # Apply pre_transform if provided and save the processed cache
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.save(data_list, self.processed_path)
        _logger.info(f"Saved processed dataset -> {self.processed_path}")


    def _cleanup(self) -> None:
        """
        Remove the dataset-specific raw folder contents. Only removes files
        under `self._raw_dir` and attempts to remove the directory if empty.
        If other processes share files under the same folder the directory may
        remain and this method will silently continue.
        """
        if self._raw_dir.exists():
            _logger.info(f"Cleaning up: {self._raw_dir}")
            # remove only the dataset-specific temp folder
            for p in sorted(self._raw_dir.rglob("*"), reverse=True):
                try:
                    p.unlink()
                except IsADirectoryError:
                    pass
            try:
                self._raw_dir.rmdir()
            except OSError:
                # not empty due to shared artifacts; leave it
                pass

    def _load_algoreas_graphs(self) -> List[Data]:
        """
        Find the matching processed `.pt` file in `self._raw_dir` and load it
        into this InMemoryDataset instance using the existing `load` method.

        The function expects the raw folder to contain a processed `.pt` file
        matching the naming convention produced by the dataset generation
        pipeline. If multiple files are present the first matching file is used.
        """
        filepaths = self._find_matching_files(
            task=self.dataset_name, nodes=self.num_nodes, difficulty=self.difficulty, split=self.split, directory=self._raw_dir
        )
        if not filepaths:
            raise FileNotFoundError(f"No matching processed files found in {self._raw_dir}")
        # load into this InMemoryDataset (populates self._data_list / slices)
        self.load(filepaths[0])

    def _find_matching_files(self,directory, task, nodes, difficulty, split):
        """
        Returns a list of filenames matching the convention in the directory.
        """
        pattern = f"{task}_{difficulty}_{nodes}.pt"
        try:
            return [os.path.join(directory, fname) for fname in os.listdir(directory) if fname == pattern]
        except FileNotFoundError:
            return []


    @property
    def raw_file_names(self) -> List[str]:  # unused, we drive our own cache
        return []

    @property
    def processed_file_names(self) -> list[str]:
        # Provide the expected processed filename for PyG compatibility. This
        # is primarily for API compatibility; loading/saving is handled by the
        # class via `self.processed_path`.
        return [f"{self.dataset_name}_{self.difficulty}_{self.num_nodes}_{self.split}.pt"]
    



if __name__ == "__main__":
    dataset = AlgoReasDataset(root="datatest", name="test_16_easy", split="train", generate=False)
    print(dataset)