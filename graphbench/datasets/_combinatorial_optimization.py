"""
co datasets loader
------------------

This module provides `CODataset`, a PyTorch Geometric `InMemoryDataset`
wrapper for the combinatorial optimization (CO) benchmark datasets used in the
graphbench. The class supports two modes:

- Generation: build synthetic graph collections via `co_helpers.datasets` and
    split them into train/val/test (currently disabled).
- Download & load: fetch preprocessed `.pt` files, convert into PyG Data
    objects and cache a processed file for fast loading.

The `name` argument selects among supported dataset variants (e.g. 'ba_small',
'er_large'), and `split` must be one of 'train', 'val', or 'test'.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Literal, Optional, TYPE_CHECKING, Union

from graphbench._co_helpers import BADataset, ERDataset, RBDataset
from graphbench._helpers import download_and_unpack, split_dataset, SourceSpec, get_logger
from ._base import GraphDataset


if TYPE_CHECKING:
    from torch_geometric.data import Data


# (i) helper functions

# -----------------------------------------------------------------------------#
# (a) Utilities
# -----------------------------------------------------------------------------#

_logger = get_logger(__name__)
 

class CODataset(GraphDataset):
    r"""
    Combinatorial Optimization (CO) datasets.

    Note:
        This class **should only be used directly when generating new datasets**.
        To access provided datasets, please consider using :class:`graphbench.Loader`.
        The sections below give details on the data available through the :class:`graphbench.Loader` interface.

    Overview:
        In the CO datasets, we consider 3 classic NP-hard graph problems: the maximum independent set (MIS), the max-cut, and the graph coloring problem. 
        We include tasks for both supervised and unsupervised learning settings:

        - **Supervised setup**: Given a collection of CO
          instances :math:`\mathcal{I}` and, for each instance
          :math:`I \in \mathcal{I}`, an optimal solution
          :math:`S^{*}_{I}` with objective value :math:`c(S^{*}_{I})` and a model :math:`m:\mathcal{I}\to\mathbb{R}`.
          The learning task is to minimize the MAE between predicted and true optimal objectives:

          .. math::
            
              \frac{1}{|\mathcal{I}|} \sum_{I \in \mathcal{I}} \left| m(I) - c(S^{*}_{I}) \right|

          The supervised setup requires solver-generated
          solutions and can be computationally expensive for large instances.


        - **Unsupervised setup**: This setting is motivated when ground-truth
          solutions are unavailable or expensive to obtain, and it directly
          targets the CO problem. We provide a
          differentiable surrogate loss :math:`\mathcal{L}:\mathbb{R}^{|\Omega(I)|}\to\mathbb{R}`
          together with decoders :math:`d:\mathbb{R}^{|\Omega(I)|}\to F(I)`
          for each CO problem. The learning task is to train the model :math:`m : I \to \mathbb{R}^{|\Omega(I)|}`
          in an unsupervised fashion to predict a score for each
          variable that indicates whether it belongs to the solution set,
          minimizing

          .. math::

              \frac{1}{|\mathcal{I}|} \sum_{I \in \mathcal{I}} \mathcal{L}(m(I))

        Currently, only MIS includes heuristic solutions generated with the
        `KaMIS <https://github.com/KarlsruheMIS/KaMIS>`_ (Karlsruhe Maximum Independent Sets) solver.

        We synthetically generate optimization problems across 3 well-established random graph families:

        - Barabási-Albert (BA)
        - Erdős-Rényi (ER)
        - RB

        Each graph family is available in 2 configurations: small and large, totaling 6 distinct datasets.
        There are 50,000 graphs in each dataset. The small graphs contain 200-300 nodes, while the large graphs contain 700-1200 nodes (700-800 for ER and BA, 800-1200 for RB).
        Note that the BA graphs are considerably less dense than the ER and RB graphs.

        Please refer to the `GraphBench paper <https://arxiv.org/abs/2512.04475>`__ for the exact parameters used for graph generation.
    
    Splits:
        All datasets use a 70% / 15% / 15% split for training, validation,
        and testing.

    Graph Attributes:
        The generated graphs include no node or edge features, as the task focuses on combinatorial optimization based solely on graph structure.

        .. list-table::
           :header-rows: 1

           * - Attribute name
             - Size
             - Description
           * - ``num_mis``
             - ``[1]``
             - Optimal objective value of the given CO instance
           * - ``mis_solution``
             - ``[num_nodes]``
             - Binary node indicators representing the optimal Maximum Independent Set solution

    List of Available Datasets:
        The available datasets are named using the pattern ``co_{generator}_{size}``,
        where generator is one of ``ba``, ``er``, or ``rb``, and size is either ``small`` or ``large``.

        Hence, valid dataset names for the loader are:
        ``co_ba_small``,
        ``co_er_small``,
        ``co_rb_small``,
        ``co_ba_large``,
        ``co_er_large``,
        and ``co_rb_large``.

        For example:

        .. code:: python

            from graphbench import Loader
            # loads the Barabási-Albert small dataset
            dataset = Loader("data", "co_ba_small").load()

        In addition to this, we provide ``co`` as a convenience identifier to load all of the above datasets.

    Usage Notes:
        The dataset class supports two modes:

        1. Generate synthetic graphs using NetworkX random graph generators
        2. Download and load pre-generated graphs.
           We recommend against using this directly; use the interface provided by :class:`graphbench.Loader` instead
    """
    def __init__(
        self,
        name: str,
        split: Literal["train", "val", "test"],
        root: Union[str, Path],
        transform: Optional[Callable[[Data], Data]] = None,
        pre_transform: Optional[Callable[[Data], Data]] = None,
        pre_filter: Optional[Callable[[Data], bool]] = None,
        generate: Optional[bool] = False,
        num_samples: Optional[int] = None,
        cleanup_raw: bool = False,
        # TODO: This should be removed in the future -- the user will download these files
        load_preprocessed = False,
    ):
        """
        Args:
            name: Dataset identifier in the form ``co_{graph_type}_{size}``, e.g. ``co_rb_large``.
            split: Whether to load the train, validation, or test split of the dataset.
            root: Root directory where the dataset folder will be created.
            transform: Optional PyG transform applied to data objects before every access.
            pre_transform: Optional PyG transform applied before saving data objects to disk.
            pre_filter: A function that indicates whether a data object should be included in the final dataset.
            generate: If True, generate synthetic graphs instead of downloading.
            num_samples: Number of synthetic graphs to generate when generate=True.
            cleanup_raw: If True, remove raw files after processing.
            load_preprocessed: If True, load existing processed objects instead of regenerating.
        """
        #currently downloads everything at once for a single dataset. Up to the user to manually unpack it so far
        self.SOURCES: Dict[str, SourceSpec] = {
            "ba_small": SourceSpec(
                url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_CO/resolve/main/ba_small_mis_labeled.tar.gz",
                raw_folder="co_ba_small",
            ),
            "er_small": SourceSpec(
                url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_CO/resolve/main/er_small_mis_labeled.tar.gz",
                raw_folder="co_er_small",
            ),
            "rb_small": SourceSpec(
                url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_CO/resolve/main/rb_small_mis_labeled.tar.gz",
                raw_folder="co_rb_small",
            ),
            "rb_large": SourceSpec(
                url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_CO/resolve/main/rb_large_mis_labeled.tar.gz",
                raw_folder="co_rb_large",
            ),
            "er_large": SourceSpec(
                url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_CO/resolve/main/er_large_mis_labeled.tar.gz",
                raw_folder="co_er_large",
            ),
            "ba_large": SourceSpec(
                url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_CO/resolve/main/ba_large_mis_labeled.tar.gz",
                raw_folder="co_ba_large",
            ),
        }

        self.LABEL_SOURCES: Dict[str, SourceSpec] = {
            "labels": SourceSpec(
                url="redacted",
                raw_folder="supervised_labels",
            ),
        }

        #self.name_temp = name.replace("_"," ")
        #self.dataset_name = self.name_temp.lower().split(" ")[0]
        #self.size = self.name_temp.lower().split(" ")[1]
        #self.dataset_name = name.lower()
        self.num_samples = num_samples
        self.dataset_name = name.lower().split("_")[1] + "_" + name.lower().split("_")[2]
        if self.dataset_name not in self.SOURCES:
            raise ValueError(f"Unsupported dataset name: {self.dataset_name}")
        assert split in ["train", "val", "test"], "Only 'train', 'val', 'test' splits are supported."

        self.generate = generate
        self.split = split
        self.source = self.SOURCES[self.dataset_name]
        self._logger = _logger
        self.cleanup_raw = cleanup_raw
        self.load_preprocessed = load_preprocessed

        # paths
        self.co_dir = Path(root) / "co"
        self._raw_dir = self.co_dir / self.SOURCES[self.dataset_name].raw_folder / "raw"
        self.processed_path = self.co_dir / self.SOURCES[self.dataset_name].raw_folder / "processed" / "data.pt"
        super().__init__(str(self.co_dir), transform, pre_transform, pre_filter)

        self._load_cached_or_prepare(
            processed_path=self.processed_path,
            cleanup_raw=self.cleanup_raw,
            logger=_logger,
        )
        

    def _generate(self) -> None:
        # TODO RBDataset etc. may be using processed_dir where they should be using raw_dir. Refactor and get rid of SyntheticDataset.
        if self.num_samples is None:
            raise ValueError("num_samples cannot be None when generating a new dataset")
        dataset_folder = self.co_dir / self.SOURCES[self.dataset_name].raw_folder
        if "rb" in self.dataset_name:
            data = RBDataset(root=dataset_folder, num_samples=self.num_samples)

        elif "er" in self.dataset_name:
            data = ERDataset(root=dataset_folder, num_samples=self.num_samples)

        elif "ba" in self.dataset_name:
            data = BADataset(root=dataset_folder, num_samples=self.num_samples)
        else:
            raise ValueError(f"Dataset generation not supported for {self.dataset_name}")

        train_data, valid_data, test_data = split_dataset(data, 0.7, 0.15, 0.15)
        if self.split == "train":
            return train_data
        elif self.split == "val":
            return valid_data
        elif self.split == "test":  
            return test_data
        else:
            raise ValueError(f"Unknown split: {self.split}")

    def _prepare(self) -> None:
        # (b) Download & unpack helpers
        if self.generate:
            return

        download_and_unpack(
            source=self.source,
            raw_dir=self._raw_dir,
            processed_dir=self.processed_path,
            logger=_logger,
        )

    def _load_graphs(self):
        if self.generate:
            return self._generate()

        filepaths = self._find_matching_files(task=self.dataset_name, directory=self._raw_dir)
        self.load(filepaths[0])

        return [self.get(i) for i in range(len(self))]

    def _find_matching_files(self, directory, task, split: Optional[str] = None, size: Optional[str] = None):
        """
        Returns a list of filenames matching the convention in the directory.
        """
        return [str(self.processed_path)]
    
    def process(self):
        self._prepare()
        

    @property
    def raw_file_names(self) -> list[str]:
        return []

    @property
    def processed_file_names(self) -> list[str]:
        return ['data.pt']
