"""
algorithmic reasoning  dataset loader
-----------------------

This module provides an `AlgoReasDataset` class implementing a PyTorch Geometric
`InMemoryDataset` for the Algorithmic Reasoning (AlgoReas) datasets used in the
project. The dataset class supports two modes:

- generate: programmatically build synthetic graphs using various NetworkX
    random graph generators (controlled by `algoreas_helpers.algoreas_utils`).
- download & load: fetch preprocessed `.pt` files from the dataset `raw`
    directory (via `helpers.download._download_and_unpack`) and load them into
    PyG `Data` objects.

Usage notes:
- The dataset `name` is expected in the form: "{task}_{num_nodes}_{difficulty}".
    Example: `bipartitematching_16_easy`.
- The class writes a processed file named
    `{dataset_name}_{num_nodes}_{difficulty}_{split}.pt` into the dataset folder
    under `root/algoreas/<raw_folder>/processed/`.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from torch_geometric.data import Data, InMemoryDataset

from graphbench.algoreas_helpers.algoreas_utils import generate_algoreas_data
from graphbench.helpers.download import _download_and_unpack


# (i) helper functions

# -----------------------------------------------------------------------------#
# (a) Utilities
# -----------------------------------------------------------------------------#

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class _SourceSpec:
    url: str
    raw_folder: str  # folder name inside tmp/ where data will appear

class AlgoReasDataset(InMemoryDataset):
    def __init__(
        self,
        name: str,
        split: str,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        generate: Optional[bool] = False,
        num_nodes: Optional[int] = 16,
        difficulty: Optional[str] = "easy",
        follower_subgraph: bool = False,
        cleanup_raw: bool = True,
        # TODO: This should be removed in the future -- the user will download these files
        load_preprocessed=False,
    ):
        """
        Initialize the AlgoReasDataset.

        Parameters
        - name (str): Dataset identifier in the form "{task}_{num_nodes}_{difficulty}".
        - split (str): One of 'train', 'val', 'test'.
        - root (str|Path): Root directory where `algoreas` dataset folder lives.
        - transform, pre_transform (callable | None): Optional PyG transforms.
        - generate (bool): If True, generate synthetic graphs instead of downloading.
        - num_nodes (int): Number of nodes (used when generating datasets programmatically).
        - difficulty (str): Difficulty level used for sampling configuration.
        - follower_subgraph (bool): (Unused) placeholder for follower-subgraph extraction.
        - cleanup_raw (bool): If True, remove raw files after processing.
        - load_preprocessed (bool): If True, load existing processed objects instead of regenerating.

        """
        



        #currently downloads everything at once for a single dataset. Up to the user to manually unpack it so far
        self.SOURCES: Dict[str, _SourceSpec] = {
        "topologicalorder": _SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_Algoreas/resolve/main/topologicalorder.tar.gz",
            raw_folder="topological_order",
        ),
        "bipartitematching": _SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_Algoreas/resolve/main/bipartitematching.tar.gz",
            raw_folder="bipartite_matching",
        ),
        "mst": _SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_Algoreas/resolve/main/mst.tar.gz",
            raw_folder="mst",
        ),
        "steinertree": _SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_Algoreas/resolve/main/steinertree.tar.gz",
            raw_folder="steiner_tree",
        ),
        "bridges": _SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_Algoreas/resolve/main/bridges.tar.gz",
            raw_folder="bridges",
        ),
        "maxclique": _SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_Algoreas/resolve/main/maxclique.tar.gz",
            raw_folder="max_clique",
        ),
        "flow": _SourceSpec(
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
        super().__init__(str(self.algoreas_dir), transform, pre_transform)

        # process data if needed
        if self.processed_path.exists():
            logger.info(f"Loading cached processed data: {self.processed_path}")
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
            _download_and_unpack(
                source=self.source, raw_dir=self._raw_dir, logger=logger, processed_dir=self.processed_path
            )

            # The loader places the data into this InMemoryDataset instance
            self._load_algoreas_graphs()

            # After loading into `self`, expose all elements as a list
            data_list = [self.get(i) for i in range(len(self))]

        # Apply pre_transform if provided and save the processed cache
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.save(data_list, self.processed_path)
        logger.info(f"Saved processed dataset -> {self.processed_path}")


    def _cleanup(self) -> None:
        """
        Remove the dataset-specific raw folder contents. Only removes files
        under `self._raw_dir` and attempts to remove the directory if empty.
        If other processes share files under the same folder the directory may
        remain and this method will silently continue.
        """
        if self._raw_dir.exists():
            logger.info(f"Cleaning up: {self._raw_dir}")
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
        """
        Provide the expected processed filename for PyG compatibility. This
        is primarily for API compatibility; loading/saving is handled by the
        class via `self.processed_path`.
        """
        return [f"{self.dataset_name}_{self.difficulty}_{self.num_nodes}_{self.split}.pt"]
    



if __name__ == "__main__":
    dataset = AlgoReasDataset(root="datatest", name="test_16_easy", split="train", generate=False)
    print(dataset)