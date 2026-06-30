from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Union

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from graphbench._helpers import download_and_unpack, SourceSpec, get_logger
from ._base import GraphDataset


# (i) helper functions

# -----------------------------------------------------------------------------#
# (a) Utilities
# -----------------------------------------------------------------------------#

_logger = get_logger(__name__)


class ChipDesignDataset(GraphDataset):
    """
    chipdesign dataset loader
    -------------------------

    A PyTorch Geometric `InMemoryDataset`
    wrapper for the Chip Design dataset used by the project. The dataset class is
    responsible for downloading/unpacking the original archive (via
    `helpers.download._download_and_unpack`), locating the preprocessed `.pth`
    files, converting each example into a PyG `Data` object and caching a processed
    `.pt` file for fast subsequent loading.

    Usage notes:
    - The dataset expects a directory structure where the archive unpacks into a
        `raw/Data/` directory containing `train.pth`, `val.pth` and `test.pth` files.
    - Instantiating the dataset will load a processed cache if available; otherwise
        it will download/unpack and convert the raw files then write the processed
        cache to `root/chipdesign/chipdesign/processed/<name>_<split>.pt`.
    """

    def __init__(
        self,
        name: str,
        split: Literal["train", "val", "test"],
        root: Union[str, Path],
        transform: Optional[Callable[[Data], Data]] = None,
        pre_transform: Optional[Callable[[Data], Data]] = None,
        pre_filter: Optional[Callable[[Data], bool]] = None,
        cleanup_raw: bool = False,  # TODO Disabling this for now since it leads to errors on my machine
    ):
        """
        Args:
            name: Dataset identifier. If loading the existing dataset, this must be ``chipdesign``.
            split: Whether to load the train, validation, or test split of the dataset.
            root: Root directory where the dataset folder will be created.
            transform: Optional PyG transform applied to data objects before every access.
            pre_transform: Optional PyG transform applied before saving data objects to disk.
            pre_filter: A function that indicates whether a data object should be included in the final dataset.
            cleanup_raw: If True, remove raw files after processing.
        """

        # currently downloads everything at once for a single dataset. Up to the user to manually unpack it so far
        self.SOURCES: Dict[str, SourceSpec] = {
            "chipdesign": SourceSpec(
                url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_chipdesign/resolve/main/chipdesign.zip",
                raw_folder="chipdesign",
            ),
        }


        self.name = name.lower()
        if self.name not in self.SOURCES:
            raise ValueError(f"Unsupported dataset name: {self.name}")
        assert split in ["train", "val", "test"], "Only 'train', 'val', 'test' splits are supported."

        self.split = split
        self.source = self.SOURCES[self.name]
        self._logger = _logger
        self.cleanup_raw = cleanup_raw

        # paths
        self.chipdesign_dir = Path(root) / "chipdesign"
        self._raw_dir = (self.chipdesign_dir  / self.SOURCES[self.name].raw_folder / "raw")
        # Include time window & task in the processed filename to avoid collisions
        self.processed_path = self.chipdesign_dir / self.SOURCES[self.name].raw_folder / 'processed' / f"{self.name}_{split}.pt"
        super().__init__(str(self.chipdesign_dir), transform, pre_transform, pre_filter)

        self._load_cached_or_prepare(
            processed_path=self.processed_path,
            cleanup_raw=self.cleanup_raw,
            logger=_logger,
        )

    def _prepare(self) -> None:
        """
        Download (if needed) and unpack raw ChipDesign files.
        """

        download_and_unpack(
            source=self.source,
            raw_dir=self._raw_dir,
            processed_dir=self.processed_path,
            logger=_logger,
        )

    def _load_graphs(self) -> List[Data]:
        """
        Convert raw `.pth` files into a list of PyG `Data` objects.
        """
        data_list = self._load_chipdesign_graphs()
        return data_list

    def _load_chipdesign_graphs(self) -> List[Data]:
        """
        Convert raw `.pth` files into a list of PyG `Data` objects.

        The ChipDesign raw files are expected to be Python mappings where each
        key corresponds to a configuration (e.g. 'in3_out2') and the value is
        a dict containing arrays for 'x', 'edge_index', 'edge_attr' and
        'truth_vectors'. This method iterates over all configurations and
        samples, building a PyG `Data` object per sample.
        """
        data_list: List[Data] = []
        filepaths = self._find_matching_files(task=self.name, split=self.split, directory=self._raw_dir)
        for path in filepaths:
            raw = torch.load(path, weights_only=False)
            sample_idx = 0

            for config_key, config_data in raw.items():
                if config_key.startswith('_'):
                    continue

                # Extract configuration info from key like 'in3_out2'
                parts = config_key.split('_')
                num_inputs = int(parts[0].replace('in', ''))
                num_outputs = int(parts[1].replace('out', ''))
                num_samples = len(config_data['x'])

                for i in tqdm(range(num_samples), desc=f"Processing {config_key}"):
                    pyg_data = self._load_sample(config_data, i, num_inputs, num_outputs)

                    # Extract truth vectors for equivalence checking
                    truth_vectors = config_data['truth_vectors'][i]
                    stored_truth = self._extract_truth_vectors(truth_vectors, num_inputs, num_outputs)

                    if stored_truth is not None:
                        pyg_data.sample_idx = sample_idx
                        pyg_data.config_key = config_key
                        pyg_data.truth_vectors = stored_truth
                        pyg_data.num_nodes = pyg_data.x.shape[0]

                        data_list.append(pyg_data)
                        sample_idx += 1

        return data_list

    def _find_matching_files(self, directory, task, split):
        """
        Returns a list containing the path to the split file (train.pth, val.pth, or test.pth)
        in the Data subdirectory of the given directory.
        """
        data_dir = os.path.join(directory, "Data")
        pattern = f"{split}.pth"
        file_path = os.path.join(data_dir, pattern)
        if os.path.exists(file_path):
            return [file_path]
        else:
            return []

    def _load_sample(self, config_data, sample_idx, num_inputs, num_outputs):
        """
        Convert a single sample (from the raw config dict) into a PyG Data
        object.

        Parameters
        - config_data (dict): Mapping containing arrays for 'x', 'edge_index', 'edge_attr', etc.
        - sample_idx (int): Index of the sample within the arrays.
        - num_inputs (int): Number of input pins for the circuit.
        - num_outputs (int): Number of output pins for the circuit.

        Returns
        - Data: A torch_geometric Data object for the sample.
        """
        x = np.array(config_data['x'][sample_idx], dtype=np.float32)
        edge_index = np.array(config_data['edge_index'][sample_idx], dtype=np.int64)
        edge_attr = np.array(config_data['edge_attr'][sample_idx], dtype=np.float32)

        # Create PyG data
        data = Data(
            x=torch.from_numpy(x),
            edge_index=torch.from_numpy(edge_index),
            edge_attr=torch.from_numpy(edge_attr).unsqueeze(-1),
        )

        data.num_inputs = num_inputs
        data.num_outputs = num_outputs

        return data

    def _extract_truth_vectors(self,truth_vectors, num_inputs, num_outputs):
        """
        Fast truth vector extraction with numpy operations.

        The raw truth vectors may be padded with `-1` values. This function
        determines the effective length, validates it against the expected
        length (2**num_inputs) and returns a compact numpy array of shape
        `(num_outputs, expected_length)` or `None` if validation fails.
        """
        expected_length = 2 ** num_inputs
        result = np.zeros((num_outputs, expected_length), dtype=np.uint8)

        for output_idx, truth_vector in enumerate(truth_vectors):
            # Find length (-1 padding)
            length = 0
            for val in truth_vector:
                if val == -1:
                    break
                length += 1

            if length != expected_length:
                return None  # Invalid truth vector

            result[output_idx] = truth_vector[:length]

        return result
   
    @property
    def raw_file_names(self) -> List[str]: 
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return [f"{self.name}_{self.split}.pt"]
