"""
electronic circuits dataset loader
----------------------------------

This module provides `ECDataset`, a PyTorch Geometric `InMemoryDataset`
implementation for Electronic Circuits benchmark datasets. It downloads and
unpacks archived JSON representations of circuit instances and converts them
into PyG `Data` objects. The class supports loading preprocessed caches and
offers utilities for label normalization and dataset splitting.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from tokenize import Name
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import tqdm
from torch_geometric.data import Data, InMemoryDataset

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

class ECDataset(InMemoryDataset):
    def __init__(
        self,
        name: str,
        split: str,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        generate: Optional[bool] = False,
        cleanup_raw: bool = True,
        target_vout : Optional[float] = None,
        vout_norm_method : Optional[str] = 'min-max',
        # TODO: This should be removed in the future -- the user will download these files
        load_preprocessed = False,
    ):
        """
        Initialize the Electronic Circuits dataset wrapper.

        Parameters
        - name (str): Dataset identifier, e.g. 'electronic_circuits_5_eff'.
        - split (str): 'train', 'val' or 'test'.
        - root (str|Path): Root dataset directory.
        - generate (bool): If True, attempt to generate dataset (not supported).
        - target_vout (float|None): Optional target value for vout normalization.
        - vout_norm_method (str): Normalization method for vout labels.

       
        """
        #currently downloads everything at once for a single dataset. Up to the user to manually unpack it so far
        self.SOURCES: Dict[str, _SourceSpec] = {
        "electronic_circuits_5_eff": _SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_ElectronicCircuits/resolve/main/ec_5.zip",
            raw_folder="electronic_circuits_5_eff",
        ), 
        "electronic_circuits_5_vout": _SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_ElectronicCircuits/resolve/main/ec_5.zip",
            raw_folder="electronic_circuits_5_vout",
        ),
        "electronic_circuits_7_eff": _SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_ElectronicCircuits/resolve/main/ec_7.zip",
            raw_folder="electronic_circuit_7_eff",
        ), 
        "electronic_circuits_7_vout": _SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_ElectronicCircuits/resolve/main/ec_7.zip",
            raw_folder="electronic_circuits_7_vout",
        ),
        "electronic_circuits_10_eff": _SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_ElectronicCircuits/resolve/main/ec_10.zip",
            raw_folder="electronic_circuits_10_eff",
        ), 
        "electronic_circuits_10_vout": _SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_ElectronicCircuits/resolve/main/ec_10.zip",
            raw_folder="electronic_circuits_10_vout",
        ),

    }

        self.root = root
        self.dataset_name = name.lower()
        if self.dataset_name not in self.SOURCES:
            raise ValueError(f"Unsupported dataset name: {self.dataset_name}")
        assert split in ["train", "val", "test"], "Only 'train', 'val', 'test' splits are supported."
        self.target = self.dataset_name.split("_")[-1]
        self.component_size = int(self.dataset_name.split("_")[-2])
        self._target = self.target
        self._target_vout = target_vout
        self._vout_norm_method = vout_norm_method
        self.generate = generate
        self.split = split
        self.source = self.SOURCES[self.dataset_name]
        self.cleanup_raw = cleanup_raw
        self.load_preprocessed = load_preprocessed

        # paths
        self.ec_dir = Path(root) / "electroniccircuits"
        
        
        self._raw_dir = (self.ec_dir / self.SOURCES[self.dataset_name].raw_folder / "raw")
        # Include time window & task in the processed filename to avoid collisions
        self.processed_path = (self.ec_dir /self.SOURCES[self.dataset_name].raw_folder / "processed" / f"{self.dataset_name}_{self.split}.pt")
        super().__init__(str(self.ec_dir), transform, pre_transform)

        # process data if needed
        if self.processed_path.exists():
            self.load(self.processed_path)
            
        else:
            self._prepare()  # (i) downloads, unpacks, load data + (ii) timestep handle + (e) subgraph + collate
            self.load(self.processed_path)
        if self.cleanup_raw:
            self._cleanup()
        

    def _generate(self, pre_transform, transform) -> None:
        pass 

    def _prepare(self) -> None:
        # (b) Download & unpack helpers
        if self.generate:
            raise NotImplementedError("Dataset generation not supported yet.")
        else:
            _download_and_unpack(source=self.source, raw_dir=self._raw_dir, processed_dir=self.processed_path, logger=logger)

            train_json = self.load_json(os.path.join(self._raw_dir, f"dataset_{self.component_size}_train.json"))
            valid_json = self.load_json(os.path.join(self._raw_dir, f"dataset_{self.component_size}_valid.json"))
            test_json = self.load_json(os.path.join(self._raw_dir, f"dataset_{self.component_size}_test.json"))

            data_all = train_json + valid_json + test_json


            targets = [datum['eff'] if self._target == 'eff' else datum['vout'] for datum in data_all]
            statistics = self.get_statistics(targets)
            y_range = self.get_y_range(
                target=self._target,
                statistics=statistics,
                method=self._vout_norm_method,
                target_min=-300,
                target_max=300,
            )

            # Select which split to process
            split_to_data = {"train": train_json, "val": valid_json, "test": test_json}
            split_data = split_to_data[self.split]

            # Build PyG Data objects
            data_list = self._make_datalist_from_json(
                data=split_data,
                target=self._target,
                vout_norm_method=self._vout_norm_method,
                statistics=statistics,
                y_range=y_range,
                target_vout=self._target_vout,
            )

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in tqdm(data_list, desc="pre_transform")]

            self.save(data_list, self.processed_path)
            logger.info(f"Saved processed dataset -> {self.processed_path}")


    def _cleanup(self) -> None:
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
    
    def _make_datalist_from_json(self,
        data: List[Dict[str, Any]],
        target: str,
        vout_norm_method: str,
        statistics: Dict[str, float],
        y_range: Dict[str, float],
        target_vout: Optional[float] = None,
    ) -> List[Data]:
        data_list = []
        for datum in data:
            node_features = torch.tensor(datum['node_features'], dtype=torch.float).unsqueeze(1)
            edge_index = torch.tensor(datum['edge_index'], dtype=torch.long)
            edge_features = None

            duty = torch.tensor(datum['duty'])
            y = self.get_label(
                target=target,
                datum=datum,
                method=vout_norm_method,
                target_vout=target_vout,
                statistics=statistics,
                y_range=y_range,
            )

            data_list.append(Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_features,
                y=y,
                duty=duty,
                device_ids=torch.tensor(datum['device_ids']),
                device_ids_len=torch.tensor(datum['device_ids_len']),
                port_ids=torch.tensor(datum['port_ids']),
                port_ids_len=torch.tensor(datum['port_ids_len']),
                terminal_ids=torch.tensor(datum['terminal_ids']),
                terminal_ids_len=torch.tensor(datum['terminal_ids_len']),
            ))
        return data_list
    
    def get_label(self,target, datum, method='min-max', target_vout=None, statistics=None, y_range=None):
        if target == 'eff':
            y_val = datum['eff']
            y = torch.clamp(torch.tensor(y_val), y_range['min'], y_range['max'])
        elif target == 'vout':
            if method == 'min-max':
                vout = (datum['vout'] + 300.) / 600.
            elif method == 'reward':
                vout = self.reward_norm_vout(vout=datum['vout'], target_vout=target_vout)
            elif method == 'IQR':
                vout = (datum['vout'] - statistics['q25']) / statistics['iqr']
            elif method == 'z-score':
                vout = (datum['vout'] - statistics['mean']) / statistics['std']
            else:
                raise ValueError('Unknown norm method')
            y = torch.clamp(torch.tensor(vout), y_range['min'], y_range['max'])
        else:
            raise Exception(f"Unimplemented target {target}")
        return y
    
    def reward_norm_vout(self, vout: float, target_vout: float) -> float:
    # Placeholder normalization — replace if needed.
        return 1.0 / (1.0 + abs(vout - target_vout))
    
    def get_y_range(self, target, statistics, method='min-max', target_min=-300, target_max=300):
        if target == 'eff':
            return {'min': 0., 'max': 1.}
        elif target == 'vout':
            if method in ['min-max', 'reward']:
                return {'min': 0., 'max': 1.}
            elif method == 'IQR':
                return {'min': (target_min - statistics['q25']) / statistics['iqr'],
                        'max': (target_max - statistics['q25']) / statistics['iqr']}
            elif method == 'z-score':
                return {'min': (target_min - statistics['mean']) / statistics['std'],
                        'max': (target_max - statistics['mean']) / statistics['std']}
            else:
                raise ValueError('Unknown norm method')
        else:
            raise Exception(f"Unimplemented target {target}")
        
    def get_statistics(self, data: List[float]) -> Dict[str, float]:
        data = np.array(data)
        return {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'q25': float(np.percentile(data, 25)),
            'q75': float(np.percentile(data, 75)),
            'iqr': float(np.percentile(data, 75)) - float(np.percentile(data, 25)),
    }

    def _find_matching_files(self,directory, task, split: Optional[str] = None, size: Optional[str] = None, target: Optional[str] = None):
        """
        Returns a list of filenames matching the convention in the directory.
        """
        if split is None:
            pattern = f"{task}_{size}_{target}.pt"
        elif split is None and target is None:
            pattern = f"{task}_{size}.pt"
        elif split is None and target is None and size is None:
            pattern = f"{task}.pt"
        else:
            pattern = f"{task}_{size}_{target}_{split}.pt"
        return [os.path.join(directory, fname)
                for fname in os.listdir(directory)
                if fname == pattern]

    def load_json(self, name: str) -> list:
        """Load a JSON file and ensure it's returned as a list of dictionaries."""
        path = name
        with open(path, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return list(data.values())
        else:
            raise ValueError(f"Unsupported JSON structure in {path}: {type(data)}")
        

    @property
    def raw_file_names(self) -> list[str]:
        return []

    @property
    def processed_file_names(self) -> list[str]:
        return [f"{self.dataset_name}_{self.split}.pt"]
    
        
