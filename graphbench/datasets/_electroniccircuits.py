from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import numpy as np
import torch
from torch_geometric.data import Data

from graphbench._helpers import download_and_unpack, SourceSpec, get_logger
from ._base import GraphDataset


# (i) helper functions

# -----------------------------------------------------------------------------#
# (a) Utilities
# -----------------------------------------------------------------------------#

_logger = get_logger(__name__)


class ECDataset(GraphDataset):
    r"""
    Electronic Circuits (EC) datasets.


    Note:
        This class **should only be used directly when generating new datasets**.
        To access provided datasets, please consider using :class:`graphbench.Loader`.
        The sections below give details on the data available through the :class:`graphbench.Loader` interface.


    Overview:
        We model analog circuit design as graphs :math:`G = (V, E)`, where the node set :math:`V` encodes device components,
        and the edges represent electrical interconnections between ports. Each graph corresponds to a circuit,
        with the goal of predicting two continuous performance metrics: the *voltage conversion ratio* and
        the *power conversion efficiency*.


        The graphs in this dataset are of three complexity levels, generated from random valid topologies with
        5, 7, and 10 components. Each instance is simulated with
        `NGSPICE <https://ngspice.sourceforge.io/>`_
        to obtain ground truth labels for both performance metrics.


        The dataset comprises more than 350,000 graphs with 13–24 nodes and 30–56 edges across the three complexity levels. Concretely,
        there are 334,419 graphs with 5 components, 13,711 graphs with 7 components, and 4,630 graphs with 10 components.


        For details on the circuit generation methodology and exact graph configurations, please refer to the
        `GraphBench paper <https://arxiv.org/abs/2512.04475>`__.




    Splits:
        All datasets use a 70% / 10% / 20% random split for training, validation,
        and testing.


    Graph Attributes:
        Each graph has the following attributes:


        .. list-table::
            :header-rows: 1


            * - Attribute name
              - Size
              - Description
            * - ``x``
              - ``[num_nodes, 1, 9]``
              - One-hot encoded vectors representing device component properties
            * - ``duty``
              - ``[1]``
              - Duty cycle value for the circuit
            * - ``device_ids``
              - ``[num_devices]``
              - Device type identifiers for each device in the circuit
            * - ``port_ids``
              - ``[num_ports]``
              - Port identifiers for circuit connections
            * - ``terminal_ids``
              - ``[num_terminals]``
              - Terminal identifiers for circuit interconnections




    Targets:
        The dataset provides two distinct targets corresponding to circuit performance metrics. The desired target can be selected by loading the dataset with the correct suffix, as documented below.


        .. list-table::
            :header-rows: 1


            * - Task
              - Output size
              - Description
            * - Power Conversion Efficiency (datasets with suffix ``_eff``)
              - ``[1]``
              - Continuous target representing the power conversion efficiency, raction of input power delivered to the load.
            * - Voltage Conversion Ratio (datasets with suffix ``_vout``)
              - ``[1]``
              - Continuous target representing the output voltage (Vout), which is the output-to-input voltage.




    List of Available Datasets:
        We provide one dataset for each combination of component complexity and performance metric target.
        These can be loaded with ``electronic_circuits_{component_size}_{target}``, where ``component_size`` is one of
        ``5``,
        ``7``,
        or ``10``,
        and ``target`` is one of ``eff`` or ``vout``.


        This totals 6 valid dataset names:


        - ``electronic_circuits_5_eff``
        - ``electronic_circuits_5_vout``
        - ``electronic_circuits_7_eff``
        - ``electronic_circuits_7_vout``
        - ``electronic_circuits_10_eff``
        - ``electronic_circuits_10_vout``


        For example:


        .. code:: python


            from graphbench import Loader
            # Loads circuits with 7 components, predicting voltage conversion ratio
            dataset = Loader("data", "electronic_circuits_7_vout").load()


        In addition to this, we provide ``electronic_circuits`` as a convenience identifier to load all of the above datasets.


    Usage Notes:
        The class supports various normalization methods for the voltage conversion ratio target,
        including ``min-max``, ``z-score``, ``IQR``, and ``reward``-based normalization.
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
        cleanup_raw: bool = False,
        target_vout : Optional[float] = None,
        vout_norm_method : Optional[str] = 'min-max',
        # TODO: This should be removed in the future -- the user will download these files
        load_preprocessed = False,
    ):
        """
        Args:
            name: Dataset identifier in the form ``electronic_circuits_{component_size}_{target}``, e.g. ``electronic_circuits_7_eff``.
            split: Whether to load the train, validation, or test split of the dataset.
            root: Root directory where the dataset folder will be created.
            transform: Optional PyG transform applied to data objects before every access.
            pre_transform: Optional PyG transform applied before saving data objects to disk.
            pre_filter: A function that indicates whether a data object should be included in the final dataset.
            generate: If True, generate synthetic graphs instead of downloading.
            cleanup_raw: If True, remove raw files after processing.
            target_vout: Optional target value for vout normalization.
            vout_norm_method: Normalization method for vout labels.
            load_preprocessed: If True, load existing processed objects instead of regenerating.
        """

        #currently downloads everything at once for a single dataset. Up to the user to manually unpack it so far
        self.SOURCES: Dict[str, SourceSpec] = {
            "electronic_circuits_5_eff": SourceSpec(
                url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_ElectronicCircuits/resolve/main/ec_5.zip",
                raw_folder="electronic_circuits_5_eff",
            ),
            "electronic_circuits_5_vout": SourceSpec(
                url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_ElectronicCircuits/resolve/main/ec_5.zip",
                raw_folder="electronic_circuits_5_vout",
            ),
            "electronic_circuits_7_eff": SourceSpec(
                url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_ElectronicCircuits/resolve/main/ec_7.zip",
                raw_folder="electronic_circuit_7_eff",
            ),
            "electronic_circuits_7_vout": SourceSpec(
                url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_ElectronicCircuits/resolve/main/ec_7.zip",
                raw_folder="electronic_circuits_7_vout",
            ),
            "electronic_circuits_10_eff": SourceSpec(
                url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_ElectronicCircuits/resolve/main/ec_10.zip",
                raw_folder="electronic_circuits_10_eff",
            ),
            "electronic_circuits_10_vout": SourceSpec(
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
        self._logger = _logger
        self.cleanup_raw = cleanup_raw
        self.load_preprocessed = load_preprocessed

        # paths
        self.ec_dir = Path(root) / "electroniccircuits"
        self._raw_dir = (self.ec_dir / self.SOURCES[self.dataset_name].raw_folder / "raw")
        
        # Include time window & task in the processed filename to avoid collisions
        self.processed_path = (self.ec_dir /self.SOURCES[self.dataset_name].raw_folder / "processed" / f"{self.dataset_name}_{self.split}.pt")
        super().__init__(str(self.ec_dir), transform, pre_transform, pre_filter)

        self._load_cached_or_prepare(
            processed_path=self.processed_path,
            cleanup_raw=self.cleanup_raw,
            logger=_logger,
        )
        

    def _generate(self, pre_transform, transform) -> None:
        raise NotImplementedError("Dataset generation not supported yet.")

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

    def _load_graphs(self) -> List[Data]:
        if self.generate:
            return self._generate(None, None)

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
        return data_list
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
                port_ids=torch.tensor(datum['port_ids']),
                terminal_ids=torch.tensor(datum['terminal_ids']),
            ))
        return data_list
    
    def get_label(self, target, datum, method='min-max', target_vout=None, statistics=None, y_range=None):
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
    
        
