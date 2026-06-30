"""
weather forecasting dataset loader
----------------------------------

This module implements `WeatherforecastingDataset`, a PyG `InMemoryDataset`
that prepares graph-based weather forecasting examples. It downloads preprocessed weather data
which then can be used in downstream tasks. Furthermore, support for generation of the dataset is given (currently disabled)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Union

from torch_geometric.data import Data

from graphbench._helpers import download_and_unpack, SourceSpec, get_logger
from ._base import GraphDataset


# (i) helper functions

# -----------------------------------------------------------------------------#
# (a) Utilities
# -----------------------------------------------------------------------------#

_logger = get_logger(__name__)


class WeatherforecastingDataset(GraphDataset):
    """
    Benchmark dataset class for weather forecasting graph data.
    Handles downloading, processing, and loading splits for PyG experiments.
    """
    def __init__(
        self,
        name: str,
        split: Literal["train", "val", "test"],
        root: Union[str, Path],
        transform: Optional[Callable[[Data], Data]] = None,
        pre_transform: Optional[Callable[[Data], Data]] = None,
        pre_filter: Optional[Callable[[Data], bool]] = None,
        generate: bool = False,
        size : Optional[int] = 64,
    ):
        #currently downloads everything at once for a single dataset. Up to the user to manually unpack it so far
        self.SOURCES: Dict[str, SourceSpec] = {
            "weather_64": SourceSpec(
                url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_Weather/resolve/main/weather_64.pt",
                raw_folder="weather_64",
            ),
        }
        self.name = name.lower()
        if self.name not in self.SOURCES:
            raise ValueError(f"Unsupported dataset name: {self.name}")
        assert split in ["train", "val", "test"], "Only 'train', 'val', 'test' splits are supported."
        self.generate = generate
        self.split = split
        self.source = self.SOURCES[self.name]
        self._logger = _logger
        self.size = size

        self.weather_dir = Path(root) / "weatherforecasting"
        self._raw_dir = (self.weather_dir / self.SOURCES[self.name].raw_folder) / "raw"
        self.processed_path = self.weather_dir / self.SOURCES[self.name].raw_folder / "processed" / f"{self.name}.pt"
        super().__init__(str(self.weather_dir), transform, pre_transform, pre_filter)

        self._load_cached_or_prepare(
            processed_path=self.processed_path,
            cleanup_raw=False,
            logger=_logger,
        )

    def _generate(self) -> None:
        #generate the corresponding weatherforecasting reasoning dataset
        raise NotImplementedError("Dataset generation not supported yet.")
        #fs = gcsfs.GCSFileSystem(token='anon')

        #mapper = fs.get_mapper('weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr')

        #data = xr.open_zarr(mapper, consolidated=False)

        #single_timestep = data.isel().load()

        #single_timestep.to_zarr("data/weather_64", mode="w", consolidated=True)


        #timestamp = xr.open_zarr("data/weather_64", consolidated=False)

        #print("RAM requirement:", timestamp.nbytes / 1024 / 1024, "MB")

        #data = create_graph_dataset()
        #data_list = [data[i] for i in range(len(data))]
        #return data_list

    def _prepare(self) -> None:
        """
        Download and unpack the weather data if it is not already cached.
        """

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
            data_list = self._generate()
        else:
            data_list = self._load_weather_graphs()
        return data_list

    def _load_weather_graphs(self) -> List[Data]:
        """
        Load weather graph data files matching the dataset split and size.
        """
        filepaths = self._find_matching_files(task=self.name, split=self.split, directory=self._raw_dir, size=self.size)
        self.load(filepaths[0])
        return [self.get(i) for i in range(len(self))]

    def _find_matching_files(self,directory, task, size, split):
        """
        Find and return filenames matching the expected pattern for this dataset split and size.
        """
        pattern = f"weather_{size}.pt"
        print(directory)
        return [os.path.join(directory, fname)
                for fname in os.listdir(directory)
                if fname == pattern]

    # --- InMemoryDataset API (not used directly but kept for PyG hygiene) -----

    @property
    def raw_file_names(self) -> list[str]:
        return []

    @property
    def processed_file_names(self) -> list[str]:
        return [f"{self.name}_{self.split}.pt"]
