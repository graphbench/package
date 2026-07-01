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
    r"""
    Weather forecasting dataset.

    Note:
        This class **should not be used directly**, please use :class:`graphbench.Loader` instead to access the provided
        datasets.
        The purpose of this page is merely to provide details on the dataset.


    Overview:
        We provide a graph-based medium-range weather forecasting dataset derived from the
        `ERA5 <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels>`_ reanalysis dataset.
        We use a down-sampled version of ERA5 with a ``64 x 32`` equiangular grid and a temporal resolution of six
        hours.

        The dataset combines two structures into a single graph:

        - A **grid graph** of ``2,048`` nodes, where each node corresponds to one grid cell and stores the atmospheric
          state at that location.
        - An **icosahedron mesh graph** of ``2,562`` nodes spanning the globe, on which message passing is performed.

        Each grid node is connected by a directed edge to one of the icosahedron mesh nodes; together this yields a
        single graph with ``4,610`` nodes and ``59,667`` edges.
        Each grid node carries 5 surface variables and 6 atmospheric variables defined across 13 pressure levels
        (50 hPa to 1,000 hPa), for a total of ``83`` input channels per grid node.

        The task is to model medium-range weather evolution by predicting the *residual* change in the atmospheric
        state over a fixed 12-hour horizon.
        Following `GraphCast <https://www.science.org/doi/10.1126/science.adi2336>`__, the model ingests the current
        atmospheric state, maps it to the icosahedron mesh, performs several rounds of message passing, and maps the
        result back to the grid; the predicted residual is added to the input to obtain the 12-hour-ahead state.


    List of Available Datasets:
        We currently provide a single dataset, called ``weather``.

        It can be loaded like this:

        .. code:: python

            from graphbench import Loader
            dataset = Loader("data", "weather").load()
    """

    def __init__(
        self,
        name: str,
        split: Literal["train", "val", "test"],
        root: Union[str, Path],
        transform: Optional[Callable[[Data], Data]] = None,
        pre_transform: Optional[Callable[[Data], Data]] = None,
        pre_filter: Optional[Callable[[Data], bool]] = None,
        size : Optional[int] = 64,
    ):
        """
        Args:
            name: Dataset identifier. Must be ``weather_64`` in order to load the provided dataset.
                  Note that this differs from the name provided to :class:`graphbench.Loader`.
            split: Whether to load the train, validation, or test split of the dataset.
            root: Root directory where the dataset folder will be created.
            transform: Optional PyG transform applied to data objects before every access.
            pre_transform: Optional PyG transform applied before saving data objects to disk.
            pre_filter: A function that indicates whether a data object should be included in the final dataset.
            size: The grid size of the dataset. Currently, only ``64`` is supported for loading the provided dataset.
        """
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

    def _prepare(self) -> None:
        """
        Download and unpack the weather data if it is not already cached.
        """

        download_and_unpack(
            source=self.source,
            raw_dir=self._raw_dir,
            processed_dir=self.processed_path,
            logger=_logger,
        )

    def _load_graphs(self) -> List[Data]:
        return self._load_weather_graphs()

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
