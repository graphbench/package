"""
weather forecasting dataset loader
----------------------------------

This module implements `WeatherforecastingDataset`, a PyG `InMemoryDataset`
that prepares graph-based weather forecasting examples. It downloads preprocessed weather data
which then can be used in downstream tasks. Furthermore, support for generation of the dataset is given (currently disabled)
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

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


class WeatherforecastingDataset(InMemoryDataset):
    """
    Benchmark dataset class for weather forecasting graph data.
    Handles downloading, processing, and loading splits for PyG experiments.
    """
    def __init__(
            self,
        name: str,
        split: str,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        generate: Optional[bool] = False,
        size : Optional[int] = 64,
        # TODO: This should be removed in the future -- the user will download these files
        load_preprocessed = False,
        *args, 
        **kwargs):
        


        #currently downloads everything at once for a single dataset. Up to the user to manually unpack it so far
        self.SOURCES: Dict[str, _SourceSpec] = {
            "weather_64": _SourceSpec(
                url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_Weather/resolve/main/https%3A/huggingface.co/datasets/log-rwth-aachen/Graphbench_Weather/tree/main",
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
        self.load_preprocessed = load_preprocessed
        self.size = size
        self.weather_dir = Path(root) / "weatherforecasting"
        self._raw_dir = (self.weather_dir / self.SOURCES[self.name].raw_folder) / "raw"
        self.processed_path = self.weather_dir / self.SOURCES[self.name].raw_folder / "processed" / f"{self.name}.pt"
        super().__init__(str(self.weather_dir), transform, pre_transform)

        # process data if needed
        if self.processed_path.exists():
            logger.info(f"Loading cached processed data: {self.processed_path}")
            self.load(self.processed_path)
            return

        self._prepare()  # (i) downloads, unpacks, load data + (ii) timestep handle + (e) subgraph + collate
        self.load(self.processed_path)



    def _generate(self) -> None:
        #generate the corresponding weatherforecasting reasoning dataset
        pass 
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
        Download, unpack, and process the dataset. Applies transforms and saves processed data.
        """
        if self.generate:
            #currently not implemented
            pass
            #data_list = self._generate()
        else:
            _download_and_unpack(source=self.source, raw_dir=self._raw_dir, logger=logger, processed_dir=self.processed_path)
            loader = self._load_weather_graphs
            loader_kwargs = {}
            data_list = loader(**loader_kwargs)
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
        self.save(data_list, self.processed_path)
        logger.info(f"Saved processed dataset -> {self.processed_path}")



    def _cleanup(self) -> None:
        """
        Remove temporary raw data files for this dataset split.
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


    def _load_weather_graphs(self) -> List[Data]:
        """
        Load weather graph data files matching the dataset split and size.
        """
        filepaths = self._find_matching_files(task=self.name, split=self.split, directory=self._raw_dir, size=self.size)
        self.load(filepaths[0])

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
    def raw_file_names(self) -> List[str]:
        """
        Returns list of raw file names (unused)
        """
        return []

    @property
    def processed_file_names(self) -> List[str]:
        """
        Returns list of processed file names
        """
        return [f"{self.name}_{self.split}.pt"]