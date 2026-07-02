from collections.abc import Iterable

import os
from pathlib import Path
from typing import Callable, List, Optional, Union

import requests
from torch_geometric.data import Data, InMemoryDataset

from graphbench._metadata import expand_dataset_names
from ._dataset_registry import DatasetRegistry
from ._split_strategies import AlgoReasSplitStrategy, FixedSplitStrategy, RatioSplitStrategy, TrainValTestSet



class Loader():
    """
    Download and load pre-built GraphBench datasets with standard training, validation, and test splits.

    Datasets are selected via dataset names, which can be either concrete dataset identifiers (e.g. ``"co_rb_small"``)
    or group aliases (e.g. ``"co"``).
    Read the specific dataset page you are interested in to see the available dataset names and aliases:

    - :class:`~graphbench.datasets.AlgoReasDataset`
    - :class:`~graphbench.datasets.BlueSkyDataset`
    - :class:`~graphbench.datasets.ChipDesignDataset`
    - :class:`~graphbench.datasets.CODataset`
    - :class:`~graphbench.datasets.ECDataset`
    - :class:`~graphbench.datasets.SATDataset`
    - :class:`~graphbench.datasets.WeatherforecastingDataset`

    :meth:`load` returns a list of dictionaries containing splits, one per resolved dataset name.
    Each dictionary contains the keys ``"train"``, ``"val"``, and ``"test"``, and each value is a dataset instance such
    as :class:`~graphbench.datasets.CODataset`.


    Example:
        .. code-block:: python

            datasets = Loader(dataset_names="co_rb_small", root="data").load()

        This returns a single dataset like this:

        .. code-block:: python

            [{"train": CODataset(35000), "val": CODataset(7500), "test": CODataset(7500)}]
    """

    def __init__(
        self,
        root: Union[str, Path],
        dataset_names: Union[str, Iterable[str]],
        transform: Optional[Callable[[Data], Data]] = None,
        pre_transform: Optional[Callable[[Data], Data]] = None,
        pre_filter: Optional[Callable[[Data], bool]] = None,
        generate_fallback: bool = False,
        sat_solver: Optional[str] = None,
        use_satzilla_features: bool = False,
    ):
        """
        Create a loader for one or more datasets.

        Args:
            root: Base directory where raw/processed dataset files are stored.
            dataset_names: Dataset selector(s) to resolve and load. This can be a concrete dataset id
                           (e.g. ``"co_rb_small"``) or a group alias (e.g. ``"co"``).
            transform: Optional PyG transform applied to data objects before every access.
            pre_transform: Optional PyG transform applied before saving data objects to disk.
            pre_filter: A function that indicates whether a data object should be included in the final dataset.
            generate_fallback: If ``True``, generate datasets when files are not available locally.
                               Only applies to datasets that support data generation.
            sat_solver: SAT solver name. Only relevant for SAT datasets.
            use_satzilla_features: Whether to include SATzilla features. Only relevant for SAT datasets.
        """
        self.root = root
        self.dataset_names = dataset_names
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.generate_fallback = generate_fallback
        self.data_list: List[TrainValTestSet] = []
        self.sat_solver = sat_solver
        self.use_satzilla_features = use_satzilla_features
        self.generate = False

        if self.generate_fallback:
            self.generate = True
            print("Activated fallback to generate dataset if not found.")

        self._registry = DatasetRegistry()
        self._registry.register(
            lambda name: "algoreas" in name,
            self._make_algoreas_dataset,
            AlgoReasSplitStrategy(),
        )
        self._registry.register(
            lambda name: "bluesky" in name,
            self._make_bluesky_dataset,
            FixedSplitStrategy(),
        )
        self._registry.register(
            lambda name: "chipdesign" in name,
            self._make_chipdesign_dataset,
            FixedSplitStrategy(),
        )
        self._registry.register(
            lambda name: "weather" in name,
            self._make_weather_dataset,
            RatioSplitStrategy(0.8, 0.1, 0.1),
        )
        self._registry.register(
            lambda name: "co" in name,
            self._make_co_dataset,
            RatioSplitStrategy(0.7, 0.15, 0.15),
        )
        self._registry.register(
            lambda name: "sat" in name,
            self._make_sat_dataset,
            RatioSplitStrategy(0.8, 0.1, 0.1),
        )
        self._registry.register(
            lambda name: "electronic_circuits" in name,
            self._make_ec_dataset,
            FixedSplitStrategy(),
        )

    def _get_dataset_names(self) -> List[str]:
        """Read `datasets.csv` and return expanded dataset identifiers.

        The CSV is expected to contain a header with at least
        `dataset_name` and `datasets`. The `datasets` column may contain
        a semicolon-separated list of actual dataset identifiers; each
        identifier is stripped and empty entries are ignored.

        Returns:
            list[str]: A list of resolved dataset identifiers.
        Raises:
            FileNotFoundError: When `datasets.csv` is not found in the
                module directory.
        """
        return expand_dataset_names(self.dataset_names)

    def _check_for_updates(self) -> None:
        # Download the remote version file
        remote_version_url = ""
        try:
            response = requests.get(remote_version_url)
            response.raise_for_status()
            remote_versions = {}
            for line in response.text.strip().splitlines():
                if ';' in line:
                    name, version = line.strip().split(';', 1)
                    remote_versions[name.strip()] = version.strip()
        except Exception as e:
            print(f"Could not download remote version file: {e}")
            return

        # Check local version files for each dataset
        # A plain string should be treated as a single dataset name
        dataset_names = [self.dataset_names] if isinstance(self.dataset_names, str) else self.dataset_names
        for dataset_name in dataset_names:
            local_version_file = os.path.join(self.root, dataset_name, "version.txt")
            if os.path.exists(local_version_file):
                with open(local_version_file, "r") as f:
                    local_version = f.read().strip()
                remote_version = remote_versions.get(dataset_name)
                if remote_version and local_version != remote_version:
                    print(f"Version mismatch for {dataset_name}: local={local_version}, remote={remote_version}")
                elif not remote_version:
                    print(f"No remote version info for {dataset_name}")
            else:
                print(
                    f"No local version file for {dataset_name}. "
                    "This could be due to missing dataset files or first-time setup. No update action will be taken."
                )

    def load(self) -> list[TrainValTestSet]:
        """Resolve the specified datasets and return loaded training, validation, and test sets.

        Returns:
            One dictionary of splits per resolved dataset name.
            Each dictionary has the keys ``"train"``, ``"val"``, and ``"test"``.
        """
        # TODO version file does not exist yet, so checking for updates does nothing other than printing a warning
        # self._check_for_updates()
        datasets = self._get_dataset_names()
        for dataset in datasets:
            data = self._loader(dataset)
            self.data_list.append(data)

        return self.data_list
        
    def _loader(self, dataset_name: str) -> TrainValTestSet:
        return self._registry.build(dataset_name)

    def _make_algoreas_dataset(
        self,
        dataset_name: str,
        split: str,
        name_override: Optional[str] = None,
    ) -> InMemoryDataset:
        from graphbench.datasets import AlgoReasDataset

        return AlgoReasDataset(
            root=self.root,
            name=name_override or dataset_name,
            pre_filter=self.pre_filter,
            pre_transform=self.pre_transform,
            transform=self.transform,
            split=split,
            generate=self.generate,
        )

    def _make_bluesky_dataset(
        self,
        dataset_name: str,
        split: str,
        name_override: Optional[str] = None,
    ) -> InMemoryDataset:
        from graphbench.datasets import BlueSkyDataset

        return BlueSkyDataset(
            root=self.root,
            name=name_override or dataset_name,
            pre_filter=self.pre_filter,
            pre_transform=self.pre_transform,
            transform=self.transform,
            split=split,
            cleanup_raw=False,
            load_preprocessed=True,
        )

    def _make_chipdesign_dataset(
        self,
        dataset_name: str,
        split: str,
        name_override: Optional[str] = None,
    ) -> InMemoryDataset:
        from graphbench.datasets import ChipDesignDataset

        return ChipDesignDataset(
            root=self.root,
            name=name_override or dataset_name,
            pre_filter=self.pre_filter,
            pre_transform=self.pre_transform,
            transform=self.transform,
            split=split,
        )

    def _make_weather_dataset(
        self,
        dataset_name: str,
        split: str,
        name_override: Optional[str] = None,
    ) -> InMemoryDataset:
        from graphbench.datasets import WeatherforecastingDataset

        return WeatherforecastingDataset(
            root=self.root,
            name=name_override or dataset_name,
            pre_filter=self.pre_filter,
            pre_transform=self.pre_transform,
            transform=self.transform,
            split=split,
        )

    def _make_co_dataset(
        self,
        dataset_name: str,
        split: str,
        name_override: Optional[str] = None,
    ) -> InMemoryDataset:
        from graphbench.datasets import CODataset

        return CODataset(
            root=self.root,
            name=name_override or dataset_name,
            pre_filter=self.pre_filter,
            pre_transform=self.pre_transform,
            transform=self.transform,
            split=split,
            generate=self.generate,
        )

    def _make_sat_dataset(
        self,
        dataset_name: str,
        split: str,
        name_override: Optional[str] = None,
    ) -> InMemoryDataset:
        from graphbench.datasets import SATDataset

        return SATDataset(
            root=self.root,
            name=name_override or dataset_name,
            pre_filter=self.pre_filter,
            pre_transform=self.pre_transform,
            transform=self.transform,
            split=split,
            solver=self.sat_solver,
            use_satzilla_features=self.use_satzilla_features,
        )

    def _make_ec_dataset(
        self,
        dataset_name: str,
        split: str,
        name_override: Optional[str] = None,
    ) -> InMemoryDataset:
        from graphbench.datasets import ECDataset

        return ECDataset(
            root=self.root,
            name=name_override or dataset_name,
            pre_filter=self.pre_filter,
            pre_transform=self.pre_transform,
            transform=self.transform,
            split=split,
        )


if __name__ == "__main__":
    loader = Loader(root="datatest_graphbench", dataset_names="co_ba_small")
    dataset_list = loader.load()
    print(dataset_list)
