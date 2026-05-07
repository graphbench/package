import os
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Union

import requests

from graphbench._metadata import expand_dataset_names
from ._dataset_registry import DatasetRegistry
from ._split_strategies import AlgoReasSplitStrategy, FixedSplitStrategy, RatioSplitStrategy


if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from torch_geometric.data import Data, InMemoryDataset


class Loader():
    def __init__(
        self,
        root: Union[str, Path],
        dataset_names: Iterable[str],
        pre_filter: Optional[Callable[[Data], bool]] = None,
        pre_transform: Optional[Callable[[Data], Data]] = None,
        transform: Optional[Callable[[Data], Data]] = None,
        generate_fallback: bool = False,
        update: bool = False,
        sat_solver: Optional[str] = None,
        use_satzilla_features: bool = False,
    ) -> None:
        self.root = root
        self.dataset_names = dataset_names
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform
        self.transform = transform
        self.generate_fallback = generate_fallback
        self.data_list: List[Dict[str, Any]] = []
        self.update = update
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
        for dataset_name in self.dataset_names:
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

    def load(self) -> List[Dict[str, Any]]:
        # TODO version file does not exist yet, so checking for updates does nothing other than printing a warning
        # self._check_for_updates()
        datasets = self._get_dataset_names()
        for dataset in datasets:
            data = self._loader(dataset)
            self.data_list.append(data)

        return self.data_list
        
    def _loader(self, dataset_name: str) -> Dict[str, Any]:
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
            follower_subgraph=False,
            cleanup_raw=True,
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
            generate=self.generate,
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
            generate=self.generate,
        )


if __name__ == "__main__":
    loader = Loader(root="datatest_graphbench", dataset_names="co_ba_small")
    dataset_list = loader.load()
    print(dataset_list)
