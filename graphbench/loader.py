import csv
import os

import requests

from graphbench.co_helpers.split_dataset import split_dataset


class Loader():

    def __init__(self, root, dataset_names, pre_filter=None, pre_transform=None, transform=None, generate_fallback=False, update=False, solver = None, use_satzilla_features = False) -> None:
        self.root = root
        self.dataset_names = dataset_names
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform
        self.transform = transform
        self.generate_fallback = generate_fallback
        self.data_list = []
        self.update = update
        self.solver = solver
        self.use_satzilla_features = use_satzilla_features
        self.generate = False

        if self.generate_fallback:
            self.generate = True
            print("Activated fallback to generate dataset if not found.")

    def _get_dataset_names(self):
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
        csv_path = os.path.join(os.path.dirname(__file__), 'datasets.csv')
        result = []
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Support passing a single dataset name string or an iterable of names
        if isinstance(self.dataset_names, str):
            target_names = {self.dataset_names}
        else:
            try:
                target_names = set(self.dataset_names)
            except Exception:
                target_names = {self.dataset_names}

        with open(csv_path, newline='') as csvfile:
            # skipinitialspace handles cases like: header ", datasets" or values with leading spaces
            reader = csv.DictReader(csvfile, skipinitialspace=True)
            for row in reader:
                # normalize keys (strip any accidental whitespace in header names)
                row = { (k.strip() if isinstance(k, str) else k): v for k, v in row.items() }

                name = row.get('dataset_name')
                # fallback: some CSVs might have a header with a leading space
                datasets_str = row.get('datasets') or row.get(' datasets') or ''

                if not name:
                    continue

                if name in target_names:
                    # datasets_str may be empty or contain only semicolons; split and strip entries
                    datasets = [ds.strip() for ds in datasets_str.split(';') if ds and ds.strip()]
                    result.extend(datasets)
        return result

    def _check_for_updates(self):
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
                print(f"No local version file for {dataset_name}. This could be due to missing dataset files or first-time setup. No update action will be taken.")
                pass

    def load(self):
        # TODO version file does not exist yet, so checking for updates does nothing other than printing a warning
        # self._check_for_updates()
        datasets = self._get_dataset_names()
        for dataset in datasets:
            data = self._loader(dataset)
            self.data_list.append(data)

        return self.data_list
        
    def _loader(self, dataset_name):
        

        if 'algoreas' in dataset_name:
            from graphbench.datasets.algoreas import AlgoReasDataset
            # Special handling for sizegen datasets
            if 'sizegen' in dataset_name:
                train_dataset, valid_dataset = None, None
                test_dataset =  AlgoReasDataset(root=self.root, name=dataset_name, pre_transform=self.pre_transform, transform=self.transform, split="test", generate=self.generate)
            else:
                dataset =  AlgoReasDataset(root=self.root, name=f"{dataset_name}_16", pre_transform=self.pre_transform, transform=self.transform, split="train", generate=self.generate)
                train_dataset, valid_dataset, _ = split_dataset(dataset, 0.99, 0.01, 0)
                if 'flow' in dataset_name:
                    test_dataset =  AlgoReasDataset(root=self.root, name=f"{dataset_name}_64", pre_transform=self.pre_transform, transform=self.transform, split="test", generate=self.generate)
                else:
                    test_dataset =  AlgoReasDataset(root=self.root, name=f"{dataset_name}_128", pre_transform=self.pre_transform, transform=self.transform, split="test", generate=self.generate)

        elif 'bluesky' in dataset_name:
            from graphbench.datasets.bluesky import BlueSkyDataset
            train_dataset =  BlueSkyDataset(root=self.root, name=dataset_name, pre_transform=self.pre_transform, transform=self.transform, split="train", follower_subgraph=False, cleanup_raw=True,load_preprocessed=True)
            valid_dataset =  BlueSkyDataset(root=self.root, name=dataset_name, pre_transform=self.pre_transform, transform=self.transform, split="val", follower_subgraph=False, cleanup_raw=True,load_preprocessed=True)
            test_dataset =  BlueSkyDataset(root=self.root, name=dataset_name, pre_transform=self.pre_transform, transform=self.transform, split="test", follower_subgraph=False, cleanup_raw=True,load_preprocessed=True)

        elif 'chipdesign' in dataset_name:
            from graphbench.datasets.chipdesign import ChipDesignDataset
            train_dataset =  ChipDesignDataset(root=self.root, name=dataset_name, pre_transform=self.pre_transform, transform=self.transform, split="train")
            valid_dataset =  ChipDesignDataset(root=self.root, name=dataset_name, pre_transform=self.pre_transform, transform=self.transform, split="val")
            test_dataset =  ChipDesignDataset(root=self.root, name=dataset_name, pre_transform=self.pre_transform, transform=self.transform, split="test")

        elif 'weather' in dataset_name:
            from graphbench.datasets.weatherforecasting import WeatherforecastingDataset
            dataset =  WeatherforecastingDataset(root=self.root, name=dataset_name, pre_transform=self.pre_transform, transform=self.transform, split="train")
            train_dataset, valid_dataset, test_dataset = split_dataset(dataset, 0.8, 0.1, 0.1)
        elif 'co' in dataset_name:
            from graphbench.datasets.combinatorial_optimization import CODataset
            dataset =  CODataset(root=self.root, name=dataset_name, pre_transform=self.pre_transform, transform=self.transform, split="train", generate=self.generate)
            train_dataset, valid_dataset, test_dataset = split_dataset(dataset, 0.7, 0.15, 0.15)

        elif 'sat' in dataset_name:
            from graphbench.datasets.sat import SATDataset
            dataset =  SATDataset(root=self.root, name=dataset_name, pre_transform=self.pre_transform, transform=self.transform, split="train", generate=self.generate, solver=self.solver, use_satzilla_features=self.use_satzilla_features)
            train_dataset, valid_dataset, test_dataset = split_dataset(dataset, 0.8, 0.1, 0.1)
        elif 'electronic_circuits' in dataset_name:
            from graphbench.datasets.electroniccircuits import ECDataset
            train_dataset =  ECDataset(root=self.root, name=dataset_name, pre_transform=self.pre_transform, transform=self.transform, split="train", generate=self.generate)
            valid_dataset =  ECDataset(root=self.root, name=dataset_name, pre_transform=self.pre_transform, transform=self.transform, split="val", generate=self.generate)
            test_dataset =  ECDataset(root=self.root, name=dataset_name, pre_transform=self.pre_transform, transform=self.transform, split="test", generate=self.generate)

        else:
            raise ValueError(f"Dataset {dataset_name} is not supported.")



        return {
            "train": train_dataset, 
            "valid": valid_dataset,
            "test": test_dataset
        }


if __name__ == "__main__":
    loader = Loader(root="datatest_graphbench", dataset_names="co_ba_small")
    dataset_list = loader.load()
    print(dataset_list)