"""
co datasets loader
------------------

This module provides `CODataset`, a PyTorch Geometric `InMemoryDataset`
wrapper for the combinatorial optimization (CO) benchmark datasets used in the
graphbench. The class supports two modes:

- Generation: build synthetic graph collections via `co_helpers.datasets` and
    split them into train/val/test (currently disabled).
- Download & load: fetch preprocessed `.pt` files, convert into PyG Data
    objects and cache a processed file for fast loading.

The `name` argument selects among supported dataset variants (e.g. 'ba_small',
'er_large'), and `split` must be one of 'train', 'val', or 'test'.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from torch_geometric.data import Data, InMemoryDataset

from graphbench.co_helpers.datasets import BADataset, ERDataset, RBDataset
from graphbench.co_helpers.split_dataset import split_dataset
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

class CODataset(InMemoryDataset):
    def __init__(
        self,
        name: str,
        split: str,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        target: Optional[str] = None,
        generate: Optional[bool] = False,
        cleanup_raw: bool = True,
        # TODO: This should be removed in the future -- the user will download these files
        load_preprocessed = False,
    ):
        """
        Initialize a CODataset instance.

        Parameters
        - name (str): Dataset identifier such as 'ba_small', 'er_large', etc.
        - split (str): One of 'train', 'val', 'test'.
        - root (str|Path): Root directory where the dataset folder will be created.
        - transform, pre_transform: Optional PyG transforms applied at load time.
        - target (str|None): Optional task variant (unused for unsupervised tasks).
        - generate (bool): If True, generate synthetic graphs instead of downloading.
        - cleanup_raw (bool): Whether to remove raw files after processing.

        """
        #currently downloads everything at once for a single dataset. Up to the user to manually unpack it so far
        self.SOURCES: Dict[str, _SourceSpec] = {
        "ba_small": _SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_CO/resolve/main/ba_small_mis_labeled.tar.gz",
            raw_folder="co_ba_small",
        ), 
        "er_small": _SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_CO/resolve/main/er_small_mis_labeled.tar.gz",
            raw_folder="co_er_small",
        ),
        "rb_small": _SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_CO/resolve/main/rb_small_mis_labeled.tar.gz",
            raw_folder="co_rb_small",
        ),
        "rb_large": _SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_CO/resolve/main/rb_large_mis_labeled.tar.gz",
            raw_folder="co_rb_large",
        ),
        "er_large": _SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_CO/resolve/main/er_large_mis_labeled.tar.gz",
            raw_folder="co_er_large",
        ),
        "ba_large": _SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_CO/resolve/main/ba_large_mis_labeled.tar.gz",
            raw_folder="co_ba_large",
        ),
    }
        self.LABEL_SOURCES : Dict[str, _SourceSpec] = {
        "labels": _SourceSpec(
            url="redacted",
            raw_folder="supervised_labels",
        ),}
        self.root = root
        #self.name_temp = name.replace("_"," ")
        #self.dataset_name = self.name_temp.lower().split(" ")[0]
        #self.size = self.name_temp.lower().split(" ")[1]
        #self.target = self.name_temp.lower().split(" ")[2]
        #self.dataset_name = name.lower()
        self.target = target
        self.dataset_name = name.lower().split("_")[1] + "_" + name.lower().split("_")[2]
        if self.dataset_name not in self.SOURCES:
            raise ValueError(f"Unsupported dataset name: {self.dataset_name}")
        assert split in ["train", "val", "test"], "Only 'train', 'val', 'test' splits are supported."

        self.generate = generate
        self.split = split
        self.source = self.SOURCES[self.dataset_name]
        self.cleanup_raw = cleanup_raw
        self.load_preprocessed = load_preprocessed

        # paths
        self.algoreas_dir = Path(root) / "co"
        
        
        self._raw_dir = (self.algoreas_dir /  self.SOURCES[self.dataset_name].raw_folder / "raw")
        # Include time window & task in the processed filename to avoid collisions
        self.processed_path = self.algoreas_dir /self.SOURCES[self.dataset_name].raw_folder/"processed"/ "data.pt"
        super().__init__(str(self.algoreas_dir), transform, pre_transform)

        # process data if needed
        if self.processed_path.exists():
            self.load(self.processed_path)
            #logger.info(f"Loading cached processed data: {self.processed_path}")
            #if "rb" in self.dataset_name:
            #    data = RBDataset(root=self.algoreas_dir / f"{self.dataset_name}")

            #elif "er" in self.dataset_name:
            #    data = ERDataset(root=self.algoreas_dir / f"{self.dataset_name}")
            #    print(data)

            #elif "ba" in self.dataset_name:
            #    data = BADataset(root=self.algoreas_dir / f"{self.dataset_name}")
            #else:
            #    raise ValueError(f"Dataset generation not supported for {self.dataset_name}")
            #self.data = data
            #print(self.data)
        
        else:
            self._prepare()  # (i) downloads, unpacks, load data + (ii) timestep handle + (e) subgraph + collate
            self.load(self.processed_path)
        if self.cleanup_raw:
            self._cleanup()
        

    def _generate(self, pre_transform, transform) -> None:
        #generate the corresponding algorithmic reasoning dataset
        if "rb" in self.dataset_name:
            data = RBDataset(root=self.root, pre_transform=pre_transform, transform=transform)

        elif "er" in self.dataset_name:
            data = ERDataset(root=self.root, pre_transform=pre_transform, transform=transform)

        elif "ba" in self.dataset_name:
            data = BADataset(root=self.root, pre_transform=pre_transform, transform=transform)
        else:
            raise ValueError(f"Dataset generation not supported for {self.dataset_name}")

        train_data, valid_data, test_data = split_dataset(data, 0.7, 0.15, 0.15)
        if self.split == "train":
            return train_data
        elif self.split == "val":
            return valid_data
        elif self.split == "test":  
            return test_data
        else:
            raise ValueError(f"Unknown split: {self.split}")

    def _prepare(self) -> None:
        # (b) Download & unpack helpers
        if self.generate:
            data = self._generate(self.pre_transform, self.transform)
            self.save(data, self.processed_path)
            logger.info(f"Saved processed dataset -> {self.processed_path}")
        else:
            _download_and_unpack(source=self.source, raw_dir=self._raw_dir, processed_dir=self.processed_path, logger=logger)

            loader = self._load_co_graphs
            loader_kwargs = {}
            loader(**loader_kwargs)
        
            # collate & save
            data_list = [self.get(i) for i in range(len(self))]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            self.save(data_list, self.processed_path)

            #idea: concat the labels to the data object here
            #if "supervised" in self.target:
                #_download_and_unpack(source=self.LABEL_SOURCES["labels"], raw_dir=self.LABEL_SOURCES["labels"].raw_folder, processed_dir=self.processed_path, logger=logger)
                #labels_path = self.load_labels()
                #labels = torch.load(labels_path[0], weights_only=False)
                #for i, data in enumerate(data_list):
                    #data.y = labels[i].y

            #further download the data labels beforehand if needed

            #data, slices = self.collate(data_list)
            #torch.save((data, slices), self.processed_path)
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

    def _load_co_graphs(self) -> List[Data]:
        filepaths = self._find_matching_files(task=self.dataset_name, directory=self._raw_dir)
        self.load(filepaths[0])

    def _find_matching_files(self,directory, task, split: Optional[str] = None, size: Optional[str] = None, target: Optional[str] = None):
        """
        Returns a list of filenames matching the convention in the directory.
        """
        pattern = "data.pt"
        return [os.path.join(directory,"processed", fname)
                for fname in os.listdir(os.path.join(directory, 'processed'))
                if fname == pattern]
    

    def load_labels(self):
        labels_path = self._find_matching_files(task=self.target, directory=self.processed_path)
        #download the labels for the supervised tasks if needed
        return labels_path
    
    def process(self):
        self._prepare()
        

    @property
    def raw_file_names(self) -> list[str]:
        return []

    @property
    def processed_file_names(self) -> list[str]:
        return ['data.pt']
    
        
