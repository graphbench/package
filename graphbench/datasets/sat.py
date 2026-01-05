"""
sat dataset loader
------------------

This module implements the `SATDataset` class which prepares several SAT
benchmark datasets as PyG `InMemoryDataset` objects. 

The class handles downloading SAT datasets and supplementing labels via csv files. 
"""

import gc
import logging
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from sklearn.decomposition import PCA
from torch_geometric.data import Data, HeteroData, InMemoryDataset
from torch_geometric.io import fs
from tqdm import tqdm

from graphbench.helpers.download import _download_and_unpack


# (0) Constants
SMALL_N_VARS = 3_000
MEDIUM_N_VARS = 20_000
# SMALL_N_CLAUSES = 2000_000
# SMALL_N_VARS = 500_000
# MAX_TIME = 60
MAX_TIME = 6000000000
# SMALL_N_VARS = 100000000_000
SMALL_N_CLAUSES = 15_000
MEDIUM_N_CLAUSES = 90_000



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

class SATDataset(InMemoryDataset):
    def __init__(
        self,
        name: str,
        split: str,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        generate: Optional[bool] = False,
        use_satzilla_features: Optional[bool] =False,
        cleanup_raw: bool = True,
        solver: Optional[str] = None,

        # TODO: This should be removed in the future -- the user will download these files
        load_preprocessed = False,):
        


        #currently downloads everything at once for a single dataset. Up to the user to manually unpack it so far

        self.SOURCES: Dict[str, _SourceSpec] = {
        "sat_lcg_as": _SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_SAT/resolve/main/data_small_lcg_no_trans.pt.xz",
            raw_folder="sat_lcg_as",
        ), 
        "sat_vcg_as": _SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_SAT/resolve/main/data_small_vcg_no_trans.pt.xz",
            raw_folder="sat_vcg_as",
        ),
        "sat_vg_as": _SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_SAT/resolve/main/data_small_vg_no_trans.pt.xz",
            raw_folder="sat_vg_as",
        ),
        "sat_lcg_epm": _SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_SAT/resolve/main/data_small_lcg_no_trans.pt.xz",
            raw_folder="sat_lcg_epm",
        ),
        "sat_vcg_epm": _SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_SAT/resolve/main/data_small_vcg_no_trans.pt.xz",
            raw_folder="sat_vcg_epm",
        ),
        "sat_vg_epm": _SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_SAT/resolve/main/data_small_vg_no_trans.pt.xz",
            raw_folder="sat_vg_epm",
        ),
    }
        self.SOURCE_CSV = _SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_SAT/resolve/main/sat_csv.zip",
            raw_folder="sat_csv",
        )
        self.name_temp = name.replace("_"," ")
        """
        Initialize a SATDataset instance.

        Parameters
        - name (str): Dataset identifier, e.g. 'sat_vg_as'.
        - split (str): One of 'train', 'val', 'test'.
        - root (str|Path): Root dataset directory.
        - use_satzilla_features (bool): Whether to include satzilla meta-features.
        - generate (bool): If True, attempt to generate dataset programmatically (slow).

        Behavior
        The constructor will ensure supplementary CSVs are present (downloading
        them if necessary), determine the dataset type and graph encoding, and
        then attempt to load a cached processed file. If not found, call
        `_prepare()` to build the dataset from raw files.
        """
        csv_dir = Path(root) / self.SOURCE_CSV.raw_folder
        if not csv_dir.exists():
            print(f"Downloading supplementary CSV files to {csv_dir}...")
            _download_and_unpack(source=self.SOURCE_CSV, raw_dir=csv_dir, processed_dir=csv_dir / "processed", logger=logger)
        self.solver = solver
        self.instances_csv = pd.read_csv(Path(root) /"sat_csv"/ "instances_new.csv")
        #self.dataset_name = self.name_temp.lower().split(" ")[0]
        self.task_type = self.name_temp.lower().split(" ")[2]
        self.graph_type = self.name_temp.lower().split(" ")[1]
        self.formula_sizes = "small" #only small formula sizes for now 
        self.use_satzilla_features = use_satzilla_features

        self.runs = pd.read_csv(Path(root) /"sat_csv"/ "runs.csv", index_col=0)
        if self.formula_sizes == "small":
            self.instances_csv = self.instances_csv[
                (self.instances_csv["n_vars"] < SMALL_N_VARS)
                & (self.instances_csv["n_clauses"] < SMALL_N_CLAUSES)]

        elif self.formula_sizes == "medium":
            self.instances_csv = self.instances_csv[
                (self.instances_csv["n_vars"] < MEDIUM_N_VARS)
                & (self.instances_csv["n_clauses"] < MEDIUM_N_CLAUSES)]
        if self.use_satzilla_features:
            self.features = pd.read_csv(Path(root) / "sat_csv" / "features.csv")
            self.features.set_index("filename", inplace=True)
            pca = PCA(n_components=7)
            pca.fit(self.features)
            self.features = pd.DataFrame(
                    pca.transform(self.features), index=self.features.index
                )
        if self.task_type == "as":
            runs = self.runs.copy()
            runs.loc[runs["time"] < 0.05, "time"] = 0.05
            runs.loc[~runs["status"].str.contains("SAT|UNSAT"), "time"] = 5000 * 10
            runs_all = self.instances_csv.merge(runs, on="filename")
            runs_all = runs_all.pivot_table(index="filename", columns="solver_name", values="time")
            self.order = runs_all.sum().sort_values().index.tolist()
        
        self.name = name.lower()
        if self.name not in self.SOURCES:
            raise ValueError(f"Unsupported dataset name: {self.name}")
        assert split in ["train", "val", "test"], "Only 'train', 'val', 'test' splits are supported."


        self.generate = generate
        self.split = split
        self.source = self.SOURCES[self.name]
        self.cleanup_raw = cleanup_raw
        self.load_preprocessed = load_preprocessed

        # paths
        self.sat_dir = Path(root) / "sat"
        self._raw_dir = (self.sat_dir / self.SOURCES[self.name].raw_folder / "raw" )
        # Include time window & task in the processed filename to avoid collisions
        self.processed_path = self.sat_dir / self.SOURCES[self.name].raw_folder / "processed"
        super().__init__(str(self.sat_dir), transform, pre_transform)

        # process data if needed
        if self.processed_path.exists():
            self.load(self.processed_paths[0])
            return

        self._prepare()  # (i) downloads, unpacks, load data + (ii) timestep handle + (e) subgraph + collate
        if self.cleanup_raw:
            self._cleanup()

    def create_variable_clause_graph(self,clauses, n_vars):
        data = HeteroData()
        # data["var"].x = torch.arange(0, n_vars, dtype=torch.float).reshape(-1, 1)
        # data["clause"].x = torch.arange(0, len(clauses), dtype=torch.float).reshape(-1, 1)

        data["var"].x = torch.zeros((n_vars, 9), dtype=torch.float) 
        data["clause"].x = torch.zeros((len(clauses), 9), dtype=torch.float)

        edges = [[], []]
        edge_attr = []
        for i, clause in enumerate(clauses):
            num_pos = np.sum(np.array(clause) > 0)
            num_neg = np.sum(np.array(clause) < 0)

            abs_vals = np.abs(clause)
            unique_abs_vals = set(abs_vals)
            unique_vals = set(clause)
            if len(unique_abs_vals) != len(unique_vals):
                continue
            for var in unique_vals:
                node_id = abs(var) - 1
                edges[0].append(node_id)
                edges[1].append(i)
                edge_attr.append(1 if var > 0 else -1)

                data["var"].x[node_id, 2] += 1 if num_pos == 1 else -1  # is horn
                data["var"].x[node_id, 3] += 1 if var > 0 else 0
                data["var"].x[node_id, 4] += 1 if var < 0 else 0
                data["var"].x[node_id, 5] += 1  # degree

            data["clause"].x[i, 2] = num_pos
            data["clause"].x[i, 3] = num_neg
            data["clause"].x[i, 4] = num_pos / (num_neg + 1e-6)
            data["clause"].x[i, 5] = len(clause)
            data["clause"].x[i, 6] = 1 if len(clause) == 1 else 0
            data["clause"].x[i, 7] = 1 if len(clause) == 2 else 0
            data["clause"].x[i, 8] = 1 if len(clause) == 3 else 0

        data["var"].x[node_id, 6] = data["var"].x[node_id, 3] / (data["var"].x[node_id, 4] + 1e-6)

        data["var", "in", "clause"].edge_index = torch.tensor(edges, dtype=torch.long)
        data["var", "in", "clause"].edge_attr = torch.tensor(
            edge_attr, dtype=torch.float
        ).reshape(-1, 1)
        data["clause", "contains", "var"].edge_index = data[
            "var", "in", "clause"
        ].edge_index.flip(0)
        data["clause", "contains", "var"].edge_attr = torch.tensor(
            edge_attr, dtype=torch.float
        ).reshape(-1, 1)

        data.num_nodes = n_vars + len(clauses)
        data.num_edges = len(edges[0]) if len(edges) > 0 else 0
        return data


    def create_literal_clause_graph(self,clauses, n_vars):
        data = HeteroData()
        # data["literal"].x = torch.arange(0, n_vars * 2, dtype=torch.float).reshape(-1, 1)
        # data["clause"].x = torch.arange(0, len(clauses), dtype=torch.float).reshape(-1, 1)
        data["literal"].x = torch.zeros((n_vars * 2, 9), dtype=torch.float) 
        data["clause"].x = torch.zeros((len(clauses), 9), dtype=torch.float)

        data["literal"].x[:n_vars, 0] = 1
        data["literal"].x[n_vars:, 1] = -1
        data["clause"].x[:, 1] = -2

        edges = [[], []]
        edge_attr = []
        for i, clause in enumerate(clauses):
            unique_vals = np.unique(clause)
            num_pos = np.sum(np.array(clause) > 0)
            num_neg = np.sum(np.array(clause) < 0)

            for var in unique_vals:
                node_id = abs(var) - 1
                other_node_id = node_id
                if var < 0:
                    node_id += n_vars
                else:
                    other_node_id += n_vars

                edges[0].append(node_id)
                edges[1].append(i)
                edge_attr.append(1 if var > 0 else -1)
                data["literal"].x[node_id, 2] += 1 if num_pos == 1 else -1  # is horn
                data["literal"].x[node_id, 3] += 1
                data["literal"].x[other_node_id, 4] += 1
                data["literal"].x[node_id, 5] += 1  # degree

            
            data["clause"].x[i, 2] = num_pos
            data["clause"].x[i, 3] = num_neg
            data["clause"].x[i, 4] = num_pos / (num_neg + 1e-6)
            data["clause"].x[i, 5] = len(clause)
            data["clause"].x[i, 6] = 1 if len(clause) == 1 else 0
            data["clause"].x[i, 7] = 1 if len(clause) == 2 else 0
            data["clause"].x[i, 8] = 1 if len(clause) == 3 else 0

        data["literal"].x[node_id, 6] = data["literal"].x[node_id, 3] / (data["literal"].x[node_id, 4] + 1e-6)
        data["literal", "in", "clause"].edge_index = torch.tensor(edges, dtype=torch.long)
        data["literal", "in", "clause"].edge_attr = torch.tensor(
            edge_attr, dtype=torch.float
        ).reshape(-1, 1)
        data["clause", "contains", "literal"].edge_index = data[
            "literal", "in", "clause"
        ].edge_index.flip(0)
        data["clause", "contains", "literal"].edge_attr = torch.tensor(
            edge_attr, dtype=torch.float
        ).reshape(-1, 1)


        data.num_nodes = n_vars * 2 + len(clauses)
        data.num_edges = len(edges[0]) if len(edges) > 0 else 0

        return data


    def create_variable_graph(self,clauses, n_vars):
        start = time.time()
        k = 0
        edge_list = set()
        x = torch.zeros((n_vars, 5), dtype=torch.float)
        for clause in clauses:
            num_pos = np.sum(np.array(clause) > 0)

            abs_vars = [abs(v) - 1 for v in clause]

            k += 1
            if k % 10000 == 0:
                if time.time() - start > MAX_TIME:
                    raise TimeoutError("Timeout during graph creation")

            for i in range(len(abs_vars)):
                for j in range(i + 1, len(abs_vars)):
                    a, b = abs_vars[i], abs_vars[j]

                    if a == b:
                        continue

                    edge = (a, b) if a < b else (b, a)

                    edge_list.add(edge)

                    x[a, 0] += 1 if num_pos == 1 else -1  # is horn
                    x[a, 1] += 1 if clause[i] > 0 else 0
                    x[a, 2] += 1 if clause[i] < 0 else 0
                    x[a, 3] += 1

                    x[b, 0] += 1 if num_pos == 1 else -1  # is horn
                    x[b, 1] += 1 if clause[j] > 0 else 0
                    x[b, 2] += 1 if clause[j] < 0 else 0
                    x[b, 3] += 1

        x[:, 4] = x[:, 1] / (x[:, 2] + 1e-6)
        edge_index = torch.tensor(list(edge_list), dtype=torch.long).t().contiguous()

        data = Data(edge_index=edge_index)
        data.x = x

        data.num_nodes = n_vars
        data.num_edges = edge_index.size(1) if len(edge_list) > 0 else 0

        return data


    def create_clause_graph(self,clauses, n_vars):
        x = torch.zeros((len(clauses), 7), dtype=torch.float)
        start = time.time()
        k = 0
        clauses_for_lits = {}
        edges = []

        for cid, clause in enumerate(clauses):
            num_pos = np.sum(np.array(clause) > 0)
            num_neg = np.sum(np.array(clause) < 0)

            for var in clause:
                neg_var = -var
                k += 1
                if k % 10000 == 0:
                    if time.time() - start > MAX_TIME:
                        raise TimeoutError("Timeout during graph creation")
                if neg_var in clauses_for_lits:
                    for nc in clauses_for_lits[neg_var]:
                        if nc == cid:
                            continue

                        if nc < cid:
                            edges.append([nc, cid])

                if var not in clauses_for_lits:
                    clauses_for_lits[var] = []
                clauses_for_lits[var].append(cid)
            
            x[cid, 0] = num_pos
            x[cid, 1] = num_neg
            x[cid, 2] = num_pos / (num_neg + 1e-6)
            x[cid, 3] = len(clause)
            x[cid, 4] = 1 if len(clause) == 1 else 0
            x[cid, 5] = 1 if len(clause) == 2 else 0
            x[cid, 6] = 1 if len(clause) == 3 else 0

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        data = Data(edge_index=edge_index)
        data.x = x

        data.num_nodes = len(clauses)
        data.num_edges = edge_index.size(1) if len(edges) > 0 else 0

        return data


    # function from https://github.com/zhaoyu-li/G4SATBench
    def parse_cnf_file(self,file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            tokens = lines[i].strip().split()
            if len(tokens) < 1 or tokens[0] != "p":
                i += 1
            else:
                break

        if i == len(lines):
            return 0, []

        header = lines[i].strip().split()
        n_vars = int(header[2])
        clauses = []

        for line in lines[i + 1 :]:
            tokens = line.strip().split()
            clause = [int(s) for s in tokens[:-1]]
            clauses.append(clause)

        return n_vars, clauses


    def process_file(self,instance, graph_type, pre_transform=None, homogeneous=True):
    
        gc.collect()
        original_file_path = instance["raw_file_names"]


        n_vars, clauses = self.parse_cnf_file(original_file_path)

        if graph_type == "vcg":
            data = self.create_variable_clause_graph(clauses, n_vars)
            if homogeneous:
                data = data.to_homogeneous()
        elif graph_type == "cg":
            data = self.create_clause_graph(clauses, n_vars)
        elif graph_type == "lcg":
            data = self.create_literal_clause_graph(clauses, n_vars)
            if homogeneous:
                data = data.to_homogeneous()
        elif graph_type == "vg":
            data = self.create_variable_graph(clauses, n_vars)

        
        try:
            to_undirected = T.ToUndirected()
            data = to_undirected(data)
        except Exception as e:
            print(f"Error making graph undirected: {e}")
            print(f"File: {original_file_path}")
            
        if pre_transform is not None:
            data = pre_transform(data)
        
        fs.torch_save(data, os.path.join(tempfile.gettempdir(), f"{instance['filename']}.pt"))
        # return data

    def get(self, idx):
        data = super().get(idx)

        if self.graph_type == "vcg" or self.graph_type == "lcg":
            data = data.to_homogeneous()
            data = self.to_undirected(data)

        assert data.is_undirected()
        instance = self.instances_csv.iloc[idx]
        times = self.runs.loc[instance["filename"]]

        if self.use_satzilla_features:
            features = self.features.loc[instance["filename"]]
            features = [features.to_list()] * data.x.size(0)
            feat_tensor = torch.tensor(features, dtype=torch.bfloat16)

        if self.task_type == "epm":
            y = times[times["solver_name"] == self.solver]["time"].values[0]

            if y < 0.05:
                y = 0.05

            status = times[times["solver_name"] == self.solver]["status"].values
            if status not in ["SAT", "UNSAT"]:
                y = 50_000

            y = np.log10(y)
            if self.use_satzilla_features:
                data.x = feat_tensor.reshape(-1, 1)
            data.y = torch.tensor([y], dtype=torch.bfloat16)

            return data
        elif self.task_type == "as":
            y = []
            for solver in self.order:
                t = times[times["solver_name"] == solver]["time"].values[0]
                if t < 0.05:
                    t = 0.05
                status = times[times["solver_name"] == solver]["status"].values[0]
                if status not in ["SAT", "UNSAT"]:
                    t = 50_000
                y.append(t)
            y = np.array(y, dtype=np.float32)
            if self.use_satzilla_features:
                data.x = feat_tensor.reshape(-1, 1)
            data.y = torch.tensor(y, dtype=torch.bfloat16).unsqueeze(0)

            return data


    def _generate(self) -> None:
        futures = []
        #generate the corresponding sat dataset
        with ProcessPoolExecutor(max_workers=64) as executor:       
            for _, instance in tqdm(self.instances_csv.iterrows()):
                futures.append(executor.submit(self.process_file, instance.to_dict(), self.graph_type, self.pre_transform, True))
            # futures = [
            #     executor.submit(process_file, instance.to_dict(), self.graph_type)
            #     for _, instance in self.instances_csv.iterrows()
            # ]

            print("Waiting for results...", flush=True)
            graphs = []
            for i, f in enumerate(tqdm(futures)):
                try:
                    f.result()
                except Exception as e:
                    file = self.instances_csv.iloc[i]
                    print(file, flush=True)
                    print(f"Error processing file: {e}")
                    import traceback

                    traceback.print_exc()
                    print("", flush=True)
                    raise e
            
        print("Combining results...", flush=True)
        graphs = [fs.torch_load(os.path.join(tempfile.gettempdir(), f"{instance['filename']}.pt")) for _, instance in self.instances_csv.iterrows()]
        
        return graphs 

    def _prepare(self) -> None:
        print("Processing...", flush=True)

        # (b) Download & unpack helpers

        #not possible right now 
        if self.generate:
            pass
            #data_list = self._generate()
            #if self.pre_transform is not None:
            #    data_list = [self.pre_transform(d) for d in data_list]
            #data, slices = self.collate(data_list)
            #torch.save((data, slices), self.processed_path)

        else:
            _download_and_unpack(source=self.source, raw_dir=self._raw_dir, processed_dir=self.processed_path, logger=logger)

            loader = self._load_sat_graphs
            loader_kwargs = {}
            loader(**loader_kwargs)
            data_list = [self.get(i) for i in range(len(self))]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]
        self.save(data_list, self.processed_paths[0])
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

    def _load_sat_graphs(self) -> List[Data]:
        filepaths = self._find_matching_files(directory=self._raw_dir, size=self.formula_sizes, graph_type=self.graph_type)
        self.load(filepaths[0])
        return 

    def _find_matching_files(self,directory, size, graph_type):
        """
        Returns a list of filenames matching the convention in the directory.
        """

        pattern = f"data_{size}_{graph_type}_no_trans.pt"
        return [os.path.join(directory, fname)
                for fname in os.listdir(directory)
                if fname == pattern]

    # --- InMemoryDataset API (not used directly but kept for PyG hygiene) -----

    @property
    def raw_file_names(self) -> List[str]:  # unused, we drive our own cache
        return []

    @property
    def processed_file_names(self) -> List[str]:  # unused, we drive our own cache
        return ["data.pt"]

    def process(self):
        #self._prepare()
        return 