from __future__ import annotations

import gc
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from sklearn.decomposition import PCA
from torch_geometric.data import Data, HeteroData
from torch_geometric.io import fs
from tqdm import tqdm

from graphbench._helpers import download_and_unpack, SourceSpec, get_logger
from ._base import GraphDataset


# (0) Constants
_SMALL_N_VARS = 3_000
_MEDIUM_N_VARS = 20_000
# _SMALL_N_CLAUSES = 2000_000
# _SMALL_N_VARS = 500_000
# _MAX_TIME = 60
_MAX_TIME = 6000000000
# _SMALL_N_VARS = 100000000_000
_SMALL_N_CLAUSES = 15_000
_MEDIUM_N_CLAUSES = 90_000



# (i) helper functions

# -----------------------------------------------------------------------------#
# (a) Utilities
# -----------------------------------------------------------------------------#

_logger = get_logger(__name__)


class SATDataset(GraphDataset):
    r"""
    Boolean Satisfiability (SAT) Solving datasets.

    Note:
        This class **should not be used directly**, please use :class:`graphbench.Loader` instead to access the provided datasets.
        The purpose of this page is merely to provide details on the dataset.

        
    Overview:
            The SAT datasets include formulae from diverse real-world applications and synthetically generated instances.
            
            We include tasks for two problem settings:

            - **Performance Prediction (EPM)**: a regression problem whose goal is to predict the computation time of SAT solvers on unseen instances.
            - **Algorithm Selection (AS)**: a multi-class classification problem that aims to select the best performing algorithm for a given SAT instance. 

            Each instance is represented through 3 graph representations, capturing structural views of SAT formulae:

            - **Variable-Clause Graph (VCG)**: a bipartite, undirected graph with a node for each variable :math:`v`  and each clause :math:`c`, where an edge connects a variable to a clause if and only if the variable appears in that clause.
            - **Clause Graph (LCG)**: an undirected graph with one node per clause, where two clauses :math:`c_i` and :math:`c_j` are connected if they share at least one negated literal.
            - **Variable Graph (VG)**: an undirected graph with one node per variable, where two variables :math:`v_i` and :math:`v_j` are connected if they co-occur in at least one clause.

            We provide the same dataset on 3 scales:
            
            - Small: contains only the formulae with up to 3,000 variables and 15,000 clauses.
            - Medium: contains all formulae with up to 20,000 variables and 80,000 clauses.
            - Large: includes all formulae 

            In total, the dataset comprises over 100K problem instances (spanning from a few thousand to over 25M variables and 1.8B clauses). This results in 208,788 graphs ranging from 2 to 20,799 nodes and 2 to 4,109,936 edges.

            Please refer to the `GraphBench paper <https://arxiv.org/abs/2512.04475>`__ for the exact parameters used for formula generation, dataset selection, and solver configurations.
    Splits:
        The SAT solving datasets use a fixed 80% / 10% / 10% split for training, validation, and testing.

    Graph Attributes:
        Each graph has the following attributes:
        
        .. list-table::
           :header-rows: 1

           * - Attribute name
             - Size
             - Description
           * - ``x``
             - ``[num_nodes, 12]``
             - Node features for the SAT graph.
           * - ``y``
             - **EPM**: ``[1]`` 

               **AS**: ``[1, 11]``
             - **EPM**: Runtime of the selected solver. 

               **AS**: Target vector over 11 solvers.

    List of Available Datasets:
        We provide one dataset for each combination of graph encoding and task target.
        These can be loaded with ``sat_{graph_encoding}_{target}``, where ``graph_encoding`` is one of
        ``lcg``,
        ``vcg``,
        or ``vg``,
        and ``target`` is one of ``as`` or ``epm``.

        The valid dataset names for the loader are:

        - ``sat_lcg_as``
        - ``sat_vcg_as``
        - ``sat_vg_as``
        - ``sat_lcg_epm``
        - ``sat_vcg_epm``
        - ``sat_vg_epm``

        For example:

        .. code:: python

            from graphbench import Loader
            # loads the SAT Variable-Clause Graph dataset for Algorithm Selection
            dataset = Loader("data", "sat_vg_as").load()
        
        In addition to this, we provide ``sat`` as a convenience identifier to load all of the above datasets.

    Usage Notes:
        Currently, the loader defaults to using only small formula sizes.
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
        use_satzilla_features: Optional[bool] =False,
        cleanup_raw: bool = False,
        solver: Optional[str] = None,

        # TODO: This should be removed in the future -- the user will download these files
        load_preprocessed = False,):
        """
        Args:
            name: Dataset identifier in the form ``sat_{graph_encoding}_{target}``, e.g. ``sat_lcg_as``.
            split: Whether to load the train, validation, or test split of the dataset.
            root: Root directory where the dataset folder will be created.
            transform: Optional PyG transform applied to data objects before every access.
            pre_transform: Optional PyG transform applied before saving data objects to disk.
            pre_filter: A function that indicates whether a data object should be included in the final dataset.
            generate: If True, generate synthetic graphs instead of downloading.
            use_satzilla_features: If True, include SATZILLA features in the dataset.
            cleanup_raw: If True, remove raw files after processing.
            solver: Optional solver name to filter the dataset for a specific solver. If None, all solvers are included.
            load_preprocessed: If True, load existing processed objects instead of regenerating.
        """


        #currently downloads everything at once for a single dataset. Up to the user to manually unpack it so far

        self.SOURCES: Dict[str, SourceSpec] = {
            "sat_lcg_as": SourceSpec(
                url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_SAT/resolve/main/data_small_lcg.pt.xz",
                raw_folder="sat_lcg_as",
            ),
            "sat_vcg_as": SourceSpec(
                url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_SAT/resolve/main/data_small_vcg.pt.xz",
                raw_folder="sat_vcg_as",
            ),
            "sat_vg_as": SourceSpec(
                url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_SAT/resolve/main/data_small_vg.pt.xz",
                raw_folder="sat_vg_as",
            ),
            "sat_lcg_epm": SourceSpec(
                url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_SAT/resolve/main/data_small_lcg.pt.xz",
                raw_folder="sat_lcg_epm",
            ),
            "sat_vcg_epm": SourceSpec(
                url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_SAT/resolve/main/data_small_vcg.pt.xz",
                raw_folder="sat_vcg_epm",
            ),
            "sat_vg_epm": SourceSpec(
                url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_SAT/resolve/main/data_small_vg.pt.xz",
                raw_folder="sat_vg_epm",
            ),
        }

        self.SOURCE_CSV = SourceSpec(
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
            download_and_unpack(
                source=self.SOURCE_CSV,
                raw_dir=csv_dir,
                processed_dir=csv_dir / "processed",
                logger=_logger,
            )
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
                (self.instances_csv["n_vars"] < _SMALL_N_VARS)
                & (self.instances_csv["n_clauses"] < _SMALL_N_CLAUSES)]

        elif self.formula_sizes == "medium":
            self.instances_csv = self.instances_csv[
                (self.instances_csv["n_vars"] < _MEDIUM_N_VARS)
                & (self.instances_csv["n_clauses"] < _MEDIUM_N_CLAUSES)]
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
        self._logger = _logger
        self.cleanup_raw = cleanup_raw
        self.load_preprocessed = load_preprocessed

        # paths
        self.sat_dir = Path(root) / "sat"
        self._raw_dir = (self.sat_dir / self.SOURCES[self.name].raw_folder / "raw" )
        # Include time window & task in the processed filename to avoid collisions
        self.processed_path = self.sat_dir / self.SOURCES[self.name].raw_folder / "processed"
        super().__init__(str(self.sat_dir), transform, pre_transform, pre_filter)

        self._load_cached_or_prepare(
            processed_path=self.processed_paths[0],
            cleanup_raw=self.cleanup_raw,
            logger=_logger,
        )

    def _create_variable_clause_graph(self,clauses, n_vars):
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


    def _create_literal_clause_graph(self,clauses, n_vars):
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


    def _create_variable_graph(self,clauses, n_vars):
        start = time.time()
        k = 0
        edge_list = set()
        x = torch.zeros((n_vars, 5), dtype=torch.float)
        for clause in clauses:
            num_pos = np.sum(np.array(clause) > 0)

            abs_vars = [abs(v) - 1 for v in clause]

            k += 1
            if k % 10000 == 0:
                if time.time() - start > _MAX_TIME:
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


    def _create_clause_graph(self,clauses, n_vars):
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
                    if time.time() - start > _MAX_TIME:
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
    def _parse_cnf_file(self,file_path):
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


    def _process_file(self,instance, graph_type, pre_transform=None, homogeneous=True):
    
        gc.collect()
        original_file_path = instance["raw_file_names"]


        n_vars, clauses = self._parse_cnf_file(original_file_path)

        if graph_type == "vcg":
            data = self._create_variable_clause_graph(clauses, n_vars)
            if homogeneous:
                data = data.to_homogeneous()
        elif graph_type == "cg":
            data = self._create_clause_graph(clauses, n_vars)
        elif graph_type == "lcg":
            data = self._create_literal_clause_graph(clauses, n_vars)
            if homogeneous:
                data = data.to_homogeneous()
        elif graph_type == "vg":
            data = self._create_variable_graph(clauses, n_vars)

        
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

        if (self.graph_type == "vcg" or self.graph_type == "lcg") and isinstance(data, HeteroData):
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
            print(times)
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
                futures.append(executor.submit(self._process_file, instance.to_dict(), self.graph_type, None, True))
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
            raise NotImplementedError(
                "Data generation for SAT datasets will be implemented in a future release. "
                "Until then, please download the pre-generated datasets using Loader.load()"
            )
            # data_list = self._generate()
        else:
            data_list = self._load_sat_graphs()
        return data_list

    def _load_sat_graphs(self) -> List[Data]:
        filepaths = self._find_matching_files(directory=self._raw_dir, size=self.formula_sizes, graph_type=self.graph_type)
        self.load(filepaths[0])
        return [self.get(i) for i in range(len(self))]

    def _find_matching_files(self,directory, size, graph_type):
        """
        Returns a list of filenames matching the convention in the directory.
        """

        pattern = f"data_{size}_{graph_type}.pt"
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