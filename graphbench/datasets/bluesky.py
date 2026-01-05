from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, TypeAlias, Union

import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset

from graphbench.helpers.download import _download_and_unpack


TimeStamp: TypeAlias = Union[int, str]
FEATURE_PT_PATH = "user_post_embs.pt" #raw files, slightly different name 
TARGETS_PT_PATH = "user_post_counts.pt"
DEFAULT_TRAIN_END = 20231211
DEFAULT_PREDICTION_GAPS = [
    42,
    29,
    27]
IGNORED_FEED_GRAPHS = [ #another topology we do not use these 
    '#Disability.csv',
    'AcademicSky.csv',
    'BlackSky.csv',
    'BookSky.csv',
    'Game Dev.csv',
    'GreenSky.csv',
    "What's History.csv"
]

# -----------------------------------------------------------------------------#
# (a) Utilities
# -----------------------------------------------------------------------------#

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------#
# (c) Timestamp handling
# -----------------------------------------------------------------------------#

def _add_days_drop_time(t_0, delta):
    t_0 = _default_ts_extractor(t_0)
    date_obj = datetime.strptime(str(t_0), "%Y%m%d")
    new_date_obj = date_obj + timedelta(days=delta)
    return int(new_date_obj.strftime("%Y%m%d"))

def _default_ts_extractor(ts: TimeStamp) -> int:
    """
    Tries to preserve your original `int(str(x)[:-4])` behavior while being safer:

    - If it's an int with >= 8 digits (e.g., 202303011234), drop the last 4 digits.
    - If it's a string with length > 8, drop the last 4 chars.
    - Otherwise, just int(...) it.
    """
    if isinstance(ts, int):
        s = str(ts)
        if len(s) > 8:
            return int(s[:-4])
        return ts
    s = str(ts)
    if len(s) > 8:
        return int(s[:-4])
    return int(s)

def crop_records(
    data: Mapping[str, Sequence[Tuple[TimeStamp, Tensor]]],
    ts_start: Optional[int] = None,
    ts_end: int = None,
    ts_extractor: Callable[[TimeStamp], int] = _default_ts_extractor,
) -> Dict[str, List[Tuple[TimeStamp, Tensor]]]:
    """
    Filter a dict[user_id] -> list[(ts, tensor)] to [ts_start, ts_end].

    Inclusive on ts_end, exclusive on ts_start to match your original logic.
    At least one of ts_start / ts_end must be provided.
    """
    if ts_start is None:
        ts_start = float("-inf")  # no lower bound
    if ts_end is None:
        ts_end = float("inf")  # no upper bound

    def keep(ts: TimeStamp) -> bool:
        t = ts_extractor(ts)
        if ts_start is None:
            return t <= ts_end  # type: ignore[arg-type]
        if ts_end is None:
            return t > ts_start
        return ts_start < t <= ts_end

    out: Dict[str, List[Tuple[TimeStamp, Tensor]]] = {}
    for k, seq in data.items():
        filtered = [pair for pair in seq if keep(pair[0])]
        if filtered:
            out[k] = filtered
    return out

def _filter_edge_index(edge_index: Tensor, valid_nodes: set[int]) -> Tensor:
    """Keep only edges where both endpoints in valid_nodes."""
    src, dst = edge_index
    mask = torch.tensor(
        [(int(s) in valid_nodes) and (int(d) in valid_nodes) for s, d in zip(src.tolist(), dst.tolist())],
        dtype=torch.bool,
    )
    return edge_index[:, mask]

def aggregate_post_embeddings(
    seq: Sequence[Tuple[TimeStamp, Tensor]],
    empty_emb: Tensor,
) -> Tensor:
    """
    Aggregate a list of (ts, embedding) into a single user embedding.
    """
    vals: List[Optional[Tensor]] = []
    for _, t in seq:
        vals.append(None if torch.allclose(t, empty_emb) else t)

    # if strategy == "last":
    for t in reversed(vals):
        if t is not None:
            return t
    return empty_emb
    # if strategy == "mean":
    # nonempty = [t for t in vals if t is not None]
    # return torch.mean(torch.stack(nonempty, dim=0), dim=0)

def _reindex_edge_index(edge_index: Tensor, node_set: set[int]) -> Tuple[Tensor, Dict[int, int]]:
    """Reindex nodes into 0..N-1 and return reverse id_map: new_id -> old_id."""
    old_sorted = sorted(node_set)
    old_to_new = {old: new for new, old in enumerate(old_sorted)}
    remap = torch.tensor([old_to_new[int(x)] for x in edge_index.view(-1)], dtype=torch.long)
    remapped_edge_index = remap.view_as(edge_index)
    id_map_reverse = {new: old for old, new in old_to_new.items()}
    return remapped_edge_index, id_map_reverse

def add_edge_time(df: pd.DataFrame, format='%Y%m%d%H%M', index=2) -> Tuple[Tensor, Tensor]:
    dt = pd.to_datetime(df[df.columns[index]].astype(str), format=format, errors='coerce')
    valid = dt.notna()
    if valid.all():
        edge_index = torch.tensor(df[[df.columns[0], df.columns[1]]].values, dtype=torch.long).t().contiguous()
    else:
        df = df.loc[valid]
        edge_index = torch.tensor(df[[df.columns[0], df.columns[1]]].values, dtype=torch.long).t().contiguous()
        dt = dt.loc[valid]
    edge_time_int = dt.dt.strftime(format).astype('int64').to_numpy()
    edge_time = torch.from_numpy(edge_time_int).to(torch.long)
    return edge_time, edge_index

# -----------------------------------------------------------------------------#
# (d) Dataset
# -----------------------------------------------------------------------------#

@dataclass(frozen=True)
class _SourceSpec:
    url: str
    raw_folder: str  # folder name inside tmp/ where data will appear

class BlueSkyDataset(InMemoryDataset):
    """
    A compact PyG InMemoryDataset for BlueSky graphs.

    - Supports: 'followers', 'feed', 'quotes', 'replies', 'reposts'
    - Process node features from per-post embeddings (aggregated per user)
    - Process targets (likes / reply / repost) from a prebuilt torch dict
    - Optional follower-subgraph via BFS from a high-degree node

    Notes:
    * For large archives, we stream to disk and extract.
    * Feature and target tensors are loaded from user-provided .pt files.

    Parameters
    ----------
    name : str
        One of {'followers','feed','quotes','replies','reposts'}
    split: str
        One of {'train', 'val', 'test'}
    root : str | Path
        Root directory for caching.
    follower_subgraph : bool
        If True (only valid for 'followers'), keep a 3-hop BFS subgraph.
    cleanup_raw : bool
        If False, cleanup tmp downloads after processing.
    feature_file_name : str | Path
        Path to torch file containing dict[user_id] -> list[(ts, Tensor)].
    empty_emb_file_name : str | Path
        Path to a torch Tensor used as the 'empty' embedding.
    target_file_name : str | Path
        Path to torch file: dict[user_id] -> list[(ts, likes, replies, reposts)].
    """

    SOURCES_RAW: Dict[str, _SourceSpec] = {

        "bluesky_quotes": _SourceSpec(
            url="https://zenodo.org/records/14669616/files/graphs.tar.gz",
            raw_folder="bluesky_graphs",
        ),
        "bluesky_replies": _SourceSpec(
            url="https://zenodo.org/records/14669616/files/graphs.tar.gz",
            raw_folder="bluesky_graphs",
        ),
        "bluesky_reposts": _SourceSpec(
            url="https://zenodo.org/records/14669616/files/graphs.tar.gz",
            raw_folder="bluesky_graphs",
        ),
    }

    SOURCES: Dict[str, _SourceSpec] = {
        "bluesky_quotes": _SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_SocialMedia/resolve/main/quotes.zip",
            raw_folder="bluesky_quotes",
        ),
        "bluesky_replies": _SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_SocialMedia/resolve/main/replies.zip",
            raw_folder="bluesky_replies",
        ),
        "bluesky_reposts": _SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_SocialMedia/resolve/main/reposts.zip",
            raw_folder="bluesky_reposts",   
        ),
    }


    def __init__(
        self,
        name: str,
        split: str,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        follower_subgraph: bool = False, #not used for now 
        cleanup_raw: bool = True,
        # TODO: This should be removed in the future -- the user will download these files
        load_preprocessed = True, #to load the preprocessed files 
        feature_file_name: Union[str, Path] = FEATURE_PT_PATH,
        empty_emb_file_name: Union[str, Path] = "empty.pt",
        target_file_name: Union[str, Path] = TARGETS_PT_PATH,
    ):
        self.name = name.lower()
        if self.name not in self.SOURCES:
            raise ValueError(f"Unsupported dataset name: {self.name}")
        assert split in ["train", "val", "test"], "Only 'train', 'val', 'test', 'all_edges' and 'all_targets' splits are supported."

        self.split = split
        self.source = self.SOURCES_RAW[self.name]
        self.source_features = self.SOURCES[self.name]
        self.cleanup_raw = cleanup_raw
        self.load_preprocessed = load_preprocessed
        self.pre_transform = pre_transform
        # paths
        self.bluesky_dir = Path(root) / "bluesky"
        self._raw_dir = (self.bluesky_dir / self.SOURCES_RAW[self.name].raw_folder / "raw" )
        # Include time window & task in the processed filename to avoid collisions
        subflag = ""
        self._raw_feature_dir = (self.bluesky_dir / self.SOURCES[self.name].raw_folder / "raw")
        self.processed_path = self.bluesky_dir / self.SOURCES[self.name].raw_folder / "processed" / f"{self.name}{subflag}_{split}.pt"
        super().__init__(str(self.bluesky_dir), transform, pre_transform)

        self.feature_file_name = Path(feature_file_name)
        self.empty_file_name = Path(empty_emb_file_name)
        self.target_file_name = Path(target_file_name)


        # process data if needed
        if self.processed_path.exists():
            logger.info(f"Loading cached processed data: {self.processed_path}")
            self.load(self.processed_path)
            return

        self._prepare()  # (i) downloads, unpacks, load data + (ii) timestep handle + (e) subgraph + collate
        self.load(self.processed_path)
        if self.cleanup_raw:
            self._cleanup()

    def _prepare(self) -> None:
        # (b) Download & unpack helpers
        _download_and_unpack(self.source, self._raw_dir, Path(self.processed_dir), logger=logger)
        _download_and_unpack(self.source_features, self._raw_feature_dir, Path(self.processed_dir), logger=logger)
        # Pick default ts_train_end and gap per dataset type
       
        if self.name in {'bluesky_quotes', 'bluesky_replies', 'bluesky_reposts'}:
            loader = self._load_graphs_common
            loader_kwargs = dict(base_csv_name=f"{self.name.split('_')[-1]}.csv", ts_start=None)
        else:
            raise ValueError(f"Unsupported dataset name: {self.name}")

        # (i) Graph Processing: decide ts_end for loader
        # TODO: using default values for now, could be made graph specific
        ts_train_end = DEFAULT_TRAIN_END
        prediction_gaps = DEFAULT_PREDICTION_GAPS
        ts_known_data_end, ts_pred_data_end = self._get_time_windows(ts_train_end, prediction_gaps)

        # update loader_kwargs with ts_end
        loader_kwargs["ts_end"] = ts_known_data_end
        loader_kwargs["include_timestamps"] = (self.split == 'all_edges' or self.split == 'all_targets') #todo: check remove and impact of this
        data_list = loader(**loader_kwargs)
        
        if self.split not in {"all_edges", "all_targets"}:
            # (ii) Feature & Target Processing
            for data in data_list:
                x, y, edge_index, _ = self._process_feats_and_targets(
                    edge_index=data.edge_index,
                    ts_feat_end=ts_known_data_end,
                    ts_pred_start=ts_known_data_end,
                    ts_pred_end=ts_pred_data_end,
                )
                data.x = x
                data.y = y
                data.edge_index = edge_index

            # (e) optional followers subgraph
            

            # collate & save
            if self.pre_transform:
                data_list = [self.pre_transform(d) for d in data_list]

        if self.split == 'all_targets':
            logger.info('Loading target dictionary...')
            target_dict = torch.load(self.target_file_name, weights_only=False)
            ys = list()
            for key in target_dict:
                ys += target_dict[key]
            logger.info('Setting targets into the PyG data object...')
            for data in data_list:
                data.y = torch.tensor(ys)

        #data, slices = self.collate(data_list)
        #torch.save((data, slices), self.processed_path)
        self.save(data_list, self.processed_path)
        logger.info(f"Saved processed dataset -> {self.processed_path}")


    def _get_time_windows(self, ts_train_end, prediction_gaps) -> Optional[int]:
        assert len(prediction_gaps) == 3
        if self.split == "train":
            return (ts_train_end, _add_days_drop_time(ts_train_end, prediction_gaps[0]))
        elif self.split == "val":
            t0 = _add_days_drop_time(ts_train_end, prediction_gaps[0])
            t1 = _add_days_drop_time(t0, prediction_gaps[1])
            return (t0, t1)
        elif self.split == "test":
            t0 = _add_days_drop_time(_add_days_drop_time(ts_train_end, prediction_gaps[0]), prediction_gaps[1])
            t1 = _add_days_drop_time(t0, prediction_gaps[2])
            return (t0, t1)
        else:
            raise ValueError(f"Unsupported split: {self.split}")

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

    # -------------------------------------------------------------------------#
    # (i) Graph Processing
    # -------------------------------------------------------------------------#


    def _load_graphs_common(self, base_csv_name: str, ts_start: Optional[int], ts_end: int, include_timestamps: bool = False) -> List[Data]:
        if ts_start is None:
            ts_start = float("-inf")  # no lower bound
        data_list: List[Data] = []
        for f in self._raw_dir.rglob(base_csv_name):
            df = pd.read_csv(f)
            if df.shape[1] < 3:
                continue
            # Check for timestamp column
            if include_timestamps:
                edge_time, edge_index = add_edge_time(df, format='%Y%m%d', index=2)
                data = Data(edge_index=edge_index, edge_time=edge_time)
            else:
                df = df.drop_duplicates(subset=[df.columns[0], df.columns[1]])  
                df = df[(df.iloc[:, 2] > ts_start) & (df.iloc[:, 2] <= ts_end)]
                edge_index = torch.tensor(df.iloc[:, :2].values, dtype=torch.long).t().contiguous()
                data = Data(edge_index=edge_index)
            data_list.append(data)
        return data_list
    
    # --- InMemoryDataset API (not used directly but kept for PyG hygiene) -----

    @property
    def raw_file_names(self) -> List[str]:  # unused, we drive our own cache
        return []

    @property
    def processed_file_names(self) -> List[str]:  # unused, we drive our own cache
        return []

    # -------------------------------------------------------------------------#
    # (ii) Feature and Target Processing
    # -------------------------------------------------------------------------#

    def _process_feats_and_targets(
        self,
        edge_index: Tensor,
        ts_feat_end: int,
        ts_pred_start: int,
        ts_pred_end: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Dict[int, int]]:
        """
        Loads features and targets together, intersects users who have BOTH:
        - a valid feature (≤ ts_feat_end)
        - ≥1 post in the prediction window (ts_pred_start, ts_pred_end]
        Then filters edge_index, reindexes once, and returns aligned x, y.
        """
        if self.load_preprocessed:
            self.file_name = self.name.split('_')[-1]
            keep_uids = torch.load(os.path.join(self._raw_feature_dir, f'keep_uids_{self.file_name}_{self.split}.pt'), weights_only=False)
            x = torch.load(os.path.join(self._raw_feature_dir, f'x_{self.file_name}_{self.split}.pt'), weights_only=False)
            y = torch.load(os.path.join(self._raw_feature_dir, f'y_{self.file_name}_{self.split}.pt'), weights_only=False)
        else:
            # ---- Features (<= ts_feat_end)
            post_emb_dict: Dict[str, List[Tuple[TimeStamp, Tensor]]] = torch.load(
                self.feature_file_name, weights_only=False
            )
            empty_emb: Tensor = torch.load(self.empty_file_name, weights_only=False)

            cropped_feats = crop_records(post_emb_dict, ts_start=None, ts_end=ts_feat_end)
            user_embs: Dict[str, Tensor] = {
                uid: aggregate_post_embeddings(seq, empty_emb) for uid, seq in cropped_feats.items()
            }

            # ---- Targets in (ts_pred_start, ts_pred_end]
            target_dict: Dict[str, List[Tuple[TimeStamp, float, float, float]]] = torch.load(
                self.target_file_name, weights_only=False
            )

            # keep users that actually have posts in prediction window
            target_aggs: Dict[str, Tensor] = {}
            for uid, recs in target_dict.items():
                # filter the records by ts
                win = [r for r in recs if ts_pred_start < _default_ts_extractor(r[0]) <= ts_pred_end]
                if not win:
                    continue
                # r: (ts, likes, reply, repost)  -> columns 1..3 are targets
                mat = torch.tensor([[float(r[1]), float(r[2]), float(r[3])] for r in win], dtype=torch.float32)
                med = torch.median(mat, dim=0).values
                target_aggs[uid] = torch.log1p(med)

            # ---- Intersection: users that have BOTH features and targets
            keep_uids = set(user_embs.keys()).intersection(target_aggs.keys())
            if not keep_uids:
                raise RuntimeError("After intersection, no users have both features and prediction-window posts.")
            torch.save(keep_uids, os.path.join(self.root, 'raw', f'keep_uids_{self.name}_{self.split}.pt'))

        # ---- Filter edges to kept users, then drop isolates by connectivity
        keep_nodes = {int(u) for u in keep_uids}
        edge_index = _filter_edge_index(edge_index, keep_nodes)
        connected_nodes = set(torch.unique(edge_index).tolist())
        if not connected_nodes:
            raise RuntimeError("No connected nodes remain after filtering by features and targets.")

        # ---- Reindex once
        edge_index, id_map_rev = _reindex_edge_index(edge_index, connected_nodes)  # new->old

        if not self.load_preprocessed:

            # Build mapping new_id -> uid string
            new_to_uid = {new: str(old) for new, old in id_map_rev.items()}

            # ---- Build X and Y aligned to new indices
            x_list, y_list = [], []
            for i in range(len(new_to_uid)):
                uid = new_to_uid[i]
                # Both dicts must contain uid by construction
                x_list.append(user_embs[uid])
                y_list.append(target_aggs[uid])
            x = torch.stack(x_list, dim=0)
            y = torch.stack(y_list, dim=0)
            torch.save(x, os.path.join(self.root, 'raw', f'x_{self.name}_{self.split}.pt'))
            torch.save(y, os.path.join(self.root, 'raw', f'y_{self.name}_{self.split}.pt'))
            
        return x, y, edge_index, id_map_rev


 