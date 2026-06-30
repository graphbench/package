from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, TypeAlias, Union

import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import Data

from graphbench._helpers import download_and_unpack, SourceSpec, get_logger
from ._base import GraphDataset


TimeStamp: TypeAlias = Union[int, str]

_FEATURE_PT_PATH = "user_post_embs.pt" #raw files, slightly different name
_TARGETS_PT_PATH = "user_post_counts.pt"
_DEFAULT_TRAIN_END = 20231211
_DEFAULT_PREDICTION_GAPS = [42, 29, 27]


# -----------------------------------------------------------------------------#
# (a) Utilities
# -----------------------------------------------------------------------------#

_logger = get_logger(__name__)


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

def _crop_records(
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

def _aggregate_post_embeddings(
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

def _add_edge_time(df: pd.DataFrame, format='%Y%m%d%H%M', index=2) -> Tuple[Tensor, Tensor]:
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

class BlueSkyDataset(GraphDataset):
    r"""
    Social Networks (BlueSky) datasets.

    Note:
        This class **should not be used directly**, please use :class:`graphbench.Loader` instead to access the provided
        datasets.
        The purpose of this page is merely to provide details on the dataset.


    Overview:
        We provide three single-graph datasets derived from the BlueSky social media platform.
        Each graph represents a social network, where nodes represent users and
        directed edges represent interactions between users.
        The three graphs differ in the type of interaction ("engagement") represented by the edges;
        they can be quotes, replies, or reposts.

        Each interaction is timestamped, and we only construct edges based on interactions that occur in a pre-defined
        time window :math:`(t_0, t_1]`.
        For each user, node features are derived from the content of their posts in the same time window.
        The task is to predict the median number of engagements a user will receive on future posts in the time window
        :math:`(t_1, t_2]`.

        The dataset is based on the data curated by
        `Faila and Rossetti <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0310330>`__.


    Splits:
        We temporally split each graph into training, validation, and test sets to ensure that models are trained
        exclusively on past information and evaluated on later interactions.
        The splits are based on the following points in time:

        - :math:`t_{start}`: February 17th, 2023
        - :math:`t_A`: December 11th, 2023
        - :math:`t_B`: January 22nd, 2024
        - :math:`t_C`: February 20th, 2024
        - :math:`t_{end}`: March 18th, 2024

        These were chosen such that the proportion of posts in the intervals :math:`(t_{start}, t_A]`,
        :math:`(t_A, t_B]`, :math:`(t_B, t_C]`, :math:`(t_C, t_{end}]` amounts to, resp., 55%, 15%, 15%, 15%.
        Using these points, the dataset is split into training, validation, and test splits as follows:

        - Training:   :math:`t_0 = t_{start}, t_1 = t_A, t_2 = t_B`
        - Validation: :math:`t_0 = t_{start}, t_1 = t_B, t_2 = t_C`
        - Test:       :math:`t_0 = t_{start}, t_1 = t_C, t_2 = t_{end}`

        Setting :math:`t_0 = t_{start}` for all splits reflects a realistic scenario where a social network grows over
        time, in the sense that user representations evolve as they generate new content and their connections expand
        as they interact with more users.


    Graph Attributes:
        Each of the three graphs comes with the following attributes:

        .. list-table::
           :header-rows: 1

           * - Attribute name
             - Size
             - Description
           * - ``x``
             - ``[num_nodes, 384]``
             - Node features: aggregated embeddings of the user's posts
           * - ``y``
             - ``[num_nodes, 3]``
             - Targets: median number of engagements received by the user's posts

        The node features ``x`` describe the user with an aggregated representation of the content of their posts in the
        time interval :math:`(t_0, t_1]`, obtained by a pretrained language model.
        For each user, we employed the
        `sentence-transformers/all-MiniLM-L6-v2 <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>`__
        language model to embed the text
        obtained by concatenating the content of each of their posts at the monthly granularity.
        We then aggregated these by averaging over the time interval :math:`(t_0, t_1]`.

        The node targets ``y`` are the median number of engagements received by the user's posts in the prediction
        window :math:`(t_1, t_2]`.
        We applied a logarithmic transformation to reduce skew in these prediction targets.
        We include the ground truth targets for all three engagement types on all graphs for completeness.



    List of Available Datasets:
        We provide three datasets, each consisting of one graph.
        The graphs differ in the type of interaction represented by the edges:

        .. list-table::
           :header-rows: 1

           * - Dataset name
             - An edge from user *u* to user *v* indicates that...
           * - ``bluesky_quotes``
             - ...user *u* quoted a post by user *v*.
           * - ``bluesky_replies``
             - ...user *u* replied to a post by user *v*.
           * - ``bluesky_reposts``
             - ...user *u* reposted a post by user *v*.

        In addition to this, we provide ``socialnetwork`` as a convenience identifier to load all of the above datasets.
    """

    _SOURCES_RAW: Dict[str, SourceSpec] = {
        "bluesky_quotes": SourceSpec(
            url="https://zenodo.org/records/14669616/files/graphs.tar.gz",
            raw_folder="bluesky_graphs",
        ),
        "bluesky_replies": SourceSpec(
            url="https://zenodo.org/records/14669616/files/graphs.tar.gz",
            raw_folder="bluesky_graphs",
        ),
        "bluesky_reposts": SourceSpec(
            url="https://zenodo.org/records/14669616/files/graphs.tar.gz",
            raw_folder="bluesky_graphs",
        ),
    }

    _SOURCES: Dict[str, SourceSpec] = {
        "bluesky_quotes": SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_SocialMedia/resolve/main/quotes.zip",
            raw_folder="bluesky_quotes",
        ),
        "bluesky_replies": SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_SocialMedia/resolve/main/replies.zip",
            raw_folder="bluesky_replies",
        ),
        "bluesky_reposts": SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_SocialMedia/resolve/main/reposts.zip",
            raw_folder="bluesky_reposts",   
        ),
    }


    def __init__(
        self,
        name: Literal["bluesky_quotes", "bluesky_replies", "bluesky_reposts"],
        split: Literal["train", "val", "test"],
        root: Union[str, Path],
        transform: Optional[Callable[[Data], Data]] = None,
        pre_transform: Optional[Callable[[Data], Data]] = None,
        pre_filter: Optional[Callable[[Data], bool]] = None,
        cleanup_raw: bool = False,
        # TODO: This should be removed in the future -- the user will download these files
        load_preprocessed = True,
        feature_file_name: Union[str, Path] = _FEATURE_PT_PATH,
        empty_emb_file_name: Union[str, Path] = "empty.pt",
        target_file_name: Union[str, Path] = _TARGETS_PT_PATH,
    ):
        """
        Args:
            name: Whether to load the quotes, replies, or reposts graph.
            split: Whether to load the train, validation, or test split of the dataset.
                   This splits the graph based on pre-defined time intervals.
            root: Root directory where the ``bluesky`` dataset folder is stored.
            transform: Optional PyG transform applied to data objects before every access.
            pre_transform: Optional PyG transform applied before saving data objects to disk.
            pre_filter: A function that indicates whether a data object should be included in the final dataset.
            cleanup_raw: If True, remove raw files after processing.
            load_preprocessed: If True, load existing processed objects instead of regenerating.
            feature_file_name: Path to torch file containing `dict[user_id] -> list[(ts, Tensor)]`.
            empty_emb_file_name: Path to a torch Tensor used as the "empty" embedding.
            target_file_name: Path to torch file containing `dict[user_id] -> list[(ts, likes, replies, reposts)]`.
        """
        self.name = name.lower()
        if self.name not in self._SOURCES:
            raise ValueError(f"Unsupported dataset name: {self.name}")
        assert split in ["train", "val", "test"], "Only 'train', 'val', 'test', 'all_edges' and 'all_targets' splits are supported."

        self.split = split
        self.source = self._SOURCES_RAW[self.name]
        self.source_features = self._SOURCES[self.name]
        self._logger = _logger
        self.cleanup_raw = cleanup_raw
        self.load_preprocessed = load_preprocessed
        self.pre_transform = pre_transform
        # paths
        self.bluesky_dir = Path(root) / "bluesky"
        self._raw_dir = (self.bluesky_dir / self._SOURCES_RAW[self.name].raw_folder / "raw" )
        # Include time window & task in the processed filename to avoid collisions
        subflag = ""
        self._raw_feature_dir = (self.bluesky_dir / self._SOURCES[self.name].raw_folder / "raw")
        self.processed_path = self.bluesky_dir / self._SOURCES[self.name].raw_folder / "processed" / f"{self.name}{subflag}_{split}.pt"
        super().__init__(str(self.bluesky_dir), transform, pre_transform, pre_filter)

        self.feature_file_name = Path(feature_file_name)
        self.empty_file_name = Path(empty_emb_file_name)
        self.target_file_name = Path(target_file_name)

        # process data if needed
        self._load_cached_or_prepare(
            processed_path=self.processed_path,
            cleanup_raw=self.cleanup_raw,
            logger=_logger,
        )

    def _prepare(self) -> None:
        # (b) Download & unpack helpers
        download_and_unpack(
            source=self.source,
            raw_dir=self._raw_dir,
            processed_dir=self.processed_path,
            logger=_logger,
        )
        download_and_unpack(
            source=self.source_features,
            raw_dir=self._raw_feature_dir,
            processed_dir=self.processed_path,
            logger=_logger,
        )

    def _load_graphs(self) -> List[Data]:
        # Pick default ts_train_end and gap per dataset type
        if self.name in {'bluesky_quotes', 'bluesky_replies', 'bluesky_reposts'}:
            loader = self._load_graphs_common
            loader_kwargs = dict(base_csv_name=f"{self.name.split('_')[-1]}.csv", ts_start=None)
        else:
            raise ValueError(f"Unsupported dataset name: {self.name}")

        # (i) Graph Processing: decide ts_end for loader
        # TODO: using default values for now, could be made graph specific
        ts_train_end = _DEFAULT_TRAIN_END
        prediction_gaps = _DEFAULT_PREDICTION_GAPS
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

        if self.split == 'all_targets':
            _logger.info('Loading target dictionary...')
            target_dict = torch.load(self.target_file_name, weights_only=False)
            ys = list()
            for key in target_dict:
                ys += target_dict[key]
            _logger.info('Setting targets into the PyG data object...')
            for data in data_list:
                data.y = torch.tensor(ys)

        return data_list


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
                edge_time, edge_index = _add_edge_time(df, format='%Y%m%d', index=2)
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

            cropped_feats = _crop_records(post_emb_dict, ts_start=None, ts_end=ts_feat_end)
            user_embs: Dict[str, Tensor] = {
                uid: _aggregate_post_embeddings(seq, empty_emb) for uid, seq in cropped_feats.items()
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
