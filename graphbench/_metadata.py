import os
from typing import List

import pandas as pd


_CSV_CACHE = {}


def _read_csv_cached(path, **kwargs):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    mtime = os.path.getmtime(path)
    cached = _CSV_CACHE.get(path)
    if cached and cached["mtime"] == mtime:
        return cached["data"].copy()

    data = pd.read_csv(path, **kwargs)
    _CSV_CACHE[path] = {"mtime": mtime, "data": data}
    return data.copy()


def _module_csv_path(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def reset_cache():
    _CSV_CACHE.clear()


def get_master_df():
    return _read_csv_cached(
        _module_csv_path("master.csv"),
        index_col=0,
        keep_default_na=False,
    )


def get_datasets_df():
    df = _read_csv_cached(
        _module_csv_path("datasets.csv"),
        keep_default_na=False,
        skipinitialspace=True,
    )
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    return df


def expand_dataset_names(dataset_names) -> List[str]:
    if isinstance(dataset_names, str):
        target_names = {dataset_names}
    else:
        try:
            target_names = set(dataset_names)
        except Exception:
            target_names = {dataset_names}

    result: List[str] = []
    df = get_datasets_df()
    if "dataset_name" not in df.columns:
        return result

    for _, row in df.iterrows():
        name = row.get("dataset_name")
        if not name or name not in target_names:
            continue

        datasets_str = row.get("datasets", "")
        if pd.isna(datasets_str):
            datasets_str = ""
        if not isinstance(datasets_str, str):
            datasets_str = str(datasets_str)

        datasets = [ds.strip() for ds in datasets_str.split(";") if ds and ds.strip()]
        result.extend(datasets)

    return result
