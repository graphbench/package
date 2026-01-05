from __future__ import annotations

import gzip
import lzma
import shutil
import tarfile
import time
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import requests

if TYPE_CHECKING:
    from graphbench.datasets.bluesky import _SourceSpec


def _download_and_unpack(source: _SourceSpec, raw_dir: Union[str, Path], processed_dir: Union[str, Path], logger) -> None:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    url = source.url
    filename = url.split("/")[-1]
    local_path = raw_dir / filename
    #local_path = raw_dir / "data_small_vg_no_trans.pt.xz"
    if not processed_dir.exists() or not any(Path(processed_dir).iterdir()):
        # the directory doesn’t exist or it exists but is empty (or the processed dir is missing/empty)
        _stream_download(url, local_path, logger)
        if local_path.suffixes[-2:] == [".pt",".xz"]:
            _unpack_xz(local_path, dest_dir=raw_dir)
        elif local_path.suffixes[-2:] == [".tar", ".gz"]:
            _safe_extract_tar(local_path, raw_dir)
            _gunzip_in_tree(raw_dir)  # also handles nested .gz
        elif local_path.suffix == ".gz":
            _gunzip_file(local_path)
        elif local_path.suffix == ".zip":
            _unpack_zip(local_path, dest_dir=raw_dir)
        else:
            logger.warning(f"Unknown archive type for {local_path}; leaving as-is.")
        print(f"Downloaded and unpacked data to {raw_dir}")
    else:
        logger.info(f"Found existing download dir: {raw_dir}")

def _stream_download(
    url: str,
    dest: Path,
    logger,
    chunk_size: int = 1 << 20,
    timeout: int = 60,
    max_retries: int = 5,
    cooldown_seconds: int = 5,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {url} -> {dest}")
    if url == "redacted":
        return

    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                if r.status_code == 429:
                    if attempt == max_retries:
                        r.raise_for_status()
                    logger.warning(
                        "Received 429 (rate limited) on attempt %d/%d; retrying in %ds",
                        attempt,
                        max_retries,
                        cooldown_seconds,
                    )
                    time.sleep(cooldown_seconds)
                    continue

                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                return
        except requests.RequestException as exc:  # includes timeouts and connection errors
            if attempt == max_retries:
                raise
            logger.warning(
                "Download attempt %d/%d failed (%s); retrying in %ds",
                attempt,
                max_retries,
                exc,
                cooldown_seconds,
            )
            time.sleep(cooldown_seconds)

def _safe_extract_tar(path: Path, dest_dir: Path) -> None:
    """Extract tar.gz safely (prevents path traversal)."""
    def _is_within_directory(directory: Path, target: Path) -> bool:
        try:
            directory = directory.resolve()
            target = target.resolve()
        except Exception:
            return False
        return str(target).startswith(str(directory))

    # Collect top-level entries to detect single-folder archives
    with tarfile.open(path, "r:gz") as tar:
        members = tar.getmembers()
        top_level = set()
        for member in members:
            member_name = member.name
            if not member_name:
                continue
            # Normalize and determine top-most path component
            top = member_name.split("/")[0]
            top_level.add(top)
            member_path = dest_dir / member_name
            if not _is_within_directory(dest_dir, member_path):
                raise RuntimeError(f"Unsafe path in tar: {member.name}")
        tar.extractall(dest_dir)  # safe now

    # Remove the archive file to avoid filename collisions
    path.unlink(missing_ok=True)

    # If extraction produced a single directory at the top level, move its
    # contents up into `dest_dir`. This is more reliable than inspecting
    # member names because some archives include mixed top-level entries.
    children = [p for p in dest_dir.iterdir()]
    # Exclude any leftover archive file with the same name (defensive)
    children = [p for p in children if p.name != path.name]
    if len(children) == 1 and children[0].is_dir():
        inner = children[0]
        for child in inner.iterdir():
            # Move each item from the inner folder into dest_dir
            # `shutil.move` will overwrite if destination exists; if a file
            # already exists we make a best-effort to remove it first.
            dest = dest_dir / child.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(child), str(dest_dir))
        # remove the now-empty top-level folder
        try:
            inner.rmdir()
        except OSError:
            shutil.rmtree(inner)

def _gunzip_file(src: Path, dst: Optional[Path] = None) -> Path:
    dst = dst or src.with_suffix("")  # drop .gz
    with gzip.open(src, "rb") as f_in, open(dst, "wb") as f_out:
        f_out.write(f_in.read())
    src.unlink(missing_ok=True)
    return dst

def _gunzip_in_tree(folder: Path) -> None:
    for p in folder.rglob("*.gz"):
        _gunzip_file(p)

def _unpack_xz(src: Path, dest_dir: Optional[Path] = None):
    # If dest_dir is provided, save the unpacked file there, keeping the original filename (without .xz)
    if dest_dir:
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dst = dest_dir / src.with_suffix("").name
    else:
        raise ValueError("dest_dir must be provided for unpacking .xz files")
    with lzma.open(src, "rb") as f_in, open(dst, "wb") as f_out:
        f_out.write(f_in.read())
    src.unlink(missing_ok=True)
    return 

def _unpack_zip(src: Path, dest_dir: Optional[Path] = None):
    """
    Unpack a .zip file to the specified destination directory.
    Removes the zip file after extraction.
    """
    dest_dir = Path(dest_dir) if dest_dir else src.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
    src.unlink(missing_ok=True)
    return

