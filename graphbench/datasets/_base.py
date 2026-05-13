from pathlib import Path
from abc import ABC, abstractmethod

from torch_geometric.data import Data, InMemoryDataset


class GraphDataset(InMemoryDataset, ABC):
    """Shared lifecycle utilities for GraphBench datasets."""

    @staticmethod
    def _cleanup_path(raw_path, logger=None):
        raw_dir = Path(raw_path)
        if not raw_dir.exists():
            return

        if logger is not None:
            logger.info(f"Cleaning up: {raw_dir}")

        # remove only the dataset-specific temp folder
        for path in sorted(raw_dir.rglob("*"), reverse=True):
            try:
                path.unlink()
            except (IsADirectoryError, PermissionError):
                pass

        try:
            raw_dir.rmdir()
        except OSError:
            pass

    @abstractmethod
    def _prepare(self) -> None:
        """Optional pre-step for downloading/unpacking before loading graphs."""
        return None

    @abstractmethod
    def _load_graphs(self) -> list[Data]:
        """Return a list of Data objects ready for filtering and transforms."""
        raise NotImplementedError

    def _cleanup(self) -> None:
        raw_dir = getattr(self, "_raw_dir", None)
        if raw_dir is None:
            return
        self._cleanup_path(raw_dir, logger=getattr(self, "_logger", None))

    def _clear_processed_cache(self, processed_path, logger=None):
        processed_path = Path(processed_path)
        if processed_path.exists():
            if logger is not None:
                logger.warning(f"Removing corrupted processed cache: {processed_path}")
            try:
                processed_path.unlink()
            except IsADirectoryError:
                self._cleanup_path(processed_path, logger=logger)

        processed_dir = processed_path.parent
        if processed_dir.exists() and not any(processed_dir.iterdir()):
            try:
                processed_dir.rmdir()
            except OSError:
                pass

    def _load_cached_or_prepare(
        self,
        processed_path,
        cleanup_raw=False,
        logger=None,
        load_path=None,
        apply_transforms=True,
    ):
        """Load cached processed data if exists, otherwise prepare, process, and cache the dataset."""

        processed_path = Path(processed_path)
        resolved_load_path = Path(load_path) if load_path is not None else processed_path

        if processed_path.exists():
            try:
                if logger is not None:
                    logger.info(f"Loading cached processed data: {processed_path}")
                self.load(resolved_load_path)
                return
            except Exception as e:
                if logger is not None:
                    logger.warning(
                        f"Failed to load cached processed data at {processed_path}; rebuilding cache: {e}"
                    )
                self._clear_processed_cache(processed_path, logger=logger)

        self._prepare()
        data_list = self._load_graphs()
        if data_list is None:
            data_list = [self.get(i) for i in range(len(self))]

        if apply_transforms:
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

        self.save(data_list, processed_path)
        if logger is not None:
            logger.info(f"Saved processed dataset -> {processed_path}")

        self.load(resolved_load_path)
        if cleanup_raw:
            self._cleanup()


BaseGraphDataset = GraphDataset
