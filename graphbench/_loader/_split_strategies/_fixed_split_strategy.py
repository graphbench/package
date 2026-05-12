from typing import Dict, Literal, Optional

from ._split_strategy import DatasetFactory, SplitStrategy, TrainValTestSet


class FixedSplitStrategy(SplitStrategy):
    def __init__(self, split_map: Optional[Dict[Literal["train", "valid", "test"], str]] = None) -> None:
        self.split_map = split_map or {"train": "train", "valid": "val", "test": "test"}

    def build(self, factory: DatasetFactory, dataset_name: str) -> TrainValTestSet:
        return {
            key: factory(dataset_name, split_name, None)
            for key, split_name in self.split_map.items()
        }
