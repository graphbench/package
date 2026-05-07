from typing import Callable, List

from ._split_strategies import DatasetFactory, SplitStrategy, TrainValTestSet


class DatasetRegistry:
    def __init__(self) -> None:
        self._entries: List[tuple[Callable[[str], bool], DatasetFactory, SplitStrategy]] = []

    def register(
        self,
        matcher: Callable[[str], bool],
        factory: DatasetFactory,
        split_strategy: SplitStrategy,
    ) -> None:
        self._entries.append((matcher, factory, split_strategy))

    def build(self, dataset_name: str) -> TrainValTestSet:
        for matcher, factory, split_strategy in self._entries:
            if matcher(dataset_name):
                return split_strategy.build(factory, dataset_name)
        raise ValueError(f"Dataset {dataset_name} is not supported.")
