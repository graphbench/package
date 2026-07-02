from abc import ABC, abstractmethod
from typing import Callable, Literal, Optional, TypeAlias

from torch_geometric.data import InMemoryDataset


DatasetFactory: TypeAlias = Callable[[str, str, Optional[str]], InMemoryDataset]
TrainValTestSet: TypeAlias = dict[Literal["train", "val", "test"], InMemoryDataset]


class SplitStrategy(ABC):
    @abstractmethod
    def build(self, factory: DatasetFactory, dataset_name: str) -> TrainValTestSet:
        ...
