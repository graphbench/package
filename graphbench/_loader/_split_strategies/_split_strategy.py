from abc import ABC, abstractmethod
from typing import Callable, Dict, Literal, Optional, TypeAlias

from torch_geometric.data import InMemoryDataset


DatasetFactory: TypeAlias = Callable[[str, str, Optional[str]], InMemoryDataset]
TrainValTestSet: TypeAlias = Dict[Literal["train", "val", "test"], InMemoryDataset]


class SplitStrategy(ABC):
    @abstractmethod
    def build(self, factory: DatasetFactory, dataset_name: str) -> TrainValTestSet:
        ...
