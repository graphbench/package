from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, TypeAlias


DatasetFactory: TypeAlias = Callable[[str, str, Optional[str]], object]


class SplitStrategy(ABC):
    @abstractmethod
    def build(self, factory: DatasetFactory, dataset_name: str) -> Dict[str, object]:
        ...
