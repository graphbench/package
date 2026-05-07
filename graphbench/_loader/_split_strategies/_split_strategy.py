from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, TypeAlias


DatasetFactory: TypeAlias = Callable[[str, str, Optional[str]], Any]


class SplitStrategy(ABC):
    @abstractmethod
    def build(self, factory: DatasetFactory, dataset_name: str) -> Dict[str, Any]:
        ...
