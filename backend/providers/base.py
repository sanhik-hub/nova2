from abc import ABC, abstractmethod
from typing import Any


class ComputationProvider(ABC):

    name: str = "unknown"

    @abstractmethod
    def calculate(self, query: str) -> dict[str, Any]:
        pass

    def is_available(self) -> bool:
        return True