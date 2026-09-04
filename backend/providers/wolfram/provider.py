from typing import Any

from backend.providers.base import ComputationProvider
from backend.services.wolfram_engine import WolframEngine


class WolframProvider(ComputationProvider):

    name = "wolfram"

    def __init__(self):
        self.engine = WolframEngine()

    def calculate(self, query: str) -> dict[str, Any]:
        result = self.engine.calculate(query)

        if result.success:
            return {
                "success": True,
                "result": result.result,
                "error": None,
                "engine": self.name,
            }

        return {
            "success": False,
            "result": None,
            "error": result.error,
            "engine": self.name,
        }

    def is_available(self) -> bool:
        return self.engine.executable is not None
    def plot(
        self,
        expression: str,
        variable: str = "x",
        xmin: float = -10,
        xmax: float = 10,
    ):
        result = self.engine.plot(
            expression,
            variable,
            xmin,
            xmax,
        )

        if result.success:
            return {
                "success": True,
                "result": result.result,
                "error": None,
                "engine": self.name,
            }

        return {
            "success": False,
            "result": None,
            "error": result.error,
            "engine": self.name,
        }