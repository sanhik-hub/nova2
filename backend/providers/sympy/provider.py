from backend.providers.base import ComputationProvider


class SymPyProvider(ComputationProvider):

    name = "sympy"

    def calculate(self, query: str) -> dict:
        return {
            "success": False,
            "result": None,
            "error": "SymPy provider is not implemented yet.",
            "engine": self.name,
        }