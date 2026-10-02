from backend.services.provider_router import ProviderRouter
from backend.services.expression_normalizer import normalize_expression
from backend.services.expression_normalizer import (
    normalize_expression_result,
)

class ComputationService:

    def __init__(self):
        self.router = ProviderRouter()

    def compute(
        self,
        query: str,
        operation: str = "compute",
        provider: str = "wolfram",
    ):
        normalization = normalize_expression_result(query)

        normalized_query = normalization.normalized

        try:
            result = self.router.calculate(
                query=normalized_query,
                provider=provider,
            )
        except ValueError as exc:
            return self._error(
                query=query,
                operation=operation,
                error=str(exc),
            )

        return {
            "success": result["success"],
            "operation": operation,
            "input": query,
            "result": result["result"],
            "error": result["error"],
            "engine": result["engine"],
            "steps": [],
            "plots": [],
            "related": [],
        }

    def solve(
        self,
        expression: str,
        variable: str = "x",
        provider: str = "wolfram",
    ):
        query = f"Solve[{expression}, {variable}]"

        return self.compute(
            query=query,
            operation="solve",
            provider=provider,
        )

    def simplify(
        self,
        expression: str,
        provider: str = "wolfram",
    ):
        query = f"Simplify[{expression}]"

        return self.compute(
            query=query,
            operation="simplify",
            provider=provider,
        )

    def factor(
        self,
        expression: str,
        provider: str = "wolfram",
    ):
        query = f"Factor[{expression}]"

        return self.compute(
            query=query,
            operation="factor",
            provider=provider,
        )

    def expand(
        self,
        expression: str,
        provider: str = "wolfram",
    ):
        query = f"Expand[{expression}]"

        return self.compute(
            query=query,
            operation="expand",
            provider=provider,
        )

    def differentiate(
        self,
        expression: str,
        variable: str = "x",
        provider: str = "wolfram",
    ):
        query = f"D[{expression}, {variable}]"

        return self.compute(
            query=query,
            operation="differentiate",
            provider=provider,
        )

    def integrate(
        self,
        expression: str,
        variable: str = "x",
        provider: str = "wolfram",
    ):
        query = f"Integrate[{expression}, {variable}]"

        return self.compute(
            query=query,
            operation="integrate",
            provider=provider,
        )

    def limit(
        self,
        expression: str,
        variable: str,
        value: str,
        provider: str = "wolfram",
    ):
        query = f"Limit[{expression}, {variable} -> {value}]"

        return self.compute(
            query=query,
            operation="limit",
            provider=provider,
        )
    def plot(
        self,
        
        expression: str,
        variable: str = "x",
        xmin: float = -10,
        xmax: float = 10,
        provider: str = "wolfram",
    ):  
        normalized_expression = normalize_expression(expression)
        try:
            selected = self.router.get_provider(provider)

            if not hasattr(selected, "plot"):
                return self._error(
                    query=expression,
                    operation="plot",
                    error=f"Provider '{provider}' does not support plotting.",
                )

            result = selected.plot(
                expression=normalized_expression,
                variable=variable,
                xmin=xmin,
                xmax=xmax,
            )

            return {
                "success": result["success"],
                "operation": "plot",
                "input": expression,
                "result": None,
                "error": result["error"],
                "engine": result["engine"],
                "steps": [],
                "plots": [
                    {
                        "type": "function",
                        "variable": variable,
                        "points": result["result"],
                    }
                ] if result["success"] else [],
                "related": [],
            }

        except ValueError as exc:
            return self._error(
                query=expression,
                operation="plot",
                error=str(exc),
            )
        except Exception as exc:
            # Plotting is an API boundary: unexpected provider/serialization
            # failures must become a structured response instead of a 500.
            return self._error(
                query=expression,
                operation="plot",
                error=f"Plotting failed: {exc}",
            )
    @staticmethod
    def _error(query: str, operation: str, error: str):
        return {
            "success": False,
            "operation": operation,
            "input": query,
            "result": None,
            "error": error,
            "engine": None,
            "steps": [],
            "plots": [],
            "related": [],
        }