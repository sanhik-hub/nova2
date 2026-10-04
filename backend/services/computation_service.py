from backend.services.provider_router import ProviderRouter
from backend.services.expression_normalizer import normalize_expression
from backend.services.expression_normalizer import (
    normalize_expression_result,
)
import re

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

            plot_expression = self._graph_expression(normalized_expression, variable)
            solution_result = self._solve_graph_equation(normalized_expression, variable, selected)
            result = selected.plot(
                expression=plot_expression,
                variable=variable,
                xmin=xmin,
                xmax=xmax,
            )

            return {
                "success": result["success"],
                "operation": "plot",
                "input": expression,
                "result": solution_result["text"],
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
                "solution_points": solution_result["points"],
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
    def _graph_expression(expression: str, variable: str) -> str:
        if "=" not in expression:
            return expression
        left, right = expression.split("=", 1)
        if left.strip().lower() in {"y", "f(x)"}:
            return right.strip()
        return f"({left.strip()}) - ({right.strip()})"

    @staticmethod
    def _solve_graph_equation(expression: str, variable: str, provider):
        try:
            if "=" in expression:
                left, right = expression.split("=", 1)
                if left.strip().lower() in {"y", "f(x)"}:
                    equation = f"{right.strip()} == 0"
                else:
                    equation = expression
            else:
                equation = f"{expression} == 0"
            solved = provider.calculate(f"Solve[{equation}, {variable}, Reals]")
            if not solved.get("success"):
                return {"text": "", "points": []}
            result_text = str(solved.get("result") or "")
            points = []
            for match in re.finditer(rf"{re.escape(variable)}\s*->\s*(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", result_text):
                points.append([float(match.group(1)), 0.0])
            return {"text": result_text, "points": points}
        except Exception:
            return {"text": "", "points": []}

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