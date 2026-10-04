from backend.services.provider_router import ProviderRouter
from backend.services.expression_normalizer import normalize_expression
from backend.services.expression_normalizer import (
    normalize_expression_result,
)
import re
import math


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

            plot_expression = self._graph_expression(
                normalized_expression,
                variable,
            )
            solution_result = self._solve_graph_equation(
                normalized_expression,
                variable,
                selected,
            )

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
            return self._error(
                query=expression,
                operation="plot",
                error=f"Plotting failed: {exc}",
            )

    @staticmethod
    def _split_equation(expression: str):
        """
        Split a mathematical equation without confusing == with a second
        assignment character. Comparisons such as >=, <= and != are left
        untouched.
        """
        match = re.search(
            r"(?<![<>=!])(?:==|=)(?![=])",
            expression,
        )
        if not match:
            return None

        return (
            expression[:match.start()].strip(),
            expression[match.end():].strip(),
        )

    @classmethod
    def _graph_expression(cls, expression: str, variable: str) -> str:
        parts = cls._split_equation(expression)

        if parts is None:
            return expression

        left, right = parts

        if left.lower() in {
            "y",
            f"f({variable.lower()})",
        }:
            return right

        # For an equation such as x^3 - 5 == 0, plotting the left-minus-right
        # expression gives the corresponding y=0 curve and therefore makes
        # the equation's roots visible on the x-axis.
        return f"({left}) - ({right})"

    @classmethod
    def _solve_graph_equation(
        cls,
        expression: str,
        variable: str,
        provider,
    ):
        try:
            parts = cls._split_equation(expression)

            if parts is None:
                equation = f"{expression} == 0"
            else:
                left, right = parts
                equation = f"{left} == {right}"

            # NSolve deliberately returns numerical replacement rules. Using
            # Solve here can produce Root[...] objects for cubic and higher
            # equations, which are exact but unnecessarily difficult to
            # display and parse as graph coordinates.
            solved = provider.calculate(
                f"NSolve[{equation}, {variable}, Reals]"
            )

            if not solved.get("success"):
                return {"text": "", "points": []}

            result_text = str(solved.get("result") or "")
            points = []

            number_pattern = (
                rf"{re.escape(variable)}\s*->\s*"
                rf"(-?(?:\d+(?:\.\d*)?|\.\d+)"
                rf"(?:[eE][+-]?\d+)?)"
            )

            for match in re.finditer(number_pattern, result_text):
                x_value = float(match.group(1))

                if math.isfinite(x_value):
                    points.append([x_value, 0.0])

            return {
                "text": result_text,
                "points": points,
            }

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
