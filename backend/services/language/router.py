from dataclasses import dataclass, field
from typing import Any

from .entities import ParsedQuery
from .context import ConversationContext
from .intent import Intent


@dataclass
class RouteDecision:
    route: str
    intent: str
    provider: str | None = None
    use_llm: bool = False
    operation: str | None = None
    confidence: float = 0.0
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class NLURouter:
    """
    Decides what subsystem should handle a parsed user request.

    The router does NOT perform the computation.
    It only decides where the request should go.
    """

    def route(
        self,
        query: ParsedQuery,
        context: ConversationContext | None = None,
    ) -> RouteDecision:

        intent = query.intent

        # ---------------------------------------------------------
        # Learning / MCQ
        # ---------------------------------------------------------
        if intent in {
            Intent.GENERATE_MCQ.value,
            Intent.START_MCQ.value,
            Intent.NEXT_MCQ.value,
            Intent.HINT.value,
            Intent.SUBMIT_ANSWER.value,
            Intent.SHOW_SOLUTION.value,
            Intent.SIMILAR_PROBLEM.value,
            Intent.HARDER_PROBLEM.value,
            Intent.EASIER_PROBLEM.value,
        }:
            return RouteDecision(
                route="learning",
                intent=intent,
                operation=intent,
                confidence=query.confidence,
                reason="Learning-related request.",
            )

        # ---------------------------------------------------------
        # Follow-up requests
        # ---------------------------------------------------------
        if intent == Intent.FOLLOW_UP.value:

            metadata = query.metadata

            if metadata.get("resolved_operation") == "evaluate":
                return RouteDecision(
                    route="computation",
                    intent=intent,
                    provider="wolfram",
                    operation="evaluate",
                    confidence=query.confidence,
                    reason="Contextual evaluation of the previous problem.",
                )

            if metadata.get("resolved_operation") == "plot":
                return RouteDecision(
                    route="computation",
                    intent=intent,
                    provider="wolfram",
                    operation="plot",
                    confidence=query.confidence,
                    reason="Contextual plot request.",
                )

            return RouteDecision(
                route="conversation",
                intent=intent,
                use_llm=True,
                operation="follow_up",
                confidence=query.confidence,
                reason="Follow-up requires conversational interpretation.",
            )

        # ---------------------------------------------------------
        # Step explanation
        # ---------------------------------------------------------
        if intent == Intent.EXPLAIN.value:

            if query.metadata.get("step_number") is not None:
                return RouteDecision(
                    route="explanation",
                    intent=intent,
                    use_llm=True,
                    operation="explain_step",
                    confidence=query.confidence,
                    reason="User requested explanation of a specific solution step.",
                    metadata={
                        "step_number": query.metadata["step_number"]
                    },
                )

            return RouteDecision(
                route="reasoning",
                intent=intent,
                use_llm=True,
                operation="explain",
                confidence=query.confidence,
                reason="Conceptual explanation requires language reasoning.",
            )

        # ---------------------------------------------------------
        # Direct computational operations
        # ---------------------------------------------------------
        computational_intents = {
            Intent.CALCULATE.value,
            Intent.SIMPLIFY.value,
            Intent.EXPAND.value,
            Intent.FACTOR.value,
            Intent.DIFFERENTIATE.value,
            Intent.INTEGRATE.value,
            Intent.LIMIT.value,
            Intent.SOLVE.value,
            Intent.ROOTS.value,
            Intent.MAXIMUM.value,
            Intent.MINIMUM.value,
            Intent.MATRIX.value,
            Intent.DETERMINANT.value,
            Intent.INVERSE.value,
            Intent.MEAN.value,
            Intent.MEDIAN.value,
            Intent.VARIANCE.value,
            Intent.STANDARD_DEVIATION.value,
            Intent.PROBABILITY.value,
        }

        if intent in computational_intents:
            return RouteDecision(
                route="computation",
                intent=intent,
                provider="wolfram",
                operation=intent,
                confidence=query.confidence,
                reason="Deterministic computational operation.",
            )

        # ---------------------------------------------------------
        # Plotting
        # ---------------------------------------------------------
        if intent == Intent.PLOT.value:
            return RouteDecision(
                route="computation",
                intent=intent,
                provider="wolfram",
                operation="plot",
                confidence=query.confidence,
                reason="Graphing is handled by the computational engine.",
            )

        # ---------------------------------------------------------
        # Alternative methods
        # ---------------------------------------------------------
        if intent == Intent.ALTERNATIVE_METHOD.value:
            return RouteDecision(
                route="reasoning",
                intent=intent,
                use_llm=True,
                operation="alternative_method",
                confidence=query.confidence,
                reason="Selecting and explaining an alternative method requires reasoning.",
            )

        # ---------------------------------------------------------
        # Unknown / difficult requests
        # ---------------------------------------------------------
        return RouteDecision(
            route="reasoning",
            intent=intent,
            use_llm=True,
            operation="reason",
            confidence=query.confidence,
            reason="No deterministic route matched; use the reasoning layer.",
        )