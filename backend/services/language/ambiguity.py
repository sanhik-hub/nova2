import re
from typing import Optional

from .context import ConversationContext
from .entities import ParsedQuery
from .intent import Intent


class ContextResolver:
    """
    Resolves references in follow-up questions using the
    current computational conversation.

    Examples:

        "plot it"
        "evaluate that at x=2"
        "explain step 3"
        "give me an MCQ on this"
        "make it harder"

    These are incomplete in isolation but meaningful in context.
    """

    FOLLOW_UP_MARKERS = (
        "it",
        "this",
        "that",
        "these",
        "those",
        "the answer",
        "the result",
        "the expression",
        "the equation",
        "the problem",
        "this problem",
        "this expression",
    )

    def resolve(self, query, context):
        """
        Resolve a parsed query against the current conversation context.

        This layer handles references such as:
            "plot it"
            "what about x=2?"
            "explain step 3"
            "generate an mcq from this"
        """

        if context is None:
            return query

        # ---------------------------------------------------------
        # Determine whether this is referring to previous context
        # ---------------------------------------------------------

        if self._refers_to_previous_problem(query.original):
            self._inherit_problem_data(query, context)

        # ---------------------------------------------------------
        # Evaluation references
        # ---------------------------------------------------------

        self._resolve_evaluation_reference(query, context)

        # ---------------------------------------------------------
        # Step references
        # ---------------------------------------------------------

        step_number = self._extract_step_number(query.original)

        if step_number is not None:
            query.metadata["step_number"] = step_number

        # ---------------------------------------------------------
        # Learning actions
        # ---------------------------------------------------------

        if query.intent == "generate_mcq":
            query.metadata["learning_action"] = "generate_mcq"

        elif query.intent == "start_mcq":
            query.metadata["learning_action"] = "start_mcq"

        elif query.intent == "next_mcq":
            query.metadata["learning_action"] = "next_mcq"

        elif query.intent == "hint":
            query.metadata["learning_action"] = "hint"

        elif query.intent == "submit_answer":
            query.metadata["learning_action"] = "submit_answer"

        elif query.intent == "show_solution":
            query.metadata["learning_action"] = "show_solution"

        elif query.intent == "similar_problem":
            query.metadata["learning_action"] = "similar_problem"

        elif query.intent == "harder_problem":
            query.metadata["learning_action"] = "harder_problem"

        elif query.intent == "easier_problem":
            query.metadata["learning_action"] = "easier_problem"

        # ---------------------------------------------------------
        # Resolve the actual operation
        # ---------------------------------------------------------

        if query.metadata.get("evaluate_previous_result") and query.value:
            query.metadata["resolved_operation"] = "evaluate"

        elif step_number is not None:
            query.metadata["resolved_operation"] = "explain_step"

        elif query.intent == "generate_mcq":
            query.metadata["resolved_operation"] = "generate_mcq"

        elif query.intent == "start_mcq":
            query.metadata["resolved_operation"] = "start_mcq"

        elif query.intent == "next_mcq":
            query.metadata["resolved_operation"] = "next_mcq"

        elif query.intent == "hint":
            query.metadata["resolved_operation"] = "hint"

        elif query.intent == "show_solution":
            query.metadata["resolved_operation"] = "show_solution"

        elif query.intent == "similar_problem":
            query.metadata["resolved_operation"] = "similar_problem"

        elif query.intent == "harder_problem":
            query.metadata["resolved_operation"] = "harder_problem"

        elif query.intent == "easier_problem":
            query.metadata["resolved_operation"] = "easier_problem"

        elif query.intent == "plot" and query.expression:
            query.metadata["resolved_operation"] = "plot"

        return query
    # ---------------------------------------------------------
    # Detect contextual language
    # ---------------------------------------------------------

    def _refers_to_previous_problem(self, text: str) -> bool:
        text = text.strip().lower()

        markers = [
            "it",
            "this",
            "that",
            "the previous",
            "previous problem",
            "previous result",
            "previous answer",
            "the result",
            "the answer",
            "the expression",
            "the problem",
        ]

        # Explicit contextual requests
        contextual_starts = [
            "plot it",
            "graph it",
            "evaluate it",
            "calculate it",
            "simplify it",
            "factor it",
            "expand it",
            "differentiate it",
            "integrate it",
            "solve it",
            "explain it",
            "explain this",
            "explain that",
            "explain step",
            "show step",
            "generate an mcq from this",
            "generate mcq from this",
            "make an mcq from this",
            "make a quiz from this",
            "make it harder",
            "make it easier",
            "give me a similar problem",
            "give me a similar question",
        ]

        if any(text.startswith(start) for start in contextual_starts):
            return True

        if any(marker in text for marker in markers):
            return True

        return False
    # ---------------------------------------------------------
    # Inherit previous problem
    # ---------------------------------------------------------

    def _inherit_problem_data(
        self,
        query: ParsedQuery,
        context: ConversationContext,
    ) -> None:

        previous = context.current_problem

        if previous is None:
            return

        # ---------------------------------------------------------
        # Placeholder expressions
        # ---------------------------------------------------------

        placeholders = {
            "it",
            "this",
            "that",
            "these",
            "those",
            "this expression",
            "that expression",
            "the expression",
            "the answer",
            "the result",
            "the problem",
            "this problem",
            "that problem",
        }

        if (
            query.expression is None
            or query.expression.strip().lower() in placeholders
        ):
            query.expression = previous.expression

        # ---------------------------------------------------------
        # Previous variable
        # ---------------------------------------------------------

        if query.variable is None:
            query.variable = previous.variable

        # ---------------------------------------------------------
        # Previous bounds
        # ---------------------------------------------------------

        if query.lower_bound is None:
            query.lower_bound = previous.lower_bound

        if query.upper_bound is None:
            query.upper_bound = previous.upper_bound

        # ---------------------------------------------------------
        # Previous conditions
        # ---------------------------------------------------------

        if not query.conditions:
            query.conditions = list(previous.conditions)

        # ---------------------------------------------------------
        # Previous parameters
        # ---------------------------------------------------------

        for key, value in previous.parameters.items():
            query.parameters.setdefault(
                key,
                value,
            )

        query.metadata["contextual"] = True
        query.metadata["source_problem"] = previous.original
    # ---------------------------------------------------------
    # Resolve "at x=2"
    # ---------------------------------------------------------

    def _resolve_evaluation_reference(self, query, context):
        """
        Resolve requests such as:
            "what about at x=2"
            "evaluate it at x=5"
            "what is the answer at x=3"

        The previous problem/expression is retained and the requested
        evaluation point is stored in metadata.
        """

        import re

        text = query.original.strip()

        # Matches:
        # x=2
        # x = 2
        # t=5
        # n = 10
        match = re.search(
            r"\b([A-Za-z_]\w*)\s*=\s*([^\s,;]+)",
            text,
        )

        if not match:
            return query

        variable = match.group(1)
        value = match.group(2)

        query.variable = variable
        query.value = value

        query.metadata["evaluation"] = {
            "variable": variable,
            "value": value,
        }

        # Important: this tells the router that this is an
        # evaluation of the previous result/problem.
        query.metadata["evaluate_previous_result"] = True

        # If the user did not explicitly provide an expression,
        # inherit the previous problem.
        if context is not None and context.current_problem is not None:
            previous = context.current_problem

            if not query.expression or query.expression.lower() in {
                "it",
                "this",
                "that",
                "this expression",
                "that expression",
                "the expression",
                "the answer",
                "the result",
                "the problem",
            }:
                query.expression = previous.expression

            if not query.variable:
                query.variable = previous.variable

        return query
    # ---------------------------------------------------------
    # Step reference
    # ---------------------------------------------------------

    def _extract_step_number(
        self,
        text: str,
    ) -> Optional[int]:

        patterns = [
            r"step\s+(\d+)",
            r"step\s*(\d+)",
            r"number\s+(\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)

            if match:
                return int(match.group(1))

        return None