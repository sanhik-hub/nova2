from dataclasses import dataclass, field
from typing import Any, Optional

from .entities import ParsedQuery


@dataclass
class ComputationRecord:
    """
    Records one computational operation performed during a conversation.
    """

    operation: str
    input: str
    result: Any = None
    provider: Optional[str] = None
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationContext:
    """
    Structured state for the currently active computational conversation.

    This is deliberately separate from the raw chat history.

    Chat history answers:
        "What did the user say?"

    Conversation context answers:
        "What are we currently working on?"
    """

    conversation_id: Optional[str] = None

    # Current problem
    current_problem: Optional[ParsedQuery] = None

    # Original user wording
    problem_text: Optional[str] = None

    # Most recent computational result
    current_result: Any = None

    # History of computations performed on the current problem
    computations: list[ComputationRecord] = field(default_factory=list)

    # Generated solution steps
    steps: list[Any] = field(default_factory=list)

    # Graph information
    graphs: list[Any] = field(default_factory=list)

    # Related results / information
    related: list[Any] = field(default_factory=list)

    # Learning / MCQ state
    learning_session: Optional[dict[str, Any]] = None

    # Last generated MCQ
    current_mcq: Optional[dict[str, Any]] = None

    # Conversation metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------
    # Problem
    # ---------------------------------------------------------

    def set_problem(
        self,
        query: ParsedQuery,
    ) -> None:

        self.current_problem = query
        self.problem_text = query.original

        # A new explicit problem should normally start a new
        # computational chain.
        self.current_result = None
        self.computations.clear()
        self.steps.clear()
        self.graphs.clear()
        self.related.clear()
        self.current_mcq = None

    # ---------------------------------------------------------
    # Computations
    # ---------------------------------------------------------

    def add_computation(
        self,
        operation: str,
        input_text: str,
        result: Any = None,
        provider: Optional[str] = None,
        success: bool = True,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ComputationRecord:

        record = ComputationRecord(
            operation=operation,
            input=input_text,
            result=result,
            provider=provider,
            success=success,
            metadata=metadata or {},
        )

        self.computations.append(record)

        if success:
            self.current_result = result

        return record

    # ---------------------------------------------------------
    # Steps
    # ---------------------------------------------------------

    def set_steps(
        self,
        steps: list[Any],
    ) -> None:

        self.steps = list(steps)

    # ---------------------------------------------------------
    # Graphs
    # ---------------------------------------------------------

    def add_graph(
        self,
        graph: Any,
    ) -> None:

        self.graphs.append(graph)

    # ---------------------------------------------------------
    # Related information
    # ---------------------------------------------------------

    def add_related(
        self,
        item: Any,
    ) -> None:

        self.related.append(item)

    # ---------------------------------------------------------
    # MCQ
    # ---------------------------------------------------------

    def set_mcq(
        self,
        mcq: dict[str, Any],
    ) -> None:

        self.current_mcq = mcq

    # ---------------------------------------------------------
    # Learning
    # ---------------------------------------------------------

    def start_learning_session(
        self,
        session: Optional[dict[str, Any]] = None,
    ) -> None:

        self.learning_session = session or {
            "active": True,
            "question_index": 0,
            "questions": [],
            "attempts": [],
            "hints_used": 0,
            "correct": 0,
            "wrong": 0,
            "skipped": 0,
        }

    def end_learning_session(self) -> None:

        if self.learning_session is not None:
            self.learning_session["active"] = False

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def has_problem(self) -> bool:
        return self.current_problem is not None

    def has_result(self) -> bool:
        return self.current_result is not None

    def has_steps(self) -> bool:
        return bool(self.steps)

    def has_mcq(self) -> bool:
        return self.current_mcq is not None

    def is_learning(self) -> bool:
        return bool(
            self.learning_session
            and self.learning_session.get("active")
        )

    # ---------------------------------------------------------
    # Context summary
    # ---------------------------------------------------------

    def summary(self) -> dict[str, Any]:

        problem = None

        if self.current_problem:
            problem = {
                "original": self.current_problem.original,
                "intent": self.current_problem.intent,
                "expression": self.current_problem.expression,
                "variable": self.current_problem.variable,
                "value": self.current_problem.value,
                "lower_bound": self.current_problem.lower_bound,
                "upper_bound": self.current_problem.upper_bound,
                "conditions": self.current_problem.conditions,
                "parameters": self.current_problem.parameters,
            }

        return {
            "conversation_id": self.conversation_id,
            "problem": problem,
            "problem_text": self.problem_text,
            "current_result": self.current_result,
            "computations": [
                {
                    "operation": item.operation,
                    "input": item.input,
                    "result": item.result,
                    "provider": item.provider,
                    "success": item.success,
                }
                for item in self.computations
            ],
            "steps": self.steps,
            "graphs": self.graphs,
            "related": self.related,
            "learning_session": self.learning_session,
            "current_mcq": self.current_mcq,
        }