from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from backend.services.language import (
    LanguageParser,
    ContextResolver,
    ConversationContext,
)
from backend.services.language.router import NLURouter


@dataclass
class ExecutionResult:
    success: bool
    operation: str
    answer: Any = None
    error: Optional[str] = None

    provider: Optional[str] = None
    used_llm: bool = False

    interpretation: dict[str, Any] = field(default_factory=dict)

    steps: list[Any] = field(default_factory=list)
    plots: list[Any] = field(default_factory=list)
    related: list[Any] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)


class ExecutionService:
    """
    Central execution layer for conversational requests.

    Pipeline:

        text
          ↓
        LanguageParser
          ↓
        ContextResolver
          ↓
        NLURouter
          ↓
        provider / reasoning layer
          ↓
        ConversationContext
    """

    def __init__(
        self,
        parser: Optional[LanguageParser] = None,
        resolver: Optional[ContextResolver] = None,
        router: Optional[NLURouter] = None,
        provider_router=None,
    ):
        self.parser = parser or LanguageParser()
        self.resolver = resolver or ContextResolver()
        self.nlu_router = router or NLURouter()

        # Reuse the existing provider router.
        if provider_router is None:
            from backend.services.provider_router import ProviderRouter

            provider_router = ProviderRouter()

        self.provider_router = provider_router

    # ---------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------

    def execute(
        self,
        text: str,
        context: Optional[ConversationContext] = None,
    ) -> ExecutionResult:

        if context is None:
            context = ConversationContext()

        # 1. Parse natural language
        parsed = self.parser.parse(text)

        # 2. Resolve references to previous conversation state
        parsed = self.resolver.resolve(parsed, context)

        # 3. Decide execution route
        decision = self.nlu_router.route(parsed, context)

        # 4. Build interpretation
        interpretation = {
            "intent": parsed.intent,
            "expression": parsed.expression,
            "variable": parsed.variable,
            "value": parsed.value,
            "lower_bound": parsed.lower_bound,
            "upper_bound": parsed.upper_bound,
            "conditions": parsed.conditions,
            "parameters": parsed.parameters,
            "requested_output": parsed.requested_output,
            "confidence": parsed.confidence,
        }

        # 5. Update conversation context
        self._update_context_problem(parsed, context)

        # 6. Execute selected route
        if decision.route == "computation":
            result = self._execute_computation(
                parsed=parsed,
                decision=decision,
                context=context,
            )

        elif decision.route == "reasoning":
            result = self._execute_reasoning(
                parsed=parsed,
                decision=decision,
                context=context,
            )

        elif decision.route == "explanation":
            result = self._execute_explanation(
                parsed=parsed,
                decision=decision,
                context=context,
            )

        elif decision.route == "conversation":
            result = self._execute_conversation(
                parsed=parsed,
                decision=decision,
                context=context,
            )

        elif decision.route == "learning":
            result = self._execute_learning(
                parsed=parsed,
                decision=decision,
                context=context,
            )

        else:
            result = ExecutionResult(
                success=False,
                operation=decision.operation or parsed.intent,
                error=f"Unsupported execution route: {decision.route}",
            )

        # 7. Attach interpretation
        result.interpretation = interpretation

        # 8. Store successful execution
        if result.success:
            context.add_computation(
                operation=result.operation,
                input_text=text,
                result=result.answer,
                provider=result.provider,
                success=True,
                metadata={
                    "intent": parsed.intent,
                    "route": decision.route,
                },
            )

        return result

    # ---------------------------------------------------------
    # CONTEXT
    # ---------------------------------------------------------

    def _update_context_problem(
        self,
        parsed,
        context: ConversationContext,
    ) -> None:

        # ---------------------------------------------------------
        # Contextual request:
        # NEVER replace the existing problem.
        # ---------------------------------------------------------
        if parsed.metadata.get("contextual"):

            if context.has_problem():
                previous = context.current_problem

                if not parsed.expression:
                    parsed.expression = previous.expression

                if not parsed.variable:
                    parsed.variable = previous.variable

                if not parsed.lower_bound:
                    parsed.lower_bound = previous.lower_bound

                if not parsed.upper_bound:
                    parsed.upper_bound = previous.upper_bound

                if not parsed.conditions:
                    parsed.conditions = list(previous.conditions)

                if not parsed.parameters:
                    parsed.parameters = dict(previous.parameters)

            return

        # ---------------------------------------------------------
        # Standalone request:
        # This becomes the new active problem.
        # ---------------------------------------------------------
        if parsed.expression or parsed.original:
            context.set_problem(parsed)
    # ---------------------------------------------------------
    # COMPUTATION
    # ---------------------------------------------------------

    def _execute_computation(
        self,
        parsed,
        decision,
        context: ConversationContext,
    ) -> ExecutionResult:

        operation = decision.operation or parsed.intent
        expression = parsed.expression

        # -----------------------------------------------------
        # Plot
        # -----------------------------------------------------
        if operation == "plot":

            if not expression:
                return ExecutionResult(
                    success=False,
                    operation="plot",
                    error="No expression was available to plot.",
                )

            return self._execute_plot(
                expression=expression,
                variable=parsed.variable or "x",
                parsed=parsed,
                context=context,
            )

        # -----------------------------------------------------
        # Evaluation of previous result
        # -----------------------------------------------------
        if operation == "evaluate":

            return self._execute_evaluation(
                parsed=parsed,
                context=context,
            )

        # -----------------------------------------------------
        # Ordinary Wolfram computation
        # -----------------------------------------------------

        if not expression:
            return ExecutionResult(
                success=False,
                operation=operation,
                error="No mathematical expression was found.",
            )

        try:
            result = self.provider_router.calculate(
                query=expression,
                provider=decision.provider or "wolfram",
            )
        except Exception as exc:
            return ExecutionResult(
                success=False,
                operation=operation,
                error=str(exc),
                provider=decision.provider,
            )

        if not result.get("success"):
            return ExecutionResult(
                success=False,
                operation=operation,
                error=result.get("error") or "Computation failed.",
                provider=result.get("engine"),
            )

        answer = result.get("result")

        context.current_result = answer

        return ExecutionResult(
            success=True,
            operation=operation,
            answer=answer,
            provider=result.get("engine") or decision.provider,
            used_llm=False,
        )

    # ---------------------------------------------------------
    # PLOT
    # ---------------------------------------------------------

    def _execute_plot(
        self,
        expression: str,
        variable: str,
        parsed,
        context: ConversationContext,
    ) -> ExecutionResult:

        provider_name = "wolfram"

        try:
            provider = self.provider_router.get_provider(provider_name)

            if not hasattr(provider, "plot"):
                return ExecutionResult(
                    success=False,
                    operation="plot",
                    error="Selected provider does not support plotting.",
                    provider=provider_name,
                )

            result = provider.plot(
                expression=expression,
                variable=variable,
                xmin=-10,
                xmax=10,
            )

        except Exception as exc:
            return ExecutionResult(
                success=False,
                operation="plot",
                error=str(exc),
                provider=provider_name,
            )

        if not result.get("success"):
            return ExecutionResult(
                success=False,
                operation="plot",
                error=result.get("error") or "Plot generation failed.",
                provider=result.get("engine") or provider_name,
            )

        plot_data = result.get("result")

        context.add_graph(plot_data)

        return ExecutionResult(
            success=True,
            operation="plot",
            answer=None,
            plots=[plot_data],
            provider=result.get("engine") or provider_name,
            used_llm=False,
        )

    # ---------------------------------------------------------
    # EVALUATION
    # ---------------------------------------------------------

    def _execute_evaluation(
        self,
        parsed,
        context: ConversationContext,
    ) -> ExecutionResult:

        expression = parsed.expression
        variable = parsed.variable
        value = parsed.value

        if not expression:
            return ExecutionResult(
                success=False,
                operation="evaluate",
                error="No previous expression is available.",
            )

        if not variable or value is None:
            return ExecutionResult(
                success=False,
                operation="evaluate",
                error="Both a variable and evaluation value are required.",
            )

        # Use Wolfram for exact evaluation.
        query = f"{expression} /. {variable} -> {value}"

        try:
            result = self.provider_router.calculate(
                query=query,
                provider="wolfram",
            )
        except Exception as exc:
            return ExecutionResult(
                success=False,
                operation="evaluate",
                error=str(exc),
                provider="wolfram",
            )

        if not result.get("success"):
            return ExecutionResult(
                success=False,
                operation="evaluate",
                error=result.get("error") or "Evaluation failed.",
                provider=result.get("engine") or "wolfram",
            )

        answer = result.get("result")

        context.current_result = answer

        return ExecutionResult(
            success=True,
            operation="evaluate",
            answer=answer,
            provider=result.get("engine") or "wolfram",
            used_llm=False,
            metadata={
                "expression": expression,
                "variable": variable,
                "value": value,
            },
        )

    # ---------------------------------------------------------
    # REASONING
    # ---------------------------------------------------------

    def _execute_reasoning(
        self,
        parsed,
        decision,
        context: ConversationContext,
    ) -> ExecutionResult:

        prompt = self._build_reasoning_prompt(
            parsed=parsed,
            context=context,
        )

        try:
            result = self.provider_router.generate(
                prompt=prompt,
                max_tokens=260,
                temperature=0.0,
                system_prompt=(
                    "You are a mathematical reasoning assistant. "
                    "Give rigorous, concise reasoning. "
                    "Do not invent computational results. "
                    "When a claim requires symbolic verification, "
                    "state the mathematical argument clearly."
                ),
                enable_thinking=False,
            )

        except Exception as exc:
            return ExecutionResult(
                success=False,
                operation=decision.operation or "reason",
                error=str(exc),
                provider="local_llm",
            )

        if not result.get("success"):
            return ExecutionResult(
                success=False,
                operation=decision.operation or "reason",
                error=result.get("error") or "Reasoning failed.",
                provider="local_llm",
            )

        answer = result.get("result")

        return ExecutionResult(
            success=True,
            operation=decision.operation or "reason",
            answer=answer,
            provider="local_llm",
            used_llm=True,
        )

    # ---------------------------------------------------------
    # EXPLANATION
    # ---------------------------------------------------------

    def _execute_explanation(
        self,
        parsed,
        decision,
        context: ConversationContext,
    ) -> ExecutionResult:

        step_number = parsed.metadata.get("step_number")

        # ---------------------------------------------------------
        # Specific step requested
        # ---------------------------------------------------------
        if step_number is not None:

            # Do NOT call the LLM if there are no stored solution steps.
            if not context.has_steps():
                return ExecutionResult(
                    success=False,
                    operation="explain_step",
                    answer=None,
                    error=(
                        f"Step {step_number} is not available because "
                        "the current solution does not contain stored steps yet."
                    ),
                    provider=None,
                    used_llm=False,
                )

            # Validate requested step number.
            if step_number < 1 or step_number > len(context.steps):
                return ExecutionResult(
                    success=False,
                    operation="explain_step",
                    answer=None,
                    error=(
                        f"Step {step_number} does not exist. "
                        f"The current solution has {len(context.steps)} steps."
                    ),
                    provider=None,
                    used_llm=False,
                )

            # -----------------------------------------------------
            # Retrieve the already-generated solution step.
            # -----------------------------------------------------
            step = context.steps[step_number - 1]

            if isinstance(step, dict):
                content = (
                    step.get("content")
                    or step.get("text")
                    or step.get("explanation")
                    or str(step)
                )
            else:
                content = str(step)

            # -----------------------------------------------------
            # Return the stored step directly.
            #
            # IMPORTANT:
            # No LLM call is made here.
            # The Solution Builder will eventually provide the
            # structured/verified step.
            # -----------------------------------------------------
            answer = (
                f"### Step {step_number}\n\n"
                f"{content}"
            )

            return ExecutionResult(
                success=True,
                operation="explain_step",
                answer=answer,
                provider="solution_builder",
                used_llm=False,
                steps=[step],
            )

        # ---------------------------------------------------------
        # General explanation request
        #
        # Examples:
        #   "explain this"
        #   "explain the solution"
        #   "why does this work?"
        # ---------------------------------------------------------
        return self._execute_reasoning(
            parsed=parsed,
            context=context,
        )
    def _explain_stored_step(
        self,
        step_number: int,
        step,
        context: ConversationContext,
    ) -> ExecutionResult:

        if isinstance(step, dict):
            content = step.get("content") or step.get("text") or str(step)
        else:
            content = str(step)

        # For now, return the verified step directly.
        # We can later add an optional fast explanation layer.
        answer = (
            f"### Step {step_number}\n\n"
            f"{content}"
        )

        return ExecutionResult(
            success=True,
            operation="explain_step",
            answer=answer,
            provider="solution_builder",
            used_llm=False,
            steps=[step],
        )    
    # ---------------------------------------------------------
    # CONVERSATION / FOLLOW-UP
    # ---------------------------------------------------------

    def _execute_conversation(
        self,
        parsed,
        decision,
        context: ConversationContext,
    ) -> ExecutionResult:

        prompt = self._build_reasoning_prompt(
            parsed=parsed,
            context=context,
        )

        try:

            result = self.provider_router.generate(
                prompt=prompt,
                max_tokens=260,
                temperature=0.0,
                system_prompt=(
                    "You are a conversational mathematical assistant. "
                    "Use the previous problem and results when relevant. "
                    "Do not invent unavailable information."
                ),
                enable_thinking=False,
            )

        except Exception as exc:

            return ExecutionResult(
                success=False,
                operation="follow_up",
                error=str(exc),
                provider="local_llm",
            )

        if not result.get("success"):

            return ExecutionResult(
                success=False,
                operation="follow_up",
                error=result.get("error") or "Conversation processing failed.",
                provider="local_llm",
            )

        return ExecutionResult(
            success=True,
            operation="follow_up",
            answer=result.get("result"),
            provider="local_llm",
            used_llm=True,
        )

    # ---------------------------------------------------------
    # LEARNING
    # ---------------------------------------------------------

    def _execute_learning(
        self,
        parsed,
        decision,
        context: ConversationContext,
    ) -> ExecutionResult:

        # Learning engine will be implemented separately.
        # For now, expose a clean execution boundary.

        return ExecutionResult(
            success=False,
            operation=decision.operation or parsed.intent,
            error=(
                "Learning execution is not implemented yet. "
                "The NLU route is working and the learning engine "
                "will be connected next."
            ),
        )

    # ---------------------------------------------------------
    # PROMPT BUILDING
    # ---------------------------------------------------------

    def _build_reasoning_prompt(
        self,
        parsed,
        context: ConversationContext,
    ) -> str:

        parts: list[str] = []

        if context.problem_text:
            parts.append(
                f"Previous problem:\n{context.problem_text}"
            )

        if parsed.original:
            parts.append(
                f"Current request:\n{parsed.original}"
            )

        if parsed.expression:
            parts.append(
                f"Expression:\n{parsed.expression}"
            )

        if parsed.conditions:
            parts.append(
                "Conditions:\n"
                + "\n".join(f"- {item}" for item in parsed.conditions)
            )

        if context.current_result is not None:
            parts.append(
                f"Previous computed result:\n{context.current_result}"
            )

        if context.steps:
            parts.append(
                "Available solution steps:\n"
                + "\n".join(
                    f"{i + 1}. {step}"
                    for i, step in enumerate(context.steps)
                )
            )

        parts.append(
            "Respond to the current request directly."
        )

        return "\n\n".join(parts)