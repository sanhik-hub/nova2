import json
import re
import random
from time import perf_counter
from typing import Any

from backend.services.provider_router import ProviderRouter


class GuidedAttemptService:
    """Use advisory planning plus the main LLM to generate a sequential guided attempt."""

    def __init__(self, router: ProviderRouter | None = None):
        self.router = router or ProviderRouter()

    def generate(
        self,
        query: str,
        include_hints: bool = False,
        step_count: int | None = None,
    ) -> dict[str, Any]:
        problem = query.strip()
        if not problem:
            return {"success": False, "error": "Question is empty."}

        planned_path: list[str] | None = None
        planner_elapsed_ms = 0.0
        mcq_generation_elapsed_ms = 0.0
        guided_answer_tokens: int | None = None
        # The lightweight planner remains a benchmark only. Production guided
        # attempts use the primary reasoning model and a single pooled MCQ call.
        if step_count is None:
            main_plan_started = perf_counter()
            main_plan_response = self.router.generate(
                prompt=f"""
Original problem:
{problem}

No lightweight planning estimate is available. Use your own reasoning and
choose the actual number of progressive guided steps needed for this problem.

Create the internal solution progression for the stronger guided-attempt
generator. The progression may contain more than five steps for complex
problems. Each stage must represent a genuinely necessary intermediate
reasoning goal, and the final stage must establish the requested result.

Also estimate an advisory completion-token target for the complete guided
attempt. This is metadata only, not a generation limit.

Return ONLY:
{{"step_count":N,"answer_tokens_advisory":T,"stage_goals":["goal 1","goal 2","..."]}}
""".strip(),
                system_prompt="""
You are Nova's primary guided-attempt planning model.

Unlike the lightweight complexity estimator, you are responsible for deciding
the actual progressive structure used to generate the guided MCQs.

Analyze the original problem and choose the number of necessary stages based
on the mathematics, physics, chemistry, or other STEM reasoning required.
Simple problems may need only a few stages; complex problems may need more
than five.

Return exactly one JSON object with:
- step_count: integer from 3 through 15;
- answer_tokens_advisory: an approximate completion-token target for the whole
  guided attempt, based on the actual problem and planned stages;
- stage_goals: exactly step_count concise goals.

The token estimate is advisory only. It must never be treated as a generation
limit.

Do not write the actual MCQ questions, options, hints, or final solution.
Stage goals may describe the actual mathematical dependency because this
planning model is the stronger reasoning model used by the guided generator.
""".strip(),
                max_tokens=None,
                temperature=0,
                enable_thinking=False,
            )
            planner_elapsed_ms += (
                perf_counter() - main_plan_started
            ) * 1000

            main_plan = None
            if main_plan_response.get("success"):
                try:
                    main_plan = self._parse_json(main_plan_response.get("result"))
                except (TypeError, ValueError, json.JSONDecodeError):
                    main_plan = None

            planned_count = (
                main_plan.get("step_count")
                if isinstance(main_plan, dict)
                else None
            )
            stage_goals = (
                main_plan.get("stage_goals")
                if isinstance(main_plan, dict)
                else None
            )
            advisory_tokens = (
                main_plan.get("answer_tokens_advisory")
                if isinstance(main_plan, dict)
                else None
            )

            if (
                isinstance(planned_count, int)
                and not isinstance(planned_count, bool)
                and 3 <= planned_count <= 15
                and isinstance(stage_goals, list)
                and len(stage_goals) == planned_count
                and all(isinstance(item, str) and item.strip() for item in stage_goals)
            ):
                step_count = planned_count
                planned_path = [item.strip() for item in stage_goals]
                if (
                    isinstance(advisory_tokens, int)
                    and not isinstance(advisory_tokens, bool)
                    and advisory_tokens > 0
                ):
                    guided_answer_tokens = advisory_tokens

            # Emergency fallback only if the stronger planning call fails.
            # The lightweight planner never determines the actual step count.
            if step_count is None:
                step_count = 6

        if not isinstance(step_count, int) or not 3 <= step_count <= 15:
            return {
                "success": False,
                "error": "Guided attempt step count must be between 3 and 15.",
            }

        # Generate the complete MCQ pool in one call. The model is allowed to
        # produce the natural solution progression without being forced to decide
        # which item must be presented first. The backend reorders the pool after
        # generation so the requested final result is presented last.
        generation_prompt = f"""
Original problem:
{problem}

Generate exactly {step_count} multiple-choice questions that together form a
complete guided solution path for the original problem.

Generate the questions as a COMPLETE POOL first. Do not worry about the order
in which they will be presented. If a question that asks for the requested
final quantity/result naturally appears before prerequisite questions, that is
allowed: the backend will reorder the generated pool before returning it.

The pool must:
- cover the necessary mathematical/scientific reasoning from the original
  problem;
- contain genuinely useful intermediate decisions/results, not filler;
- contain one question that establishes the requested final result;
- avoid simply asking for quantities already supplied in the problem;
- avoid duplicate questions;
- use exactly four distinct options per question with exactly one correct;
- keep each question self-contained enough for a student to understand it;
- work across mathematics, physics, chemistry, and other STEM problems.

For derivations, continuous systems, calculus, mechanics, chemistry
calculations, and similar multi-stage problems, prefer meaningful intermediate
results such as a differential contribution, governing relation, substitution,
integration, simplification, or physical/chemical consequence rather than
memorized final formulas.

The requested final result may occur anywhere in the generated pool. Do NOT
discard a useful question merely because it currently appears to reveal the
final answer; the backend will identify and move the final-result step to the
end.

Return ONLY one JSON object:
{{"steps":[
  {{"question":"...","options":[{{"text":"..."}},{{"text":"..."}},{{"text":"..."}},{{"text":"..."}}],
    "correct_index":0,"step_goal":"..."}}
]}}

Every step must contain exactly question, options, correct_index, and step_goal
{', plus hint' if include_hints else ''}. Do not include any other JSON keys.
The options must contain only text.
{'Each step must also contain a short hint that does not reveal the correct option.' if include_hints else ''}
""".strip()

        generation_system = f"""
You are Nova's guided-attempt MCQ generator.

Generate the COMPLETE set of {step_count} MCQs for the original STEM problem
in one response. This is a generation task, not a sequential dialogue.

Do not artificially force the final-result question to be last. Generate the
best complete solution pool you can. A separate backend ordering pass will
place the final-result question at the end.

The original problem is authoritative. Do not invent data or assumptions.
Each question must have exactly four distinct text options and one correct
answer. The correct answer must be mathematically/scientifically correct.

Use concise but meaningful questions. Complex problems may require more than
five stages; do not collapse a multi-stage derivation into a single formula
recall question.

Return only the requested JSON object. No markdown, commentary, or extra keys.
""".strip()

        generation_started = perf_counter()
        generation_response = self.router.generate(
            prompt=generation_prompt,
            system_prompt=generation_system,
            max_tokens=None,
            temperature=0,
            enable_thinking=False,
        )
        mcq_generation_elapsed_ms += (
            perf_counter() - generation_started
        ) * 1000

        if not generation_response.get("success"):
            return {
                "success": False,
                "error": (
                    generation_response.get("error")
                    or "Guided attempt generation failed."
                ),
                "planner_elapsed_ms": round(planner_elapsed_ms, 2),
                "mcq_generation_elapsed_ms": round(mcq_generation_elapsed_ms, 2),
                "answer_tokens_advisory": guided_answer_tokens,
            }

        try:
            generated_payload = self._parse_json(
                generation_response.get("result")
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return {
                "success": False,
                "error": "Guided attempt generator returned invalid JSON.",
                "planner_elapsed_ms": round(planner_elapsed_ms, 2),
                "mcq_generation_elapsed_ms": round(mcq_generation_elapsed_ms, 2),
                "answer_tokens_advisory": guided_answer_tokens,
            }

        raw_steps = (
            generated_payload.get("steps")
            if isinstance(generated_payload, dict)
            else None
        )
        if not isinstance(raw_steps, list):
            return {
                "success": False,
                "error": "Guided attempt generator returned an invalid step pool.",
                "planner_elapsed_ms": round(planner_elapsed_ms, 2),
                "mcq_generation_elapsed_ms": round(mcq_generation_elapsed_ms, 2),
                "answer_tokens_advisory": guided_answer_tokens,
            }

        generated_steps = []
        for payload in raw_steps:
            normalized_step = self._validate_single_step(
                payload,
                include_hints=include_hints,
                is_final=False,
            )
            if normalized_step is None:
                return {
                    "success": False,
                    "error": "Guided attempt generator returned an invalid step contract.",
                    "planner_elapsed_ms": round(planner_elapsed_ms, 2),
                    "mcq_generation_elapsed_ms": round(mcq_generation_elapsed_ms, 2),
                    "answer_tokens_advisory": guided_answer_tokens,
                }
            generated_steps.append(normalized_step)

        generated_steps = self._reorder_final_step(problem, generated_steps)

        valid, steps_or_error = self._validate(
            {"steps": generated_steps},
            include_hints,
            step_count,
            problem,
        )

        if not valid:
            return {
                "success": False,
                "error": steps_or_error,
                "planner_elapsed_ms": planner_elapsed_ms,
                "mcq_generation_elapsed_ms": round(mcq_generation_elapsed_ms, 2),
                "answer_tokens_advisory": guided_answer_tokens,
            }

        return {
            "success": True,
            "query": problem,
            "steps": steps_or_error,
            "include_hints": include_hints,
            "include_full_solution": False,
            "planner_elapsed_ms": planner_elapsed_ms,
            "mcq_generation_elapsed_ms": round(mcq_generation_elapsed_ms, 2),
            "answer_tokens_advisory": guided_answer_tokens,
            }

    @staticmethod
    def _validate_single_step(
        payload: Any,
        include_hints: bool,
        is_final: bool,
    ) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None

        required = {"question", "options", "correct_index", "step_goal"}
        allowed = required | ({"hint"} if include_hints else set())
        if not required.issubset(payload):
            return None

        question = payload.get("question")
        if not isinstance(question, str):
            return None
        question = re.sub(
            r"^\s*(?:#{1,6}\s*)?(?:step\s*)?\d+\s*[:.)-]?\s*",
            "",
            question,
            flags=re.IGNORECASE,
        ).strip()
        if not question:
            return None

        step_goal = payload.get("step_goal")
        if not isinstance(step_goal, str) or not step_goal.strip():
            return None

        options = payload.get("options")
        if not isinstance(options, list) or len(options) != 4:
            return None

        normalized_options = []
        for option in options:
            if isinstance(option, str):
                text = option
            elif isinstance(option, dict) and "text" in option:
                text = option["text"]
            else:
                return None

            if not isinstance(text, str) or not text.strip():
                return None
            normalized_options.append({"text": text.strip()})

        correct_index = payload.get("correct_index")
        if (
            not isinstance(correct_index, int)
            or isinstance(correct_index, bool)
            or not 0 <= correct_index < 4
        ):
            return None

        result = {
            "question": question,
            "options": normalized_options,
            "correct_index": correct_index,
            "step_goal": step_goal.strip(),
        }

        if include_hints:
            hint = payload.get("hint")
            if not isinstance(hint, str) or not hint.strip():
                return None
            result["hint"] = hint.strip()

        return result

    @staticmethod
    def _reorder_final_step(
        problem: str,
        steps: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Move the step that asks for the requested final result to the end.

        The LLM generates the complete solution pool first. This pass is
        deliberately deterministic and does not call the LLM again.
        """
        if len(steps) < 2:
            return steps

        def key(value: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", value.lower())

        target_match = re.search(
            r"\b(?:determine|find|calculate|derive|evaluate|obtain)\s+"
            r"(?:the\s+)?(.+?)(?:[?.!]|$)",
            problem,
            flags=re.IGNORECASE,
        )

        target_text = target_match.group(1) if target_match else ""
        target_terms = {
            word
            for word in re.findall(r"[a-zA-Z]+", target_text.lower())
            if word not in {
                "the", "a", "an", "of", "for", "to", "from", "in",
                "on", "about", "using", "with", "given", "its",
            }
        }

        def final_score(step: dict[str, Any]) -> int:
            question = step.get("question", "").lower()
            goal = step.get("step_goal", "").lower()
            combined = f"{question} {goal}"
            score = 0

            # Explicit final-result language is the strongest signal.
            if any(
                phrase in question
                for phrase in (
                    "final answer", "final result", "requested quantity",
                    "determine the", "calculate the", "find the",
                )
            ):
                score += 6

            # Match terms from the original requested target.
            if target_terms:
                words = set(re.findall(r"[a-zA-Z]+", combined))
                score += min(6, 2 * len(target_terms & words))

            # A final step normally has a concluding goal.
            if any(
                phrase in goal
                for phrase in (
                    "final", "requested result", "requested quantity",
                    "conclude", "obtain the result", "determine the result",
                )
            ):
                score += 4

            return score

        scores = [final_score(step) for step in steps]
        candidate_index = max(range(len(steps)), key=lambda index: scores[index])

        # If no step has a meaningful final-result signal, preserve the model's
        # natural order rather than making a speculative rearrangement.
        if scores[candidate_index] <= 0:
            return steps

        candidate = steps[candidate_index]
        return [
            step
            for index, step in enumerate(steps)
            if index != candidate_index
        ] + [candidate]

    def generate_solution(
        self,
        query: str,
        steps: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate the optional complete solution using the guided MCQ path as context."""
        problem = query.strip()
        if not problem:
            return {"success": False, "error": "Question is empty."}

        if not isinstance(steps, list) or not steps:
            return {"success": False, "error": "Guided attempt has no solution path."}

        path_lines = []
        for index, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                continue
            question = str(step.get("question", "")).strip()
            options = step.get("options", [])
            correct_index = step.get("correct_index")
            if not question or not isinstance(options, list):
                continue
            if not isinstance(correct_index, int) or not 0 <= correct_index < len(options):
                continue
            correct_text = options[correct_index]
            if isinstance(correct_text, dict):
                correct_text = correct_text.get("text", "")
            path_lines.append(
                f"Step {index}: {question}\\nCorrect choice: {str(correct_text).strip()}"
            )

        if not path_lines:
            return {"success": False, "error": "Guided attempt solution path is invalid."}

        system_prompt = r"""
You are Nova's concise mathematical solution writer.

Write ONE concise but complete, self-contained solution to the original problem.
The supplied Guided Attempt is a structured solution path. Use its correct
choices as intermediate guidance, but independently check every mathematical
statement before writing it. Never blindly trust an incorrect MCQ choice.

Accuracy has priority over brevity:
- Re-derive or verify each essential equation and transformation.
- Check algebra, differentiation, integration, substitutions, signs, units, and
  final values whenever applicable.
- If the supplied path conflicts with the mathematics or the original problem,
  correct the mathematics rather than reproducing the path.
- Do not invent missing information or assumptions.

Concise means remove filler, not necessary mathematical work. For a calculation,
show the essential setup and transformations. For a derivation or proof, show
the complete mathematical chain needed to justify the result, with brief
explanations around the equations. Do not skip a necessary step merely to make
the answer shorter.

Do not mention the Guided Attempt, MCQs, choices, AI, tools, token limits, or
internal processing.
Use standard Markdown prose and standard LaTeX for mathematics. When using
inline math, use $...$; when using a displayed equation, use $...$.
Never put Markdown backticks inside $...$ or $...$. Do not write malformed
hybrid expressions such as L2, x2, or "wer" when the intended mathematical
notation is L^2, x^2, etc. Keep mathematical symbols inside LaTeX delimiters.
Do not preview or repeat the final result. State the requested final result
exactly once, as the final mathematical statement.
Return ONLY the solution text.
""".strip()

        prompt = f"""Original problem:
{problem}

Verified solution path:
{chr(10).join(path_lines)}

Using those verified intermediate choices, write the complete final solution.
""".strip()

        response = self.router.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=None,
            temperature=0,
            enable_thinking=False,
        )

        if not response.get("success"):
            return {
                "success": False,
                "error": response.get("error") or "Unable to generate the full solution.",
            }

        answer = str(response.get("result") or "").strip()
        if not answer:
            return {"success": False, "error": "The solution model returned no solution."}

        return {
            "success": True,
            "answer": answer,
            "engine": "local_llm",
            "tool_calls": [],
            "cached": False,
        }

    @staticmethod
    def _parse_json(raw: Any) -> Any:
        if not isinstance(raw, str):
            raise ValueError("Expected JSON text.")

        text = raw.strip()

        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text, count=1)

        return json.loads(text)

    def _validate(
        self,
        payload: Any,
        include_hints: bool,
        requested_step_count: int | None = None,
        problem: str = "",
    ):
        if not isinstance(payload, dict) or set(payload) != {"steps"}:
            return False, "Guided attempt generator returned an invalid structure."

        steps = payload.get("steps")
        if (
            not isinstance(steps, list)
            or not 3 <= len(steps) <= 15
            or (
                requested_step_count is not None
                and len(steps) != requested_step_count
            )
        ):
            return False, "Guided attempt generator returned the wrong number of steps."

        normalized = []

        for step in steps:
            if not isinstance(step, dict):
                return False, "Guided attempt generator returned an invalid step."

            required = {"question", "options", "correct_index", "step_goal"}
            allowed = required | ({"hint"} if include_hints else set())
            if set(step) != allowed:
                return False, "Guided attempt generator returned an invalid step contract."

            question = step["question"]
            if not isinstance(question, str) or not question.strip():
                return False, "Guided attempt contains an empty question."

            # The UI owns step numbering. Remove model-generated labels so a
            # valid question cannot render every item as "STEP 1".
            question = re.sub(
                r"^\s*(?:#{1,6}\s*)?(?:step\s*)?\d+\s*[:.)-]?\s*",
                "",
                question,
                flags=re.IGNORECASE,
            ).strip()
            if not question:
                return False, "Guided attempt contains an empty question."

            step_goal = step["step_goal"]
            if not isinstance(step_goal, str) or not step_goal.strip():
                return False, "Guided attempt contains an empty step goal."

            options = step["options"]
            if not isinstance(options, list) or len(options) != 4:
                return False, "Each guided step must have exactly four options."

            normalized_options = []
            for option in options:
                # The main LLM may occasionally attach harmless metadata such
                # as an option-local index. The UI only needs the option text;
                # normalize such extras instead of rejecting an otherwise valid
                # four-option MCQ.
                # The main LLM may emit either the requested object form
                # {"text": "..."} or the simpler JSON string form "...".
                # Both represent the same UI option, so normalize both.
                if isinstance(option, str):
                    option_text = option
                elif isinstance(option, dict) and "text" in option:
                    option_text = option["text"]
                else:
                    return False, "Guided attempt contains an invalid option."

                if not isinstance(option_text, str) or not option_text.strip():
                    return False, "Guided attempt contains an empty option."

                normalized_options.append({"text": option_text.strip()})

            index = step["correct_index"]
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < 4
            ):
                return False, "Guided attempt contains an invalid correct answer."

            normalized_step = {
                "question": question,
                "options": normalized_options,
                "correct_index": index,
            }

            if include_hints:
                hint = step["hint"]
                if not isinstance(hint, str) or not hint.strip():
                    return False, "Guided attempt contains a missing hint."
                normalized_step["hint"] = hint.strip()

            normalized.append(normalized_step)

        # The model may return a syntactically valid path that is still
        # pedagogically broken. Reject the most important forms of leakage or
        # disconnected steps rather than silently presenting them as a guided path.

        # Reject common formula-recall / restatement failures before the
        # result reaches the UI. Extract the requested target from the original
        # problem and make sure earlier questions do not simply ask for that
        # target's standard formula.
        target_match = re.search(
            r"\b(?:determine|find|calculate|derive|evaluate|obtain)\s+(?:the\s+)?(.+?)(?:[?.!]|$)",
            problem,
            flags=re.IGNORECASE,
        )
        if target_match and len(normalized) >= 2:
            target_words = [
                word
                for word in re.findall(r"[a-zA-Z]+", target_match.group(1).lower())
                if word not in {
                    "the", "a", "an", "of", "for", "to", "from", "in",
                    "on", "about", "using", "with", "given", "its",
                }
            ]
            target_terms = set(target_words[:4])
            if len(target_terms) >= 2:
                for earlier_step in normalized[:-1]:
                    earlier_text = earlier_step["question"].lower()
                    earlier_words = set(re.findall(r"[a-zA-Z]+", earlier_text))
                    overlap = target_terms & earlier_words
                    asks_formula = any(
                        phrase in earlier_text
                        for phrase in (
                            "formula", "equation", "expression for",
                            "relation for", "law for",
                        )
                    )
                    if asks_formula and len(overlap) >= 2:
                        return False, (
                            "Guided attempt asks for the requested final quantity "
                            "or its standard formula before the final step."
                        )

        # Reject steps that merely ask the student to identify a quantity
        # explicitly supplied by the original problem.
        given_quantities = set()
        for match in re.finditer(
            r"\b(?:length|mass|radius|time|speed|velocity|acceleration|"
            r"angular velocity|frequency|charge|voltage|current|pressure|"
            r"temperature|density)\s+(?:is|=)\s+([A-Za-z0-9_\\.]+)",
            problem,
            flags=re.IGNORECASE,
        ):
            given_quantities.add(match.group(0).lower().split()[0])
        if given_quantities:
            for earlier_step in normalized[:-1]:
                q = earlier_step["question"].lower()
                if "what is" in q and any(
                    quantity in q for quantity in given_quantities
                ):
                    return False, (
                        "Guided attempt asks the student to identify a quantity "
                        "that is already provided by the problem."
                    )

        # The UI should not inherit the model's positional bias. Shuffle the
        # four options while keeping exactly one correct answer and deliberately
        # vary the correct position across the path.
        rng = random.SystemRandom()
        positions = list(range(4))
        rng.shuffle(positions)
        for index in range(4, len(normalized)):
            choices = [
                position for position in range(4)
                if position != positions[index - 1]
            ]
            positions.append(rng.choice(choices))
        for step, target_position in zip(normalized, positions):
            current_index = step["correct_index"]
            if current_index == target_position:
                continue
            options = step["options"]
            correct_option = options.pop(current_index)
            options.insert(target_position, correct_option)
            step["correct_index"] = target_position

        if len(normalized) >= 2:
            def option_key(value: str) -> str:
                return re.sub(r"[^a-z0-9]+", "", value.lower())

            # Final-result ordering is handled before validation by
            # _reorder_final_step(). Do not reject the pool merely because the
            # same final expression appeared in an earlier generated option.
            # The generator is intentionally allowed to produce its natural pool.
            # Do not allow a later question to simply ask for the same
            # mathematical target as an earlier question. This catches common
            # "formula -> substitution -> formula" failures without encoding a
            # subject-specific solution.
            question_keys = [option_key(step["question"]) for step in normalized]
            if len(question_keys) != len(set(question_keys)):
                return False, "Guided attempt contains repeated questions."

        return True, normalized

