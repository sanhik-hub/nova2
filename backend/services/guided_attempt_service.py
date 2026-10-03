import json
import re
import random
from time import perf_counter
from typing import Any

from backend.services.provider_router import ProviderRouter


class GuidedAttemptService:
    """Plan a guided attempt with the small model and generate MCQs with the main LLM."""

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

        if step_count is None:
            planner_started = perf_counter()
            plan_response = self.router.plan(
                prompt=f"""
Problem:
{problem}

Choose the smallest sufficient number of progressive solution-path stages
needed to reach the final answer. Use 3 to 5 stages.

Also estimate an advisory completion-token target for the complete guided
attempt. Base it on the actual problem complexity and number of stages. This
is guidance only, not a generation limit.

Also provide one short ABSTRACT stage description for each stage. The stage
descriptions are structural guidance for another model; they must not solve
the problem.

Return ONLY this JSON object:
{{"step_count": N, "answer_tokens": T, "path": ["stage 1", "stage 2", "..."]}}
""".strip(),
                system_prompt="""
You are Nova's guided-attempt planning model.

Return exactly one JSON object with:
- "step_count": an integer from 3 through 5;
- "path": an array of exactly step_count short abstract stage descriptions;
- "answer_tokens": an advisory integer estimate for the complete guided attempt.

The answer_tokens estimate must be specific to this problem and planned
stages, not a fixed value or fixed multiplier.

Do NOT solve the problem.
Do NOT include the final answer.
Do NOT include problem-specific equations, symbolic results, numeric values,
question text, options, hints, explanations, or answer choices.

Each path item must describe only a mathematical dependency or action.
The final path item must be the only item whose purpose is to establish the
requested final result.
""".strip(),
                max_tokens=320,
            )
            planner_elapsed_ms = round((perf_counter() - planner_started) * 1000, 2)

            step_count = 3
            if plan_response.get("success"):
                try:
                    plan = self._parse_json(plan_response.get("result"))
                except (TypeError, ValueError, json.JSONDecodeError):
                    plan = None

                planned_count = plan.get("step_count") if isinstance(plan, dict) else None
                candidate_path = plan.get("path") if isinstance(plan, dict) else None
                advisory = plan.get("answer_tokens") if isinstance(plan, dict) else None

                if (
                    isinstance(planned_count, int)
                    and not isinstance(planned_count, bool)
                    and 3 <= planned_count <= 5
                    and isinstance(candidate_path, list)
                    and len(candidate_path) == planned_count
                    and all(isinstance(item, str) and item.strip() for item in candidate_path)
                ):
                    step_count = planned_count
                    planned_path = [item.strip() for item in candidate_path]
                    if (
                        isinstance(advisory, int)
                        and not isinstance(advisory, bool)
                        and 300 <= advisory <= 1200
                    ):
                        guided_answer_tokens = advisory

        if not isinstance(step_count, int) or not 3 <= step_count <= 5:
            return {
                "success": False,
                "error": "Guided attempt step count must be between 3 and 5.",
            }

        if guided_answer_tokens is None:
            guided_answer_tokens = 700

        generated_steps: list[dict[str, Any]] = []

        for step_number in range(1, step_count + 1):
            is_final = step_number == step_count
            stage_description = (
                planned_path[step_number - 1]
                if planned_path
                else (
                    "Establish the next necessary intermediate result."
                    if not is_final
                    else "Combine the required intermediate results and establish the requested result."
                )
            )

            prior_lines = []
            for index, prior in enumerate(generated_steps, start=1):
                correct = prior["options"][prior["correct_index"]]["text"]
                prior_lines.append(
                    f"Step {index} question: {prior['question']}\n"
                    f"Step {index} correct intermediate result: {correct}"
                )
            prior_context = "\n\n".join(prior_lines) or "No earlier step exists."

            if is_final:
                target_rule = """
This is the FINAL step. It must establish the answer requested by the
original problem. It may ask for the requested quantity, final expression,
final value, or final conclusion. This is the first step where the requested
final result may appear.
""".strip()
            else:
                target_rule = """
This is an INTERMEDIATE step. It must NOT ask for the requested final
quantity or final expression. It must establish only a genuinely necessary
intermediate result that the next step can use.

Do not put the requested final result, its exact expression, or an algebraically
equivalent expression in any option. Do not ask a formula-recall question for
the final quantity.
""".strip()

            system_prompt = f"""
You are Nova's guided-attempt question writer.

Generate exactly ONE multiple-choice step for a STEM problem.

The guided attempt is generated sequentially. Earlier steps have already been
created, and their correct intermediate results are supplied as context.
Build this step directly on that dependency chain.

This is step {step_number} of {step_count}.

{target_rule}

The planner's abstract stage for this step is:
{stage_description}

The original problem is authoritative. Do not invent information.

The question must ask for one concrete mathematical decision or result.
The correct option must be mathematically correct and useful to the next step.
All four options must be distinct, with exactly one correct answer.

For continuous-body mechanics, derivations, calculus, chemistry calculations,
or other multi-stage STEM problems, prefer deriving the needed local or
intermediate quantity before accumulating, substituting, simplifying, or
concluding. Do not turn a supplied quantity into a question asking the student
to identify it.

Return ONLY:
{{"question":"...","options":[{{"text":"..."}},{{"text":"..."}},{{"text":"..."}},{{"text":"..."}},],"correct_index":0,"step_goal":"..."}}.

Every response must contain exactly the keys question, options, correct_index,
and step_goal${include_hints ? ', plus hint' : ''}.
The options must contain only text.

Do not include step labels, markdown headings, explanations, or extra JSON keys.
The UI supplies the step number.
${include_hints ? 'The hint must be short and must not reveal the correct option.' : ''}
""".strip()

            prompt = f"""
Original problem:
{problem}

Previously generated solution path:
{prior_context}

Current step:
{step_number} of {step_count}

Generate only this one step.
""".strip()

            normalized_step = None
            last_error = None

            # Retry only a malformed single step; never regenerate the whole path.
            for _attempt in range(2):
                mcq_step_started = perf_counter()
                response = self.router.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=None,
                    temperature=0,
                    enable_thinking=False,
                )
                mcq_generation_elapsed_ms += (
                    perf_counter() - mcq_step_started
                ) * 1000

                if not response.get("success"):
                    last_error = (
                        response.get("error")
                        or f"Guided attempt generation failed at step {step_number}."
                    )
                    continue

                try:
                    payload = self._parse_json(response.get("result"))
                except (TypeError, ValueError, json.JSONDecodeError):
                    last_error = (
                        f"Guided attempt generator returned invalid JSON at step {step_number}."
                    )
                    continue

                normalized_step = self._validate_single_step(
                    payload,
                    include_hints=include_hints,
                    is_final=is_final,
                )
                if normalized_step is not None:
                    break

                last_error = (
                    f"Guided attempt generator returned an invalid step at step {step_number}."
                )

            if normalized_step is None:
                return {
                    "success": False,
                    "error": last_error or (
                        f"Guided attempt generation failed at step {step_number}."
                    ),
                    "planner_elapsed_ms": planner_elapsed_ms,
                    "mcq_generation_elapsed_ms": round(mcq_generation_elapsed_ms, 2),
                    "answer_tokens_advisory": guided_answer_tokens,
                }

            generated_steps.append(normalized_step)

        # The final answer is known now. If it leaked into an earlier option,
        # repair only that intermediate step instead of discarding the entire path.
        if len(generated_steps) >= 2:
            final_correct = generated_steps[-1]["options"][
                generated_steps[-1]["correct_index"]
            ]["text"]

            def _option_key(value: str) -> str:
                return re.sub(r"[^a-z0-9]+", "", value.lower())

            final_key = _option_key(final_correct)
            leaked_index = None
            for index, earlier_step in enumerate(generated_steps[:-1]):
                if any(
                    _option_key(option["text"]) == final_key
                    for option in earlier_step["options"]
                ):
                    leaked_index = index
                    break

            if leaked_index is not None:
                leaked_step = generated_steps[leaked_index]
                repair_prompt = f"""
Original problem:
{problem}

Intermediate step {leaked_index + 1} of {step_count}:
{json.dumps(leaked_step, ensure_ascii=False)}

Final result:
{final_correct}

Rewrite ONLY this intermediate step. No option may contain the final result
or an algebraically equivalent expression. Preserve the existing correct
intermediate result exactly when it is not the final result, because later
steps depend on it. Replace leaking distractors with genuinely incorrect but
plausible alternatives.

Return ONLY:
{{"question":"...","options":[{{"text":"..."}},{{"text":"..."}},{{"text":"..."}},{{"text":"..."}},],"correct_index":0,"step_goal":"..."}}.
""".strip()

                repair_system = f"""
You are repairing one intermediate MCQ in a sequential STEM solution.
The final result is known and must not appear in this intermediate step.
Return exactly one valid step with question, four text-only options,
correct_index, and step_goal{', plus hint' if include_hints else ''}.
Keep the mathematics correct and preserve the existing correct intermediate
result when it is not the final result.
{'Include one short hint without revealing the answer.' if include_hints else ''}
""".strip()

                repair_started = perf_counter()
                repair_response = self.router.generate(
                    prompt=repair_prompt,
                    system_prompt=repair_system,
                    max_tokens=None,
                    temperature=0,
                    enable_thinking=False,
                )
                mcq_generation_elapsed_ms += (
                    perf_counter() - repair_started
                ) * 1000

                if repair_response.get("success"):
                    try:
                        repaired_payload = self._parse_json(
                            repair_response.get("result")
                        )
                    except (TypeError, ValueError, json.JSONDecodeError):
                        repaired_payload = None

                    repaired_step = (
                        self._validate_single_step(
                            repaired_payload,
                            include_hints=include_hints,
                            is_final=False,
                        )
                        if repaired_payload is not None
                        else None
                    )
                    if repaired_step is not None:
                        generated_steps[leaked_index] = repaired_step

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
        if set(payload) != allowed:
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
        }

        if include_hints:
            hint = payload.get("hint")
            if not isinstance(hint, str) or not hint.strip():
                return None
            result["hint"] = hint.strip()

        return result

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
            or not 3 <= len(steps) <= 5
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

            final_correct = normalized[-1]["options"][
                normalized[-1]["correct_index"]
            ]["text"]
            final_key = option_key(final_correct)

            for earlier_step in normalized[:-1]:
                for option in earlier_step["options"]:
                    if option_key(option["text"]) == final_key:
                        return False, (
                            "Guided attempt revealed the final result before the final step."
                        )

            # Do not allow a later question to simply ask for the same
            # mathematical target as an earlier question. This catches common
            # "formula -> substitution -> formula" failures without encoding a
            # subject-specific solution.
            question_keys = [option_key(step["question"]) for step in normalized]
            if len(question_keys) != len(set(question_keys)):
                return False, "Guided attempt contains repeated questions."

        return True, normalized

