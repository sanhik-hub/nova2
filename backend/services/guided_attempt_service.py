import json
import re
import random
from time import perf_counter
from typing import Any

from backend.services.provider_router import ProviderRouter


class GuidedAttemptService:
    """Generate a sequential guided attempt with one main LLM generation call."""

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

        # Guided attempts intentionally use one LLM generation call. The old
        # LLM complexity-planning call was only producing step-count metadata,
        # but it still incurred full local-model inference latency (~10 seconds).
        # Estimate the same metadata locally so planning is effectively free.
        planner_elapsed_ms = 0.0
        mcq_generation_elapsed_ms = 0.0
        guided_answer_tokens: int | None = None

        if step_count is None:
            step_count, guided_answer_tokens = self._estimate_guided_complexity(problem)

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
    def _estimate_guided_complexity(problem: str) -> tuple[int, int]:
        """Estimate guided-path complexity without invoking the local LLM."""
        text = problem.lower()
        words = re.findall(r"[a-z]+", text)

        # Start with a useful minimum for a guided STEM solution, then add
        # stages only for concrete signs of multi-step work. This is metadata
        # only; it never limits the main MCQ generation call.
        steps = 4

        complexity_markers = (
            "derive", "derivation", "prove", "proof", "show that",
            "calculate", "determine", "evaluate", "find",
            "integrate", "integration", "differentiate", "differentiation",
            "solve", "equation", "equations", "substitute", "substitution",
            "simplify", "simplification", "limit", "series",
            "stoichiometry", "mechanism", "equilibrium", "titration",
            "moles", "reaction", "kinetic", "thermodynamic",
        )
        for marker in complexity_markers:
            if marker in text:
                steps += 1

        # Multi-part or chained conditions usually require an additional
        # intermediate decision/result.
        if any(token in text for token in ("then", "after", "next", "given that", "where")):
            steps += 1
        if any(token in text for token in ("and", "also", "respectively")):
            steps += 1

        # Mathematical/scientific notation and multiple explicit quantities
        # are useful complexity signals, but keep the estimate conservative.
        if len(words) > 55:
            steps += 1
        if len(re.findall(r"\\b(?:x|y|z|t|m|M|L|r|R|v|u|a|q|p|V|T|P|I|omega|\\u03c9)\\b", problem)) >= 4:
            steps += 1

        steps = max(3, min(15, steps))

        # Advisory completion size scales with the number of generated MCQs.
        # It is telemetry/UI guidance only and is deliberately not passed as
        # max_tokens to the main generator.
        advisory_tokens = max(420, min(1400, 90 * steps))
        return steps, advisory_tokens

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
        """Move the question that most directly establishes the requested result to the end.

        The LLM generates one complete pool. This pass is deterministic and never
        calls the LLM again. It deliberately prefers the specific target from the
        original problem over generic prerequisite/formula questions.
        """
        if len(steps) < 2:
            return steps

        def normalized(value: str) -> str:
            return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

        def terms(value: str) -> set[str]:
            return {
                word
                for word in re.findall(r"[a-zA-Z]+", value.lower())
                if word not in {
                    "the", "a", "an", "of", "for", "to", "from", "in",
                    "on", "about", "using", "with", "given", "its",
                    "is", "are", "and", "by", "as",
                }
            }

        target_match = re.search(
            r"\b(?:determine|find|calculate|derive|evaluate|obtain)\s+"
            r"(?:the\s+)?(.+?)(?:[?.!]|$)",
            problem,
            flags=re.IGNORECASE,
        )
        target_text = target_match.group(1).strip() if target_match else ""
        target_norm = normalized(target_text)
        target_terms = terms(target_text)

        # Also keep the action verb because a final question often repeats it.
        action_match = re.search(
            r"\b(determine|find|calculate|derive|evaluate|obtain)\b",
            problem,
            flags=re.IGNORECASE,
        )
        action = action_match.group(1).lower() if action_match else ""

        def final_score(step: dict[str, Any]) -> int:
            question = str(step.get("question", ""))
            goal = str(step.get("step_goal", ""))
            question_norm = normalized(question)
            goal_norm = normalized(goal)
            combined = f"{question_norm} {goal_norm}"
            question_terms = terms(question)
            score = 0

            # Exact/near-exact target wording is the strongest signal. This
            # prevents a generic "formula for angular momentum" question from
            # beating "angular momentum of the rod about the axis of rotation".
            if target_norm:
                if target_norm in question_norm:
                    score += 40
                else:
                    target_sequence = target_norm.split()
                    if len(target_sequence) >= 3:
                        matched = sum(
                            1 for token in target_sequence
                            if token in question_norm.split()
                        )
                        score += min(24, matched * 5)

            # Requested target vocabulary is useful, but cap it so generic
            # formula questions cannot dominate the exact target match.
            overlap = len(target_terms & question_terms)
            score += min(12, overlap * 2)

            # Explicit final-result language is helpful but secondary.
            if any(
                phrase in question_norm
                for phrase in (
                    "final answer", "final result", "requested quantity",
                    "determine the", "calculate the", "find the",
                    "derive the", "evaluate the", "obtain the",
                )
            ):
                score += 5

            if any(
                phrase in goal_norm
                for phrase in (
                    "final", "requested result", "requested quantity",
                    "conclude", "obtain the result", "determine the result",
                    "calculate the result",
                )
            ):
                score += 4

            # Penalize generic prerequisite/formula questions when they are not
            # themselves the requested target.
            if any(
                phrase in question_norm
                for phrase in (
                    "formula for", "relationship between", "dimensional analysis",
                    "what is the formula", "formula of",
                )
            ):
                if target_norm and target_norm not in question_norm:
                    score -= 10

            # A question that asks for a generic law/formula but only shares a
            # broad target noun should not outrank a specific requested result.
            if target_terms and overlap <= 1 and target_norm not in question_norm:
                score -= 4

            # Avoid choosing an earlier step merely because its goal says "final".
            # The actual question must carry meaningful target evidence.
            if target_terms and overlap == 0 and target_norm not in question_norm:
                score -= 12

            return score

        scores = [final_score(step) for step in steps]
        candidate_index = max(range(len(steps)), key=lambda index: scores[index])

        # If the original problem does not expose a recognizable requested
        # target, preserve the model's natural order instead of guessing.
        if not target_text or scores[candidate_index] <= 0:
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

        # Final-result ordering is handled by _reorder_final_step() before
        # validation. A generated pool is therefore allowed to contain the
        # requested result naturally; the deterministic ordering pass moves the
        # best matching final-result question to the end without another LLM call.

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

