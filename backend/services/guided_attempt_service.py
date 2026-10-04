import ast
import json
import re
import random
from time import perf_counter
from typing import Any

from backend.services.provider_router import ProviderRouter


class GuidedAttemptService:
    """Generate a guided attempt with one main LLM generation call.

    The lightweight planner supplies only coarse complexity metadata. It is
    deliberately not trusted with semantic stage planning because its small
    model can produce plausible-looking but incorrect STEM decompositions.
    """

    def __init__(self, router: ProviderRouter | None = None):
        self.router = router or ProviderRouter()

    def generate(
        self,
        query: str,
        include_hints: bool = False,
        step_count: int | None = None,
    ) -> dict[str, Any]:
        """Generate a concise verified solution first, then convert it into MCQs.

        The solution path is the source of truth for the guided attempt. The
        second model call only turns those already-connected solution steps into
        questions/options; it is not asked to discover the solution path again.
        The first response is also returned as the full solution, so no third
        generation is required at the end of the attempt.
        """
        problem = query.strip()
        if not problem:
            return {"success": False, "error": "Question is empty."}

        planner_elapsed_ms = 0.0
        solution_generation_elapsed_ms = 0.0
        mcq_generation_elapsed_ms = 0.0
        guided_answer_tokens: int | None = None

        if step_count is None:
            fallback_step_count, fallback_tokens = self._estimate_guided_complexity(problem)
            planner_started = perf_counter()
            planner_response = self.router.plan(
                prompt=(
                    "Estimate a useful concise STEM solution length. Return ONLY "
                    "STEP_COUNT|ANSWER_TOKENS. STEP_COUNT must be 3-8."
                    f"\n\nProblem: {problem}"
                ),
                system_prompt=(
                    "You are a fast planning model. Do not solve the problem. "
                    "Return only N|T, where N is the number of useful solution "
                    "steps and T is a rough completion-size estimate."
                ),
                max_tokens=48,
                compact=True,
                request_timeout=3,
            )
            planner_elapsed_ms = (perf_counter() - planner_started) * 1000

            planned = self._parse_compact_plan(planner_response.get("result"))
            if planned is not None:
                planned_steps, planned_tokens = planned
                if abs(planned_steps - fallback_step_count) <= 1:
                    step_count, guided_answer_tokens = planned_steps, planned_tokens
                else:
                    step_count, guided_answer_tokens = fallback_step_count, fallback_tokens
            else:
                step_count, guided_answer_tokens = fallback_step_count, fallback_tokens

        if not isinstance(step_count, int) or not 3 <= step_count <= 8:
            return {
                "success": False,
                "error": "Guided attempt step count must be between 3 and 8.",
            }

        # PASS 1: solve the STEM problem as a short numbered path. This is
        # deliberately plain text rather than JSON/schema constrained output.
        # The model is therefore solving a much smaller task than the old
        # one-shot MCQ generator.
        solution_prompt = f"""Problem:
{problem}

Write a concise, correct solution in exactly {step_count} numbered steps.

Rules:
- Solve the actual problem; do not discuss how to make MCQs.
- Every step must perform one necessary mathematical/scientific action.
- Use all quantities given by the problem directly; never ask for a supplied quantity.
- Keep the path connected: each step must use the result/principle established before it.
- Show the essential equation, substitution, transformation, or chemical relation.
- Do not add generic topic definitions or filler.
- The final step must establish the requested result.
- Prefer the shortest valid derivation. Do not split trivial algebra into separate steps.
- Use standard scientific notation and UTF-8 characters.
- Return ONLY the numbered solution steps, one step per line.
""".strip()

        solution_system = """
You are Nova's STEM solution-path solver.

Solve the problem accurately for mathematics, physics, chemistry, or general
science. Produce a short connected derivation that a student can follow.
Do the reasoning internally, but expose the necessary equations and scientific
transformations in the numbered steps.

For chemistry, identify the exact transformation requested and check reactants,
reagents/conditions, regiochemistry, stoichiometry, and product identity before
writing the final step. Do not substitute a related reaction or product.
For physics and mathematics, check the governing relation, substitutions,
algebra, signs, units, and the requested quantity.
Never invent missing data.

Output only the numbered solution. No heading, commentary, alternatives, or
MCQ instructions.
""".strip()

        solution_started = perf_counter()
        solution_response = self.router.generate(
            prompt=solution_prompt,
            system_prompt=solution_system,
            max_tokens=min(
                420,
                max(240, 48 * step_count + 60),
            ),
            temperature=0,
            enable_thinking=False,
        )
        solution_generation_elapsed_ms = (perf_counter() - solution_started) * 1000

        if not solution_response.get("success"):
            return {
                "success": False,
                "error": solution_response.get("error") or "Guided solution generation failed.",
                "planner_elapsed_ms": round(planner_elapsed_ms, 2),
                "solution_generation_elapsed_ms": round(solution_generation_elapsed_ms, 2),
                "mcq_generation_elapsed_ms": 0.0,
                "answer_tokens_advisory": guided_answer_tokens,
            }

        solution_text = self._normalize_generated_text(
            str(solution_response.get("result") or "").strip()
        )
        solution_steps = self._parse_numbered_solution(solution_text)
        if len(solution_steps) < 3:
            return {
                "success": False,
                "error": "Guided solution generator returned fewer than three connected steps.",
                "planner_elapsed_ms": round(planner_elapsed_ms, 2),
                "solution_generation_elapsed_ms": round(solution_generation_elapsed_ms, 2),
                "mcq_generation_elapsed_ms": 0.0,
                "answer_tokens_advisory": guided_answer_tokens,
            }

        # Use the parsed path rather than trusting an arbitrary requested count.
        # The displayed full solution is exactly this first-pass solution.
        solution_text = "\n".join(
            f"{index}. {step}"
            for index, step in enumerate(solution_steps, start=1)
        )

        # PASS 2: convert the fixed solution path into MCQs. The model no longer
        # has to discover the STEM path, which is the central reliability change.
        mcq_prompt = f"""Original problem:
{problem}

Authoritative solution path:
{solution_text}

Convert EACH numbered solution step into exactly one guided MCQ, in the same
order.

Rules:
- Do not invent, reorder, merge, or split solution steps.
- The question must ask the student to perform or identify the specific action/result in that solution step.
- The correct option must be the result established by that solution step.
- Never ask for a quantity that the original problem already gives.
- Do not repeat the same target in two different questions.
- Do not ask generic definitions or unrelated formula-recall questions.
- Each question has exactly four distinct short options and one correct option.
- Earlier questions must not reveal the final requested result unless that result is genuinely established by that earlier step.
- If hints are requested, give a short hint that points toward the current step without giving its answer.
- Return only the required JSON object.
""".strip()

        mcq_system = """
You convert an authoritative STEM solution into concise guided MCQs.

The supplied numbered solution is the source of truth. Do NOT solve a different
problem and do NOT create a new path. Preserve the exact step order and make
each question test the corresponding step.

For each step, create one useful question, four distinct options, and exactly
one correct_index. The correct option must match the result of that solution
step. Use the original problem's supplied data directly rather than asking the
student to identify it.

Keep questions concise. Avoid duplicate targets, generic definitions, and
trivial repetition. Output only the requested JSON.
""".strip()

        generation_started = perf_counter()
        generation_response = self.router.generate(
            prompt=mcq_prompt,
            system_prompt=mcq_system,
            max_tokens=min(
                500,
                max(300, 70 * len(solution_steps) + (30 if include_hints else 0)),
            ),
            temperature=0,
            enable_thinking=False,
            response_format=self._guided_response_format(
                step_count=len(solution_steps),
                include_hints=include_hints,
            ),
        )
        mcq_generation_elapsed_ms = (perf_counter() - generation_started) * 1000

        if not generation_response.get("success"):
            return {
                "success": False,
                "error": generation_response.get("error") or "Guided MCQ generation failed.",
                "planner_elapsed_ms": round(planner_elapsed_ms, 2),
                "solution_generation_elapsed_ms": round(solution_generation_elapsed_ms, 2),
                "mcq_generation_elapsed_ms": round(mcq_generation_elapsed_ms, 2),
                "answer_tokens_advisory": guided_answer_tokens,
            }

        try:
            generated_payload = self._parse_json(generation_response.get("result"))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            detail = str(exc).strip() or "unknown guided MCQ format"
            return {
                "success": False,
                "error": f"Guided MCQ generator returned an unreadable format: {detail}",
                "planner_elapsed_ms": round(planner_elapsed_ms, 2),
                "solution_generation_elapsed_ms": round(solution_generation_elapsed_ms, 2),
                "mcq_generation_elapsed_ms": round(mcq_generation_elapsed_ms, 2),
                "answer_tokens_advisory": guided_answer_tokens,
            }

        raw_steps = generated_payload.get("steps") if isinstance(generated_payload, dict) else None
        if not isinstance(raw_steps, list) or len(raw_steps) != len(solution_steps):
            return {
                "success": False,
                "error": "Guided MCQ generator returned the wrong number of steps.",
                "planner_elapsed_ms": round(planner_elapsed_ms, 2),
                "solution_generation_elapsed_ms": round(solution_generation_elapsed_ms, 2),
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
                    "error": "Guided MCQ generator returned an invalid step contract.",
                    "planner_elapsed_ms": round(planner_elapsed_ms, 2),
                    "solution_generation_elapsed_ms": round(solution_generation_elapsed_ms, 2),
                    "mcq_generation_elapsed_ms": round(mcq_generation_elapsed_ms, 2),
                    "answer_tokens_advisory": guided_answer_tokens,
                }
            generated_steps.append(normalized_step)

        valid, steps_or_error = self._validate(
            {"steps": generated_steps},
            include_hints,
            len(solution_steps),
            problem,
        )
        if not valid:
            return {
                "success": False,
                "error": steps_or_error,
                "planner_elapsed_ms": round(planner_elapsed_ms, 2),
                "solution_generation_elapsed_ms": round(solution_generation_elapsed_ms, 2),
                "mcq_generation_elapsed_ms": round(mcq_generation_elapsed_ms, 2),
                "answer_tokens_advisory": guided_answer_tokens,
            }

        return {
            "success": True,
            "query": problem,
            "steps": steps_or_error,
            "include_hints": include_hints,
            "include_full_solution": True,
            "full_solution": solution_text,
            "planner_elapsed_ms": round(planner_elapsed_ms, 2),
            "solution_generation_elapsed_ms": round(solution_generation_elapsed_ms, 2),
            "mcq_generation_elapsed_ms": round(mcq_generation_elapsed_ms, 2),
            "answer_tokens_advisory": guided_answer_tokens,
        }

    @staticmethod
    def _parse_numbered_solution(text: str) -> list[str]:
        """Parse a compact numbered solution while keeping multiline equations."""
        if not isinstance(text, str):
            return []

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        steps: list[str] = []
        current: list[str] = []

        for line in lines:
            match = re.match(r"^(?:step\s*)?(\d+)\s*[.)-]\s*(.+)$", line, flags=re.IGNORECASE)
            if match:
                if current:
                    steps.append(" ".join(current).strip())
                current = [match.group(2).strip()]
            elif current:
                current.append(line)

        if current:
            steps.append(" ".join(current).strip())

        return [
            GuidedAttemptService._normalize_generated_text(step)
            for step in steps
            if step.strip()
        ]

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
            # Count operations that genuinely add a solution stage. Do not
            # count verbs such as "determine", "find", or "calculate" because
            # those commonly describe the single requested result itself.
            "derive", "derivation", "prove", "proof", "show that",
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

        steps = max(3, min(8, steps))

        # Advisory completion size scales with the number of generated MCQs.
        # It is telemetry/UI guidance only and is deliberately not passed as
        # max_tokens to the main generator.
        advisory_tokens = max(300, min(900, 70 * steps))
        return steps, advisory_tokens

    @staticmethod
    def _parse_compact_plan(raw: Any) -> tuple[int, int] | None:
        """Parse the tiny planner's N|T response without another model call."""
        if not isinstance(raw, str):
            return None
        match = re.search(r"\\b(\\d{1,2})\\s*[|,;:]\\s*(\\d{2,4})\\b", raw.strip())
        if not match:
            return None
        step_count = int(match.group(1))
        answer_tokens = int(match.group(2))
        if not 3 <= step_count <= 15:
            return None
        if not 120 <= answer_tokens <= 2000:
            return None
        return step_count, answer_tokens

    @staticmethod
    def _clean_mojibake(value: str) -> str:
        """Repair common UTF-8/Latin-1 mojibake in guided display text."""
        if not isinstance(value, str):
            return value
        text = value
        for _ in range(2):
            if not any(marker in text for marker in ("Â", "Ã", "Î", "Ï", "â")):
                break
            try:
                repaired = text.encode("latin1").decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                break
            if repaired == text:
                break
            bad = sum(text.count(marker) for marker in ("Â", "Ã", "Î", "Ï", "â"))
            new_bad = sum(repaired.count(marker) for marker in ("Â", "Ã", "Î", "Ï", "â"))
            if new_bad >= bad:
                break
            text = repaired
        # Some local model/runtime paths produce partially decoded UTF-8,
        # e.g. IÏÂ² instead of Iω². Normalize the stable fragments explicitly
        # before applying the contextual omega replacements.
        text = text.replace("Â²", "²").replace("Â³", "³")
        text = text.replace("Â¹", "¹").replace("Â°", "°")
        text = re.sub(r"IÏ(?=\s*(?:²|³|\^|$))", "Iω", text)
        text = re.sub(r"(?<=[=+\-*/(])\s*Ï(?=\s*(?:$|[+\-*/),]))", "ω", text)
        text = re.sub(r"\bÏ\s*(?=[²2³3\^*/,)\s]|$)", "ω", text)
        # A partially decoded UTF-8 omega can survive as a bare Ï after
        # the Latin-1 repair fails. In guided mathematical output, Ï is the
        # known mojibake form of ω; normalize it directly rather than exposing
        # corrupted symbols to the UI.
        text = text.replace("Ï", "ω")
        return text

    @staticmethod
    def _normalize_generated_text(value: str) -> str:
        if not isinstance(value, str):
            return value
        text = GuidedAttemptService._clean_mojibake(value)
        text = text.replace("∗", "×")
        text = re.sub(r"\bsmallelement\b", "small element", text, flags=re.IGNORECASE)
        return text.strip()

    @staticmethod
    def _validate_single_step(
        payload: Any,
        include_hints: bool,
        is_final: bool,
    ) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None

        required = {"question", "options", "correct_index"}
        allowed = required | ({"hint"} if include_hints else set())
        if not required.issubset(payload):
            return None

        question = payload.get("question")
        stage_answer = payload.get("stage_answer")
        if stage_answer is not None and (
            not isinstance(stage_answer, str) or not stage_answer.strip()
        ):
            return None
        if not isinstance(question, str):
            return None
        question = GuidedAttemptService._normalize_generated_text(question)
        question = re.sub(
            r"^\s*(?:#{1,6}\s*)?(?:step\s*)?\d+\s*[:.)-]?\s*",
            "",
            question,
            flags=re.IGNORECASE,
        ).strip()
        if not question:
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
            normalized_options.append({
                "text": GuidedAttemptService._normalize_generated_text(text)
            })

        correct_index = payload.get("correct_index")
        if (
            not isinstance(correct_index, int)
            or isinstance(correct_index, bool)
            or not 0 <= correct_index < 4
        ):
            return None

        result = {
            "question": question,
            "stage_answer": (
                stage_answer.strip()
                if isinstance(stage_answer, str) and stage_answer.strip()
                else normalized_options[correct_index]["text"]
            ),
            "options": normalized_options,
            "correct_index": correct_index,
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
            question_norm = normalized(question)
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
    def _guided_response_format(
        step_count: int,
        include_hints: bool,
    ) -> dict[str, Any]:
        """Return a compact strict JSON schema for one guided generation call."""
        step_properties: dict[str, Any] = {
            "question": {"type": "string", "minLength": 1},
            "options": {
                "type": "array",
                "minItems": 4,
                "maxItems": 4,
                "items": {"type": "string", "minLength": 1},
            },
            "correct_index": {
                "type": "integer",
                "minimum": 0,
                "maximum": 3,
            },
        }
        required = ["question", "options", "correct_index"]
        if include_hints:
            step_properties["hint"] = {"type": "string", "minLength": 1}
            required.append("hint")

        return {
            "type": "json_object",
            "schema": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "minItems": step_count,
                        "maxItems": step_count,
                        "items": {
                            "type": "object",
                            "properties": step_properties,
                            "required": required,
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["steps"],
                "additionalProperties": False,
            },
        }

    @staticmethod
    def _parse_generated_records(raw: Any, include_hints: bool) -> dict[str, Any]:
        """Parse compact STEP records emitted by the guided generator.

        This deliberately avoids JSON syntax because the small local model is
        much more reliable with fixed delimiter-separated records.
        """
        if not isinstance(raw, str):
            raise ValueError("Expected guided STEP records.")

        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:text|txt)?\s*", "", text, count=1, flags=re.IGNORECASE)
            text = re.sub(r"\s*```\s*$", "", text, count=1)

        steps: list[dict[str, Any]] = []
        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            # Small local instruction models may prefix a valid record with
            # numbering, bullets, or a short label. Strip only that harmless
            # wrapper; the actual field structure remains strict.
            marker_index = line.upper().find("STEP|||")
            if marker_index >= 0:
                line = line[marker_index:].strip()
            else:
                continue

            fields = line.split("|||")
            expected_fields = 8 if include_hints else 7
            if len(fields) != expected_fields:
                raise ValueError(
                    f"Line {line_number} has {len(fields)} fields; expected {expected_fields}."
                )
            if fields[0].strip().upper() != "STEP":
                raise ValueError(f"Line {line_number} has an invalid record prefix.")
            question = fields[1].strip()
            options = [field.strip() for field in fields[2:6]]
            index_text = fields[6].strip()
            if not question:
                raise ValueError(f"Line {line_number} has an empty question.")
            if any(not option for option in options):
                raise ValueError(f"Line {line_number} has an empty option.")
            try:
                correct_index = int(index_text)
            except ValueError as exc:
                raise ValueError(f"Line {line_number} has a non-integer correct_index.") from exc
            if not 0 <= correct_index < 4:
                raise ValueError(f"Line {line_number} has correct_index outside 0-3.")
            step: dict[str, Any] = {
                "question": question,
                "options": options,
                "correct_index": correct_index,
            }
            if include_hints:
                hint = fields[7].strip()
                if not hint:
                    raise ValueError(f"Line {line_number} has an empty hint.")
                step["hint"] = hint
            steps.append(step)

        if not steps:
            raise ValueError("No STEP records found.")
        return {"steps": steps}

    @staticmethod
    def _parse_json(raw: Any) -> Any:
        """Parse model JSON while tolerating harmless local-model formatting."""
        if not isinstance(raw, str):
            raise ValueError("Expected JSON text.")

        text = raw.strip()

        if text.startswith("```"):
            text = re.sub(
                r"^```(?:json)?\\s*",
                "",
                text,
                count=1,
                flags=re.IGNORECASE,
            )
            text = re.sub(r"\\s*```\\s*$", "", text, count=1)

        object_text = GuidedAttemptService._extract_json_object(text)
        if object_text is None:
            # Local instruction models occasionally append an unterminated
            # string/object when generation stops at the context boundary.
            # Recover only when the output has a clear JSON object start and
            # the only missing syntax is closing JSON delimiters.
            object_text = GuidedAttemptService._repair_truncated_json(text)
            if object_text is None:
                raise ValueError("No complete JSON object found.")

        try:
            return json.loads(object_text)
        except json.JSONDecodeError as first_error:
            repaired = GuidedAttemptService._repair_json_string_escapes(object_text)
            if repaired != object_text:
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    pass

            # Some local instruction models emit a Python-literal dictionary
            # (single quotes, True/False, etc.) despite being asked for JSON.
            # This is deterministic parsing only; it never invokes another LLM.
            try:
                parsed = ast.literal_eval(object_text)
            except (SyntaxError, ValueError, TypeError):
                raise first_error
            if isinstance(parsed, (dict, list)):
                return parsed
            raise first_error

    @staticmethod
    def _extract_json_object(text: str) -> str | None:
        """Return the first balanced JSON object from model output."""
        start = text.find("{")
        if start < 0:
            return None

        depth = 0
        in_string = False
        escaped = False

        for index in range(start, len(text)):
            char = text[index]

            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]

        return None

    @staticmethod
    def _repair_truncated_json(text: str) -> str | None:
        """Close only clearly truncated JSON; never invent missing content."""
        start = text.find("{")
        if start < 0:
            return None

        fragment = text[start:]
        stack: list[str] = []
        in_string = False
        escaped = False

        for char in fragment:
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                stack.append("}")
            elif char == "[":
                stack.append("]")
            elif char in "}]" and stack and stack[-1] == char:
                stack.pop()
            elif char in "}]" and stack:
                return None

        # Do not fabricate text inside an unfinished string. If the model
        # stopped after a complete token but before delimiters, closing the
        # string and remaining containers is safe.
        if in_string or not stack:
            if in_string:
                fragment += '"'
            while stack:
                fragment += stack.pop()
            return fragment
        return None

    @staticmethod
    def _repair_json_string_escapes(text: str) -> str:
        """Escape invalid backslashes and control characters inside JSON strings."""
        valid_escapes = {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}
        chars: list[str] = []
        in_string = False
        escaped = False
        index = 0

        while index < len(text):
            char = text[index]

            if not in_string:
                chars.append(char)
                if char == '"':
                    in_string = True
                index += 1
                continue

            if escaped:
                if char not in valid_escapes:
                    chars.append("\\")
                chars.append(char)
                escaped = False
                index += 1
                continue

            if char == "\\":
                chars.append(char)
                escaped = True
                index += 1
                continue

            if char == '"':
                chars.append(char)
                in_string = False
                index += 1
                continue

            if ord(char) < 0x20:
                if char == "\n":
                    chars.append("\\n")
                elif char == "\r":
                    chars.append("\\r")
                elif char == "\t":
                    chars.append("\\t")
                else:
                    chars.append(f"\\u{ord(char):04x}")
                index += 1
                continue

            chars.append(char)
            index += 1

        return "".join(chars)

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
        if not isinstance(steps, list) or not 3 <= len(steps) <= 15:
            return False, "Guided attempt generator returned the wrong number of steps."

        normalized = []

        for step in steps:
            if not isinstance(step, dict):
                return False, "Guided attempt generator returned an invalid step."

            required = {"question", "stage_answer", "options", "correct_index"}
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

                normalized_options.append({
                    "text": GuidedAttemptService._normalize_generated_text(option_text)
                })

            index = step["correct_index"]
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < 4
            ):
                return False, "Guided attempt contains an invalid correct answer."

            stage_answer = step["stage_answer"]
            if not isinstance(stage_answer, str) or not stage_answer.strip():
                return False, "Guided attempt contains an empty stage answer."

            normalized_step = {
                "question": question,
                "stage_answer": stage_answer.strip(),
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

        # Reject only questions whose actual target is an explicitly supplied
        # quantity. Merely mentioning a supplied quantity is valid: formulas such
        # as L = I*omega must use the given angular velocity.
        supplied_terms = set()
        supplied_patterns = (
            r"\b(?:length|mass|radius|time|speed|velocity|acceleration|"
            r"angular velocity|frequency|charge|voltage|current|pressure|"
            r"temperature|density)\s+(?:is|=|of)\b",
            r"\b(?:with|having|at)\s+(?:a\s+)?(?:length|mass|radius|time|speed|"
            r"velocity|acceleration|angular velocity|frequency|charge|voltage|"
            r"current|pressure|temperature|density)\b",
        )
        quantity_names = (
            "angular velocity", "acceleration", "velocity", "frequency",
            "length", "mass", "radius", "time", "speed", "charge",
            "voltage", "current", "pressure", "temperature", "density",
        )
        for pattern in supplied_patterns:
            for match in re.finditer(pattern, problem, flags=re.IGNORECASE):
                phrase = match.group(0).lower()
                for quantity in quantity_names:
                    if quantity in phrase:
                        supplied_terms.add(quantity)

        def question_target(question: str) -> str:
            # Extract the quantity actually being requested, not every quantity
            # mentioned in the question's context. For example:
            # "What is the moment of inertia ... with angular velocity omega?"
            # targets "moment of inertia", not "angular velocity".
            match = re.search(
                r"\b(?:what is|find|determine|calculate|evaluate)\s+"
                r"(?:the\s+)?(.+?)(?:\?|$)",
                question.lower(),
            )
            if not match:
                return ""

            target = match.group(1)
            # Context normally begins at these boundaries. Keep the first
            # quantity phrase rather than treating later givens as the target.
            target = re.split(
                r"\s+(?:of|for|about|using|with|where|when|given|from|"
                r"in terms of|in)\s+",
                target,
                maxsplit=1,
            )[0]
            target = re.sub(
                r"\b(?:the|a|an|value|expression|result)\b",
                " ",
                target,
            )
            return re.sub(r"\s+", " ", target).strip()

        redundant_supplied_indices: set[int] = set()
        for index, earlier_step in enumerate(normalized[:-1]):
            target = question_target(earlier_step["question"])
            if target and any(
                re.search(rf"\b{re.escape(quantity)}\b", target)
                for quantity in supplied_terms
            ):
                # This is an unnecessary stage, not a reason to discard the
                # whole guided attempt. Remove it deterministically so the
                # student never gets asked to identify data already supplied.
                redundant_supplied_indices.add(index)

        if redundant_supplied_indices:
            normalized = [
                step
                for index, step in enumerate(normalized)
                if index not in redundant_supplied_indices
            ]
            if len(normalized) < 3:
                return False, (
                    "Guided attempt did not retain enough meaningful solution stages."
                )

        # Generic-law filtering is intentionally not performed here. The
        # path-first generation contract already requires every MCQ to correspond
        # one-for-one to a concrete solution-path stage. Keeping the old lexical
        # rejection here would incorrectly reject valid problem-specific formula
        # steps merely because their wording contains "formula" or "relationship".

        def target_key(question: str) -> str:
            q = question.lower()
            q = re.sub(r"\b(?:what is|find|determine|calculate|derive|evaluate)\b", "", q)
            q = re.sub(r"[^a-z0-9]+", " ", q)
            stop = {
                "the", "a", "an", "of", "for", "to", "from", "in", "on",
                "about", "using", "with", "given", "its", "is", "are",
                "and", "by", "as", "value", "terms", "express",
            }
            return " ".join(word for word in q.split() if word not in stop)

        # Some small local model generations collapse all MCQs onto one
        # reusable option pool even when the schema is valid. Repair this
        # deterministically instead of spending another full LLM call.
        stage_answers = []
        for step in normalized:
            answer = re.sub(r"\s+", " ", step["stage_answer"]).strip()
            if answer and answer not in stage_answers:
                stage_answers.append(answer)

        rebuilt_from_stage_answers = False
        if len(stage_answers) >= 4:
            rebuilt_sets = []
            for step in normalized:
                correct_text = re.sub(
                    r"\s+", " ", step["stage_answer"]
                ).strip()
                distractors = [
                    answer for answer in stage_answers
                    if answer != correct_text
                ][:3]
                if len(distractors) == 3:
                    rebuilt_sets.append([correct_text, *distractors])

            if len(rebuilt_sets) == len(normalized):
                rebuilt_from_stage_answers = True
                for step, choices in zip(normalized, rebuilt_sets):
                    step["options"] = [{"text": choice} for choice in choices]
                    step["correct_index"] = 0

        seen_option_sets: set[tuple[str, ...]] = set()
        for step in normalized:
            option_set = tuple(
                sorted(
                    re.sub(r"\s+", " ", option["text"].lower()).strip()
                    for option in step["options"]
                )
            )
            if len(set(option_set)) != 4:
                return False, "Guided attempt contains duplicate options in a question."
            # The same four candidate answers may legitimately be reused
            # across connected questions. The previous hard rejection caused
            # valid path-first generations (especially deterministic fallback
            # generations with exactly four stage answers) to fail despite each
            # question having four distinct options and its own correct index.
            # Do not spend another LLM call trying to repair this.
            seen_option_sets.add(option_set)

        seen_targets: set[str] = set()
        for earlier_step in normalized[:-1]:
            key = target_key(earlier_step["question"])
            if key and key in seen_targets:
                return False, "Guided attempt repeats the same intermediate target."
            seen_targets.add(key)

        # Do not require a particular derivation style. A standard closed-form
        # relation is valid when it is mathematically appropriate; connectedness
        # and target correctness are enforced by the prompt and the checks below.

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
                # A small local model can duplicate one MCQ verbatim even when
                # the rest of the generated path is usable. Keep the later
                # occurrence (the final-result occurrence is normally later
                # after _reorder_final_step) and drop the earlier duplicate.
                # This is deterministic and avoids a second expensive LLM call.
                deduped = []
                seen = set()
                for step in reversed(normalized):
                    key = option_key(step["question"])
                    if key in seen:
                        continue
                    seen.add(key)
                    deduped.append(step)
                deduped.reverse()
                if len(deduped) < 3:
                    return False, "Guided attempt contains repeated questions."
                normalized = deduped

        for step in normalized:
            step.pop("stage_answer", None)

        return True, normalized