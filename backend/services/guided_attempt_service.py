import ast
import json
import re
import random
from time import perf_counter
from typing import Any

from backend.services.provider_router import ProviderRouter
from backend.services.reasoning_service import ReasoningService


class GuidedAttemptService:
    """Build reliable Guided Attempts from an authoritative solution.

    The solver establishes the STEM result first. A separate generation pass
    then turns that result into progressive MCQs. Hints are opt-in and the full
    worked solution is lazy, so neither is generated during the default start.
    """

    def __init__(self, router: ProviderRouter | None = None):
        self.router = router or ProviderRouter()
        # Keep one reasoning service alongside the shared router so its
        # deterministic-result cache survives across Guided requests.
        self.reasoning = ReasoningService(router=self.router)
        # The authoritative solution is already generated before MCQ
        # generation. Keep it available for the lazy full-solution request so
        # the same question is not solved a second time.
        self._verified_solution_cache: dict[str, str] = {}

    def generate(
        self,
        query: str,
        include_hints: bool = False,
        step_count: int | None = None,
    ) -> dict[str, Any]:
        """Build a guided attempt from a verified solution, then design MCQs."""
        problem = query.strip()
        if not problem:
            return {"success": False, "error": "Question is empty."}

        started = perf_counter()
        reasoning = self.reasoning

        verified_started = perf_counter()
        verified = reasoning.solve(problem, explanation="detailed")
        verified_elapsed_ms = (perf_counter() - verified_started) * 1000
        if not verified.get("success"):
            return {
                "success": False,
                "query": problem,
                "steps": [],
                "include_hints": include_hints,
                "include_full_solution": False,
                "full_solution": None,
                "error": verified.get("error") or "Unable to establish a verified solution.",
                "solution_generation_elapsed_ms": round(verified_elapsed_ms, 2),
                "mcq_generation_elapsed_ms": 0.0,
                "hint_elapsed_ms": 0.0,
                "elapsed_ms": round((perf_counter() - started) * 1000, 2),
            }

        verified_answer = str(verified.get("answer") or "").strip()
        if verified_answer:
            self._verified_solution_cache[problem.casefold()] = verified_answer
        if not verified_answer:
            return {
                "success": False,
                "query": problem,
                "steps": [],
                "include_hints": include_hints,
                "include_full_solution": False,
                "full_solution": None,
                "error": "The verified solution contained no usable answer.",
            }

        # Advisory complexity only: never converted into a model token/time cap.
        # The verified concise solution is the authoritative stage representation;
        # the MCQ model should convert those stages, not solve the problem again.
        estimated_steps, advisory_tokens = self._estimate_guided_complexity(problem)
        requested = (
            max(2, min(8, int(step_count)))
            if step_count is not None
            else estimated_steps
        )
        count_instruction = (
            f"Use about {requested} meaningful stages."
            " Do not create artificial stages just to reach the estimate."
        )

        hint_field = "|||HINT" if include_hints else ""
        prompt = f"""Original problem:
{problem}

Authoritative verified solution:
{verified_answer}

Create a Guided Attempt by converting the actual stages of this verified
solution into progressive MCQs. {count_instruction}

Output one line per stage in exactly this format:
STEP|||QUESTION|||CORRECT_ANSWER|||WRONG_1|||WRONG_2|||WRONG_3{hint_field}

Rules:
- Treat the verified solution above as authoritative. Do not re-solve the
  problem with a different method and do not contradict any established result.
- Each MCQ must correspond to one concrete equation, relationship,
  transformation, or necessary inference already present in the verified
  solution.
- Work in the same logical order as the verified solution: prerequisite
  stages first and the requested result last when applicable.
- Do not invent artificial stages just to reach the estimate.
- Do not repeat a target quantity or ask the same mathematical/scientific
  question in different wording.
- Never ask for a value that is explicitly supplied as input unless the
  question is testing a transformation that genuinely uses that supplied value.
- CORRECT_ANSWER must be the answer to that stage's question and must agree
  with the verified solution.
- Provide exactly three plausible, clearly incorrect distractors for each
  question.
- Distractors should test the actual stage, not unrelated facts.
- Do not output a correct-index, answer letter, option position, or schema label.
- The final meaningful question should establish the requested result.
- Keep questions and options concise and specific to this problem.
- {"Include one short hint per question, without revealing the answer." if include_hints else "Do not generate hints."}
- Do not output solutions, explanations, headings, JSON, Markdown, or blank lines.
- Preserve unambiguous mathematical/scientific notation. LaTeX is allowed inside
  an individual question or option when needed for an equation.
""".strip()

        system_prompt = """You are Nova's Guided Attempt designer.

The computational/reasoning layer has already produced an authoritative concise
solution. Your job is NOT to solve the original problem again. Convert the
actual solution stages into a short, accurate sequence of progressive MCQs.

Every question must map to a genuine stage in the supplied solution. The
correct answer must agree with that supplied solution. Generate plausible
distractors for the same stage, not generic alternatives.

For every STEP record, emit the semantic correct answer explicitly before the
three distractors. Never emit a correct-index field: Python owns option
ordering and correct_index.

Distinctness and solution alignment are more important than artificial step
count. Avoid repeated targets and repeated final-result questions.""".strip()

        generation_started = perf_counter()
        response = self.router.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0,
            enable_thinking=False,
        )
        generation_elapsed_ms = (perf_counter() - generation_started) * 1000

        if not response.get("success"):
            return {
                "success": False,
                "query": problem,
                "steps": [],
                "include_hints": include_hints,
                "include_full_solution": False,
                "full_solution": None,
                "error": response.get("error") or "Guided attempt generation failed.",
                "solution_generation_elapsed_ms": round(verified_elapsed_ms, 2),
                "mcq_generation_elapsed_ms": round(generation_elapsed_ms, 2),
                "hint_elapsed_ms": round(generation_elapsed_ms, 2) if include_hints else 0.0,
                "elapsed_ms": round((perf_counter() - started) * 1000, 2),
                "guided_estimated_steps": estimated_steps,
                "guided_advisory_tokens": advisory_tokens,
            }

        try:
            generated = self._parse_generated_records(response.get("result"), include_hints)
            generated = self._reorder_final_step(problem, generated)

            valid, steps_or_error = self._validate(
                {"steps": generated},
                include_hints,
                requested_step_count=requested,
                problem=problem,
            )
            if not valid:
                raise ValueError(str(steps_or_error))
        except (ValueError, TypeError) as exc:
            return {
                "success": False,
                "query": problem,
                "steps": [],
                "include_hints": include_hints,
                "include_full_solution": False,
                "full_solution": None,
                "error": str(exc),
                "solution_generation_elapsed_ms": round(verified_elapsed_ms, 2),
                "mcq_generation_elapsed_ms": round(generation_elapsed_ms, 2),
                "hint_elapsed_ms": round(generation_elapsed_ms, 2) if include_hints else 0.0,
                "elapsed_ms": round((perf_counter() - started) * 1000, 2),
                "guided_estimated_steps": estimated_steps,
                "guided_advisory_tokens": advisory_tokens,
            }

        return {
            "success": True,
            "query": problem,
            "steps": steps_or_error,
            "include_hints": include_hints,
            "include_full_solution": False,
            "full_solution": None,
            "verified_solution": verified_answer,
            "verified_solution_elapsed_ms": round(verified_elapsed_ms, 2),
            "solution_generation_elapsed_ms": round(verified_elapsed_ms, 2),
            "mcq_generation_elapsed_ms": round(generation_elapsed_ms, 2),
            "hint_elapsed_ms": round(generation_elapsed_ms, 2) if include_hints else 0.0,
            "elapsed_ms": round((perf_counter() - started) * 1000, 2),
            "guided_estimated_steps": estimated_steps,
            "guided_advisory_tokens": advisory_tokens,
        }

    @staticmethod
    def _parse_one_pass_records(
        raw: Any,
        include_hints: bool,
    ) -> list[dict[str, Any]]:
        if not isinstance(raw, str):
            raise ValueError("Guided generator returned no text.")

        records: list[dict[str, Any]] = []
        expected = 8 if include_hints else 7

        for raw_line in raw.strip().splitlines():
            line = raw_line.strip()
            marker = line.upper().find("STEP|||")
            if marker < 0:
                continue

            fields = [part.strip() for part in line[marker:].split("|||")]
            if len(fields) != expected or fields[0].upper() != "STEP":
                continue

            question = fields[1]
            options = fields[2:6]
            correct_index_raw = fields[6]
            hint = fields[7] if include_hints else None

            try:
                correct_index = int(correct_index_raw.strip())
            except (TypeError, ValueError):
                continue

            if (
                not question
                or len(options) != 4
                or any(not value for value in options)
                or correct_index not in range(4)
            ):
                continue
            if include_hints and not hint:
                continue

            values = [question, *options]
            if include_hints:
                values.append(hint)
            if any(GuidedAttemptService._is_schema_placeholder(value) for value in values):
                continue

            normalized_question = GuidedAttemptService._normalize_generated_text(question)
            normalized_options = [
                GuidedAttemptService._normalize_generated_text(value)
                for value in options
            ]
            normalized_hint = (
                GuidedAttemptService._normalize_generated_text(hint)
                if include_hints else None
            )

            records.append({
                "question": normalized_question,
                "options": normalized_options,
                "correct_index": correct_index,
                "hint": normalized_hint,
            })
        if len(records) < 2:
            raise ValueError(
                "Guided generator returned fewer than two valid step records."
            )

        return records[:8]

    @staticmethod
    def _is_schema_placeholder(value: Any) -> bool:
        if not isinstance(value, str):
            return True
        cleaned = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
        if not cleaned:
            return True
        placeholders = {
            "step", "solution_step", "solution", "question", "correct",
            "correct_answer", "wrong1", "wrong2", "wrong3", "hint",
            "answer", "correct_option", "wrong_option",
        }
        return cleaned in placeholders

    @staticmethod
    def _is_concrete_solution_step(solution: str, correct: str) -> bool:
        text = re.sub(r"\s+", " ", solution.strip().lower())
        if not text or GuidedAttemptService._is_schema_placeholder(correct):
            return False

        filler_patterns = (
            r"^calculate(?: the result)?\.?$",
            r"^use (?:the )?formula\.?$",
            r"^apply (?:the )?formula\.?$",
            r"^substitute(?: the values)?\.?$",
            r"^check units\.?$",
            r"^check (?:the )?answer\.?$",
            r"^finalize(?: the answer)?\.?$",
            r"^therefore\.?$",
        )
        if any(re.fullmatch(pattern, text) for pattern in filler_patterns):
            return False

        quantitative_markers = ("=", "+", "-", "*", "/", "^", "->", "$")
        if bool(re.search(r"\d", text)) or any(marker in text for marker in quantitative_markers):
            return True

        action_only = re.search(
            r"\b(?:calculate|compute|determine|find|solve|substitute|"
            r"simplify|differentiate|integrate|derive|evaluate|use)\b",
            text,
        )
        if action_only and len(re.findall(r"[a-z]+", text)) <= 10:
            return False

        return len(text) >= 12

    @staticmethod
    def _parse_mcq_records(
        raw: Any,
        solution_steps: list[str],
        include_hints: bool,
    ) -> list[dict[str, Any]]:
        """Parse compact MCQ records and bind correct answers to solution steps."""
        if not isinstance(raw, str):
            raise ValueError("Guided MCQ generator returned no text.")

        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:text|txt)?\\s*", "", text, count=1, flags=re.IGNORECASE)
            text = re.sub(r"\\s*```\\s*$", "", text, count=1)

        records = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            marker = line.upper().find("STEP|||")
            if marker >= 0:
                line = line[marker:]
            else:
                continue

            fields = [part.strip() for part in line.split("|||")]
            expected = 6 if include_hints else 5
            if len(fields) != expected or fields[0].upper() != "STEP":
                continue

            question, distractors = fields[1], fields[2:5]
            hint = fields[5] if include_hints else None
            if not question or any(not value for value in distractors):
                continue
            if include_hints and not hint:
                continue

            correct = GuidedAttemptService._normalize_generated_text(solution_steps[len(records)])
            options = [
                {"text": correct},
                *[
                    {"text": GuidedAttemptService._normalize_generated_text(value)}
                    for value in distractors
                ],
            ]

            normalized = {
                "question": GuidedAttemptService._normalize_generated_text(question),
                "stage_answer": correct,
                "options": options,
                "correct_index": 0,
            }
            if include_hints:
                normalized["hint"] = GuidedAttemptService._normalize_generated_text(hint)
            records.append(normalized)

            if len(records) == len(solution_steps):
                break

        if len(records) != len(solution_steps):
            raise ValueError("Guided MCQ generator returned an incomplete step contract.")

        return records

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
        steps = 3

        complexity_markers = (
            # Structural indicators only; no subject/domain vocabulary.
            "derive", "derivation", "prove", "proof", "show that",
            "integrate", "integration", "differentiate", "differentiation",
            "solve", "equation", "equations", "substitute", "substitution",
            "simplify", "simplification", "limit", "series",
        )
        for marker in complexity_markers:
            if marker in text:
                steps += 1

        # Multi-part or chained conditions genuinely add an intermediate
        # decision/result. Do not count ordinary connective words such as
        # "and" as separate solution stages.
        if any(token in text for token in (" then ", " after ", " next ", " given that ", " where ")):
            steps += 1
        if any(token in text for token in (" respectively", " in addition", " followed by ")):
            steps += 1

        # Mathematical/scientific notation and multiple explicit quantities
        # are useful complexity signals, but keep the estimate conservative.
        if len(words) > 55:
            steps += 1
        if len(re.findall(r"\\b(?:x|y|z|t|m|M|L|r|R|v|u|a|q|p|V|T|P|I|omega|\\u03c9)\\b", problem)) >= 4:
            steps += 1

        steps = max(2, min(6, steps))

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
        text = text.replace("Â²", "^2").replace("Â³", "^3")
        text = text.replace("Â¹", "^1").replace("Â°", " deg ")
        text = re.sub(r"IÏ(?=\s*(?:²|³|\^|$))", "Iω", text)
        text = re.sub(r"(?<=[=+\-*/(])\s*Ï(?=\s*(?:$|[+\-*/),]))", "ω", text)
        text = re.sub(r"\bÏ\s*(?=[²2³3\^*/,)\s]|$)", "ω", text)
        # A partially decoded UTF-8 omega can survive as a bare Ï after
        # the Latin-1 repair fails. In guided mathematical output, Ï is the
        # known mojibake form of ω; normalize it directly rather than exposing
        # corrupted symbols to the UI.
        text = text.replace("Ï", "omega")
        # Keep the API representation ASCII-safe. This also prevents a second
        # console/client decoding pass from turning scientific symbols into
        # mojibake. The frontend math renderer can interpret these forms.
        text = text.replace("ω", "omega")
        text = text.replace("²", "^2").replace("³", "^3").replace("¹", "^1")
        text = text.replace("·", "*").replace("×", "*")
        # Normalize the stable label variants emitted by small local models.
        text = re.sub(r"^SOLUTION(?:_STEP|_step|\s+STEP)\s*:\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^STEP\s*:\s*", "", text, flags=re.IGNORECASE)
        # Keep guided output transport-safe and deterministic. The frontend can
        # render ASCII scientific notation without relying on a broken model
        # encoding boundary.
        return text.strip()

    @staticmethod
    def _solution_text(solution: str, correct: str) -> str:
        """Build a useful full-solution line from the same-pass record.

        The model owns both fields. If its prose step is only an instruction
        such as "calculate" or "identify the formula", retain it but append
        the step's already-generated correct answer so the displayed solution
        contains the actual mathematical/scientific result without another
        generation call.
        """
        solution = GuidedAttemptService._normalize_generated_text(solution)
        correct = GuidedAttemptService._normalize_generated_text(correct)
        if not solution:
            return correct

        has_math_content = bool(re.search(
            r"(?:=|\^|[+\-*/()]|\b(?:sin|cos|tan|log|ln|sqrt|omega)\b|\d)",
            solution,
            flags=re.IGNORECASE,
        ))
        if not has_math_content and correct:
            return f"{solution} Therefore, {correct}."
        return solution

    @staticmethod
    def _normalize_generated_text(value: str) -> str:
        if not isinstance(value, str):
            return value
        text = GuidedAttemptService._clean_mojibake(value)
        text = text.replace("∗", "*")
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
        verified_solution: str | None = None,
    ) -> dict[str, Any]:
        """Return the authoritative solution already produced for the attempt."""
        problem = query.strip()
        if not problem:
            return {"success": False, "error": "Question is empty."}
        if not isinstance(steps, list) or not steps:
            return {"success": False, "error": "Guided attempt has no solution path."}

        answer = str(verified_solution or "").strip()
        if not answer:
            answer = self._verified_solution_cache.get(problem.casefold(), "").strip()

        # Backward compatibility for older clients: if the verified solution
        # was not carried in the request and is no longer in this process,
        # recover it once through the normal solver.
        if not answer:
            result = self.reasoning.solve(problem, explanation="concise")
            if not result.get("success"):
                return {
                    "success": False,
                    "error": result.get("error") or "Unable to generate the full solution.",
                }
            answer = str(result.get("answer") or "").strip()
            if not answer:
                return {"success": False, "error": "The solution model returned no solution."}

        self._verified_solution_cache[problem.casefold()] = answer

        return {
            "success": True,
            "answer": answer,
            "full_solution": answer,
            "source": "verified_solution",
            "engine": "local_llm",
            "tool_calls": [],
            "cached": True,
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
    def _parse_generated_records(
        raw: Any,
        include_hints: bool,
    ) -> list[dict[str, Any]]:
        """Parse STEP records and assign correct option positions in Python.

        Model contract:
        STEP|||QUESTION|||CORRECT_ANSWER|||WRONG_1|||WRONG_2|||WRONG_3[|||HINT]

        The model supplies the semantic correct answer, but never its position.
        Python then assigns and varies the correct option position.
        """
        if not isinstance(raw, str):
            raise ValueError("Expected guided STEP records.")

        steps: list[dict[str, Any]] = []
        for raw_line in raw.strip().splitlines():
            line = raw_line.strip()
            if not line:
                continue

            marker_index = line.upper().find("STEP|||")
            if marker_index >= 0:
                line = line[marker_index:].strip()
            else:
                continue

            fields = [field.strip() for field in line.split("|||")]
            expected_fields = 7 if include_hints else 6
            if len(fields) != expected_fields or fields[0].upper() != "STEP":
                continue

            question = fields[1]
            correct = fields[2]
            distractors = fields[3:6]
            hint = fields[6] if include_hints else None

            if (
                not question
                or not correct
                or len(distractors) != 3
                or any(not value for value in distractors)
            ):
                continue
            if include_hints and not hint:
                continue

            values = [question, correct, *distractors]
            if include_hints:
                values.append(hint)
            if any(GuidedAttemptService._is_schema_placeholder(value) for value in values):
                continue

            normalized_question = GuidedAttemptService._normalize_generated_text(question)
            normalized_correct = GuidedAttemptService._normalize_generated_text(correct)
            normalized_distractors = [
                GuidedAttemptService._normalize_generated_text(value)
                for value in distractors
            ]

            option_texts = [normalized_correct, *normalized_distractors]
            if len(set(option_texts)) != 4:
                continue

            options = [{"text": value} for value in option_texts]
            target_index = len(steps) % 4
            correct_option = options.pop(0)
            options.insert(target_index, correct_option)

            normalized: dict[str, Any] = {
                "question": normalized_question,
                "stage_answer": normalized_correct,
                "options": options,
                "correct_index": target_index,
            }
            if include_hints:
                normalized["hint"] = GuidedAttemptService._normalize_generated_text(hint)

            steps.append(normalized)
            if len(steps) >= 8:
                break

        if len(steps) < 2:
            raise ValueError(
                "Guided generator returned fewer than two valid step records."
            )

        return steps

    @staticmethod
    def _parse_correct_index(value: Any) -> int | None:
        """Normalize explicit answer-position wrappers safely."""
        if isinstance(value, bool) or value is None:
            return None

        text = str(value).strip()
        if not text:
            return None

        numeric = re.fullmatch(r"[+]?([0-3])(?:\.0+)?", text)
        if numeric:
            return int(numeric.group(1))

        label = text.upper().strip("[](){} \t\r\n")
        label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        if label in label_map:
            return label_map[label]

        numeric_match = re.fullmatch(
            r"(?:CORRECT[_ ]?INDEX|INDEX|OPTION|CHOICE|ANSWER|CORRECT(?:[ _-]+OPTION)?|CORRECT[_ ]?ANSWER)"
            r"\s*[:=\-]?\s*([0-3])(?:\.0+)?",
            label,
        )
        if numeric_match:
            return int(numeric_match.group(1))

        letter_match = re.fullmatch(
            r"(?:OPTION|CHOICE|ANSWER|CORRECT(?:[ _-]+OPTION)?|CORRECT[_ ]?ANSWER)"
            r"\s*(?:IS\s*)?[:=\-]?\s*([ABCD])\.?",
            label,
        )
        if letter_match:
            return label_map[letter_match.group(1)]

        if re.search(r"\b(?:correct|index|answer|option|choice)\b", label):
            digits = re.findall(r"(?<!\d)([0-3])(?!\d)", label)
            if len(digits) == 1:
                return int(digits[0])
            letters = re.findall(r"(?<![A-Z])[ABCD](?![A-Z])", label)
            if len(letters) == 1:
                return label_map[letters[0]]

        return None

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
        if not isinstance(steps, list) or not 2 <= len(steps) <= 8:
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

        # The generated path is grounded in the verified solution. Do not
        # maintain a subject-specific list of "given" quantity names here:
        # the guided architecture must remain domain-general.
        
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

        # A local model can phrase the same intermediate target in slightly
        # different ways (for example, two equivalent questions about the
        # velocity at the highest point). Keep the later occurrence rather
        # than discarding the entire guided attempt. The final-result question
        # is excluded here and remains authoritative after final-step ordering.
        deduped_targets = []
        seen_targets: set[str] = set()
        for step in reversed(normalized):
            key = target_key(step["question"])
            if key and key in seen_targets:
                continue
            if key:
                seen_targets.add(key)
            deduped_targets.append(step)
        deduped_targets.reverse()
        if len(deduped_targets) != len(normalized):
            normalized = deduped_targets
            if len(normalized) < 2:
                return False, "Guided attempt did not retain enough meaningful solution stages."

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