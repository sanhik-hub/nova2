import json
import re
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

        if step_count is None:
            plan_response = self.router.plan(
                prompt=f"""
Problem:
{problem}

Choose the smallest useful number of progressive MCQ steps for this problem.
Use 2 to 8 steps:
- simple problem: 2-3
- ordinary problem: about 4
- large or multi-stage problem: more than 4 when necessary

Return ONLY this JSON object:
{{"step_count": N}}
""".strip(),
                system_prompt="""
You are Nova's guided-attempt planner.

Your ONLY job is to decide how many progressive multiple-choice steps are
needed for the given mathematical problem.

Do not solve the problem.
Do not write any questions, options, answers, hints, explanations, or solution.
Return ONLY one JSON object with exactly one key: "step_count".
The value must be an integer from 2 through 8.
Choose the smallest sufficient number.
""".strip(),
                max_tokens=80,
            )

            if not plan_response.get("success"):
                return {
                    "success": False,
                    "error": plan_response.get("error")
                    or "Guided attempt planning failed.",
                }

            try:
                plan = self._parse_json(plan_response.get("result"))
            except (TypeError, ValueError, json.JSONDecodeError):
                return {
                    "success": False,
                    "error": "Guided attempt planner returned invalid JSON.",
                }

            planned_count = plan.get("step_count") if isinstance(plan, dict) else None
            if (
                not isinstance(planned_count, int)
                or isinstance(planned_count, bool)
                or not 2 <= planned_count <= 8
            ):
                return {
                    "success": False,
                    "error": "Guided attempt planner returned an invalid step count.",
                }

            step_count = planned_count

        if not isinstance(step_count, int) or not 2 <= step_count <= 8:
            return {
                "success": False,
                "error": "Guided attempt step count must be between 2 and 8.",
            }

        hint_instruction = (
            "Include one short hint for each step."
            if include_hints
            else "Do not include a hint field."
        )

        generation_budget = min(
            3600,
            max(1200, step_count * (420 if include_hints else 330)),
        )

        system_prompt = """
You are Nova's guided-attempt question writer.

Generate ONLY the multiple-choice steps for the mathematical problem.
The small planning model has already selected the number of steps.

Create exactly the requested number of progressive MCQs. Each step must
move the student toward solving the original problem without revealing
later steps prematurely.

Return ONLY one JSON object with exactly one top-level key: "steps".

Every step must contain exactly:
- "question": a concise question
- "options": exactly four objects, each containing only "text"
- "correct_index": zero-based index of the correct option

Do not include explanations, summaries, introductions, conclusions, or any
other fields. Do not put correct_index inside an option.

Keep the questions and options concise but mathematically correct.
Do not invent information that the original problem does not provide.
$HINT_INSTRUCTION
""".strip().replace("$HINT_INSTRUCTION", hint_instruction)

        if include_hints:
            system_prompt = system_prompt.replace(
                "Do not include explanations, summaries, introductions, conclusions, or any\nother fields.",
                'Do not include explanations, summaries, introductions, conclusions, or any\nother fields except the requested "hint" field.',
            )
            system_prompt = system_prompt.replace(
                '- "correct_index": zero-based index of the correct option',
                '- "correct_index": zero-based index of the correct option\n- "hint": one short hint for this step',
            )

        prompt = f"""
Original problem:
{problem}

Generate exactly {step_count} progressive MCQs.
{hint_instruction}
Output only the JSON object.
""".strip()

        response = self.router.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=generation_budget,
            temperature=0,
            enable_thinking=False,
        )

        if not response.get("success"):
            return {
                "success": False,
                "error": response.get("error")
                or "Guided attempt generation failed.",
            }

        try:
            payload = self._parse_json(response.get("result"))
        except (TypeError, ValueError, json.JSONDecodeError):
            return {
                "success": False,
                "error": "Guided attempt generator returned invalid JSON.",
            }

        valid, steps_or_error = self._validate(
            payload,
            include_hints,
            step_count,
        )
        if not valid:
            return {"success": False, "error": steps_or_error}

        return {
            "success": True,
            "query": problem,
            "steps": steps_or_error,
            "include_hints": include_hints,
            "include_full_solution": False,
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
    ):
        if not isinstance(payload, dict) or set(payload) != {"steps"}:
            return False, "Guided attempt generator returned an invalid structure."

        steps = payload.get("steps")
        if (
            not isinstance(steps, list)
            or not 2 <= len(steps) <= 8
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

            required = {"question", "options", "correct_index"}
            allowed = required | ({"hint"} if include_hints else set())
            if set(step) != allowed:
                return False, "Guided attempt generator returned an invalid step contract."

            question = step["question"]
            if not isinstance(question, str) or not question.strip():
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
                if not isinstance(option, dict) or "text" not in option:
                    return False, "Guided attempt contains an invalid option."

                option_text = option["text"]
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
                "question": question.strip(),
                "options": normalized_options,
                "correct_index": index,
            }

            if include_hints:
                hint = step["hint"]
                if not isinstance(hint, str) or not hint.strip():
                    return False, "Guided attempt contains a missing hint."
                normalized_step["hint"] = hint.strip()

            normalized.append(normalized_step)

        return True, normalized
