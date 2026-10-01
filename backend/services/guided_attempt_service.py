import json
from typing import Any

from backend.services.provider_router import ProviderRouter


class GuidedAttemptService:
    """Generate a concise, validated MCQ guided attempt using the small planner model."""

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

        optional_fields = (
            'The only optional field is "hint". Include it only because hints were requested.'
            if include_hints
            else "Do not generate hints."
        )
        hint_instruction = (
            "Include one short, useful hint for each step."
            if include_hints
            else "Do not include any hint field."
        )

        system_prompt = """
You are Nova's guided-attempt planner.

Create ONLY the multiple-choice steps needed to guide the user through the
given mathematical problem. Do not solve the whole problem, write a full
solution, add explanations, summaries, introductions, conclusions, or any
other content.

Return ONLY one valid JSON object with exactly one top-level key: "steps".
Choose the smallest useful number of steps for the problem:
- very short/simple problems: fewer than four steps
- ordinary problems: about four steps
- large or multi-stage problems: more than four steps when necessary
Use between 2 and 8 steps. Do not add steps merely to reach a target count.

Each step must be a progressive MCQ. Every step must contain exactly:
"question", "options", and "correct_index".
"options" must contain exactly four objects, each with exactly one key: "text".
"correct_index" is zero-based and identifies the single correct option.

Do not reveal later answers inside earlier questions.
Do not invent information that the problem does not provide.
Keep every question and option concise.
$OPTIONAL_FIELDS
""".strip().replace("$OPTIONAL_FIELDS", optional_fields)

        count_instruction = (
            f"Create exactly {step_count} progressive MCQs."
            if step_count is not None
            else "Choose the smallest sufficient number of progressive MCQs yourself."
        )

        prompt = f"""
Problem:
{problem}

{count_instruction}
{hint_instruction}
Output only the JSON object.
""".strip()

        # The planner now emits only MCQ data. Reserve extra completion space
        # only when hints are explicitly requested.
        planner_max_tokens = 1800 if not include_hints else 2400

        response = self.router.plan(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=planner_max_tokens,
        )
        if not response.get("success"):
            return {
                "success": False,
                "error": response.get("error") or "Guided attempt generation failed.",
            }

        try:
            payload = json.loads(response["result"])
        except (TypeError, json.JSONDecodeError):
            return {
                "success": False,
                "error": "Guided attempt planner returned invalid JSON.",
            }

        valid, steps_or_error = self._validate(payload, include_hints, step_count)
        if not valid:
            return {"success": False, "error": steps_or_error}

        return {
            "success": True,
            "query": problem,
            "steps": steps_or_error,
            "include_hints": include_hints,
            "include_full_solution": False,
        }

    def _validate(
        self,
        payload: Any,
        include_hints: bool,
        requested_step_count: int | None = None,
    ):
        if not isinstance(payload, dict) or set(payload) != {"steps"}:
            return False, "Guided attempt planner returned an invalid structure."

        steps = payload.get("steps")
        if not isinstance(steps, list) or not 2 <= len(steps) <= 8:
            return False, "Guided attempt planner returned an invalid number of steps."
        if requested_step_count is not None and len(steps) != requested_step_count:
            return False, "Guided attempt planner returned the wrong number of steps."

        normalized = []
        for step in steps:
            if not isinstance(step, dict):
                return False, "Guided attempt planner returned an invalid step."

            required = {"question", "options", "correct_index"}
            allowed = required | ({"hint"} if include_hints else set())
            if set(step) != allowed:
                return False, "Guided attempt planner returned an invalid step contract."

            question = step["question"]
            if not isinstance(question, str) or not question.strip():
                return False, "Guided attempt contains an empty question."

            options = step["options"]
            if not isinstance(options, list) or len(options) != 4:
                return False, "Each guided step must have exactly four options."

            normalized_options = []
            for option in options:
                if not isinstance(option, dict) or set(option) != {"text"}:
                    return False, "Guided attempt contains an invalid option."
                text = option["text"]
                if not isinstance(text, str) or not text.strip():
                    return False, "Guided attempt contains an empty option."
                normalized_options.append({"text": text.strip()})

            index = step["correct_index"]
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < 4:
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
