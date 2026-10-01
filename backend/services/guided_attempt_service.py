import json
from typing import Any

from backend.services.provider_router import ProviderRouter


class GuidedAttemptService:
    """Generate a validated, step-by-step MCQ attempt using the small planner model."""

    def __init__(self, router: ProviderRouter | None = None):
        self.router = router or ProviderRouter()

    def generate(
        self,
        query: str,
        include_hints: bool = True,
        step_count: int = 4,
    ) -> dict[str, Any]:
        problem = query.strip()
        if not problem:
            return {"success": False, "error": "Question is empty."}

        step_count = max(3, min(step_count, 6))
        hint_instruction = (
            "Provide a useful hint for every step."
            if include_hints
            else "Set every hint field to null. Do not include hints."
        )

        system_prompt = """
You are Nova's guided-attempt planner.
Create a step-by-step mathematical learning attempt for the user's problem.
Return ONLY one valid JSON object. Do not use markdown or code fences.
The JSON must have exactly one top-level key: steps.
steps must be an array of the requested length.
Every step must have exactly these keys:
question, options, correct_index, hint, explanation.
options must contain exactly four objects, each with exactly one key: text.
correct_index is zero-based and must identify the single correct option.
explanation must explain why the correct option is correct.
The steps must progress logically from the given problem toward its solution.
Do not reveal later answers inside earlier questions.
Do not invent information that the problem does not provide.
""".strip()

        prompt = f"""
Problem:
{problem}

Create exactly {step_count} progressive multiple-choice steps.
{hint_instruction}
""".strip()

        # A guided attempt is a substantially larger JSON document than the
        # normal semantic routing plan. Reserve enough completion space for
        # all steps, options, hints, and explanations so the JSON is not
        # truncated before it can be parsed.
        planner_max_tokens = max(1200, step_count * 450)

        response = self.router.plan(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=planner_max_tokens,
        )
        if not response.get("success"):
            return {"success": False, "error": response.get("error") or "Guided attempt generation failed."}

        try:
            payload = json.loads(response["result"])
        except (TypeError, json.JSONDecodeError) as exc:
            return {"success": False, "error": "Guided attempt planner returned invalid JSON."}

        valid, steps_or_error = self._validate(payload, step_count, include_hints)
        if not valid:
            return {"success": False, "error": steps_or_error}

        return {
            "success": True,
            "query": problem,
            "steps": steps_or_error,
            "include_hints": include_hints,
        }

    def _validate(self, payload: Any, step_count: int, include_hints: bool):
        if not isinstance(payload, dict) or set(payload) != {"steps"}:
            return False, "Guided attempt planner returned an invalid structure."

        steps = payload.get("steps")
        if not isinstance(steps, list) or len(steps) != step_count:
            return False, "Guided attempt planner returned the wrong number of steps."

        normalized = []
        for step in steps:
            if not isinstance(step, dict):
                return False, "Guided attempt planner returned an invalid step."
            required = {"question", "options", "correct_index", "hint", "explanation"}
            if set(step) != required:
                return False, "Guided attempt planner returned an invalid step contract."
            if not isinstance(step["question"], str) or not step["question"].strip():
                return False, "Guided attempt contains an empty question."
            options = step["options"]
            if not isinstance(options, list) or len(options) != 4:
                return False, "Each guided step must have exactly four options."
            normalized_options = []
            for option in options:
                if not isinstance(option, dict) or set(option) != {"text"}:
                    return False, "Guided attempt contains an invalid option."
                if not isinstance(option["text"], str) or not option["text"].strip():
                    return False, "Guided attempt contains an empty option."
                normalized_options.append({"text": option["text"].strip()})
            index = step["correct_index"]
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < 4:
                return False, "Guided attempt contains an invalid correct answer."
            explanation = step["explanation"]
            if not isinstance(explanation, str) or not explanation.strip():
                return False, "Guided attempt contains an empty explanation."
            hint = step["hint"]
            if include_hints:
                if not isinstance(hint, str) or not hint.strip():
                    return False, "Guided attempt contains a missing hint."
                hint_value = hint.strip()
            else:
                if hint is not None:
                    return False, "Guided attempt unexpectedly included a hint."
                hint_value = None
            normalized.append({
                "question": step["question"].strip(),
                "options": normalized_options,
                "correct_index": index,
                "hint": hint_value,
                "explanation": explanation.strip(),
            })

        return True, normalized
