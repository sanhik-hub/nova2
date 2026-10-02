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

Choose the number of meaningful solution-path steps required to reach the final
answer. Use 2 to 8 steps:
- simple one-rule problem: 2-3
- ordinary multi-step problem: about 4
- large or multi-stage derivation: more than 4 when necessary

Count mathematical reasoning steps, not merely the number of operations.

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
Choose the smallest sufficient number of meaningful solution-path steps.
""".strip(),
                max_tokens=80,
            )

            # The small planner is advisory. It can occasionally return a
            # syntactically valid but out-of-range estimate (for example 1).
            # Normalize that estimate instead of allowing a weak planner
            # response to break the entire Guided Attempt.
            step_count = 4

            if plan_response.get("success"):
                try:
                    plan = self._parse_json(plan_response.get("result"))
                except (TypeError, ValueError, json.JSONDecodeError):
                    plan = None

                planned_count = (
                    plan.get("step_count")
                    if isinstance(plan, dict)
                    else None
                )

                if (
                    isinstance(planned_count, int)
                    and not isinstance(planned_count, bool)
                ):
                    step_count = planned_count if 2 <= planned_count <= 8 else 4

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

Create exactly the requested number of progressive MCQs that collectively
construct the solution to the original problem.

The MCQs are not independent concept questions. They are the solution path
broken into answerable decisions. Each step must establish a mathematical
result, rule, transformation, substitution, or conclusion that is useful for
the next step.

Every later step must build on the result of the previous step. Do not skip
from the setup directly to the final answer when intermediate mathematics is
needed.

The final step must bring the student to the requested final result. Do not
make the final step merely ask them to repeat an answer that was already given
earlier.

Return ONLY one JSON object with exactly one top-level key: "steps".

Every step must contain exactly:
- "question": a concise question
- "options": exactly four objects, each containing only "text"
- "correct_index": zero-based index of the correct option

Do not include explanations, summaries, introductions, conclusions, or any
other fields. Do not put correct_index inside an option.

Keep the questions and options concise but mathematically correct.
All four options in each question must be distinct.
Exactly one option must be correct.

Prefer concrete mathematical decisions over meta-questions such as
"What is the first step?" when the actual mathematical operation can be asked
directly.

Do NOT put "Step 1", "Step 2", "STEP 1", numbered headings, markdown headings,
or any other step label in the question text. The UI supplies the step number.
Do not repeat the same question across steps. Each question must ask for a
different intermediate result that is necessary for the next step.

Do not reveal the final answer in an earlier step.
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

Generate exactly {step_count} progressive MCQs forming a complete solution path.
For every step, write only the actual mathematical question in the "question"
field. Do not prefix it with a step number or heading.
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

        return True, normalized
