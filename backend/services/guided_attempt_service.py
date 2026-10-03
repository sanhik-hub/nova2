import json
import re
import random
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

        if step_count is None:
            plan_response = self.router.plan(
                prompt=f"""
Problem:
{problem}

Choose the smallest sufficient number of progressive solution-path stages
needed to reach the final answer. Use 2 to 5 stages.

Also provide one short ABSTRACT stage description for each stage. The stage
descriptions are structural guidance for another model; they must not solve the
problem.

A stage description must:
- describe a mathematical action or dependency, not a question;
- contain no computed result, final formula, numeric answer, or problem-specific
  equation;
- be useful for constructing the next stage;
- make the final stage the only stage that establishes the requested result.

Return ONLY this JSON object:
{{"step_count": N, "path": ["stage 1", "stage 2", "..."]}}
""".strip(),
                system_prompt="""
You are Nova's guided-attempt planning model.

Your job is ONLY to create a structural dependency plan for a mathematical
problem.

Return exactly one JSON object with:
- "step_count": an integer from 2 through 5;
- "path": an array of exactly step_count short abstract stage descriptions.

Do NOT solve the problem.
Do NOT include the final answer.
Do NOT include problem-specific equations, symbolic results, numeric values,
question text, options, hints, explanations, or answer choices.
Do NOT copy an equation from the problem into the path.
Do NOT describe the answer itself.

Each path item must describe only the kind of dependency that the next model
must turn into a concrete MCQ, for example:
- establish the governing definition or local quantity;
- derive an intermediate relationship;
- apply the relevant condition or substitution;
- combine previously obtained results;
- simplify/evaluate and establish the requested result.

Choose the smallest sufficient dependency chain. Do not force every example
stage into every problem. The final path item must be the only item whose
purpose is to establish the requested final result.
""".strip(),
                max_tokens=220,
            )

            step_count = 3

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
                candidate_path = (
                    plan.get("path")
                    if isinstance(plan, dict)
                    else None
                )

                if (
                    isinstance(planned_count, int)
                    and not isinstance(planned_count, bool)
                    and 2 <= planned_count <= 5
                    and isinstance(candidate_path, list)
                    and len(candidate_path) == planned_count
                    and all(
                        isinstance(item, str) and item.strip()
                        for item in candidate_path
                    )
                ):
                    step_count = planned_count
                    planned_path = [
                        item.strip()
                        for item in candidate_path
                    ]

        if not isinstance(step_count, int) or not 2 <= step_count <= 5:
            return {
                "success": False,
                "error": "Guided attempt step count must be between 2 and 5.",
            }

        hint_instruction = (
            "Include one short hint for each step."
            if include_hints
            else "Do not include a hint field."
        )

        # Keep guided MCQs concise. The planner determines the number of
        # stages; this budget scales with that structure rather than imposing
        # a fixed response size on every problem.
        generation_budget = min(
            900,
            max(600, step_count * (220 if include_hints else 190)),
        )

        system_prompt = """
You are Nova's guided-attempt question writer.

Generate ONLY the multiple-choice steps for the STEM problem.
The small planning model has already selected the number of steps.

Create exactly the requested number of progressive MCQs that collectively
construct the solution to the original STEM problem.

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

A correct answer to an earlier step must be a usable intermediate result,
not a fact that is merely related to the topic. Each step must create an input
for a later step. Do not write independent concept questions that could be
answered without using the previous step.

Never ask for the requested final quantity, its final formula, or an algebraic
equivalent of that final formula before the final step, unless the original
problem is genuinely a one-step calculation. In particular, do not turn a
derivation into a formula-recall quiz.

For derivations and proofs, construct a dependency chain from the first needed
quantity to the requested result. A useful generic pattern is:
- identify the governing definition/law or the first local quantity;
- derive the next intermediate expression from it;
- substitute the relevant relationship or condition;
- accumulate, solve, simplify, or evaluate as required;
- state the requested result only at the final step.
Use only the stages that are actually necessary; do not force every pattern
stage into every problem.

For differentiation, use a dependency chain such as selecting the applicable
rule, differentiating the required terms or inner/outer functions, combining
those results, and simplifying. For equations, use the given condition,
substitution/isolation, solving, and checking only when those stages are needed.
For proofs, establish the necessary identity/definition first, transform it in
small dependent steps, and conclude only at the end. For physics derivations,
start from the local law or differential quantity, express it in terms of the
given variables, then integrate/sum/substitute and simplify as required. Never
ask the student to identify a quantity or parameter that the original problem
already gives explicitly. If a standard governing relation is needed, use it
only when it creates a genuinely new intermediate result; do not turn the
attempt into a formula-recall quiz. These are generic pedagogical patterns,
not fixed question templates.

Do NOT put the final expression, or an algebraically equivalent form of it, in
an earlier step's options. The final expression must be constructed from
preceding results rather than recognized from memory.

Return ONLY one JSON object with exactly one top-level key: "steps".

Every step must contain exactly:
- "question": a concise question
- "options": exactly four objects, each containing only "text"
- "correct_index": zero-based index of the correct option
- "step_goal": a short internal description of the intermediate result this step establishes

"step_goal" is internal generation metadata and will be removed before the
response is returned to the UI.

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
Each "step_goal" must be distinct. The first step_goal must be an intermediate
setup/derivation result; only the final step_goal may be the requested result.

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

        planner_guidance = ""
        if planned_path:
            planner_guidance = f"""
The small planner supplied this ABSTRACT dependency path. It is structural
guidance only; it contains no solution and must not be expanded into a copied
answer. Use exactly one path item as the dependency target for each MCQ, in
order:

{chr(10).join(
    f"{index}. {stage}"
    for index, stage in enumerate(planned_path, start=1)
)}

The concrete mathematical content, equations, options, and correct choices
must be derived by you from the original problem. Do not add, remove, reorder,
or merge stages.
""".strip()

        prompt = f"""
Original problem:
{problem}

Generate exactly {step_count} progressive MCQs forming a complete solution path.
{planner_guidance}

For every step, write only the actual mathematical question in the "question"
field. Do not prefix it with a step number or heading. Also provide a distinct
internal "step_goal" describing the intermediate result established by that
question. The first step_goal must not be the final answer.
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
            problem,
        )

        # Do not automatically run a second full LLM generation when
        # the first draft violates a pedagogical rule. A retry doubles the
        # expensive local-model latency and can reproduce the same violation.
        # Formatting/contract failures are surfaced directly as well; the
        # generation prompt is deliberately strict enough that retries are not
        # part of the normal guided path.
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
        problem: str = "",
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

