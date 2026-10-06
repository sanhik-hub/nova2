import json
import re
import time
from collections import OrderedDict
from threading import Lock
from typing import Any

from backend.services.provider_router import ProviderRouter
from backend.services.expression_normalizer import (
    CONSTANT_MAP,
    FUNCTION_MAP,
    normalize_expression,
)
class ReasoningService:

    # =========================================================
    # PERFORMANCE CONFIG
    # =========================================================

    MAX_TOOL_CALLS = 2

    # The planner only emits a small routing object. This bound applies only
    # to the planner's routing metadata, never to the final STEM answer.
    PLANNER_TOKENS = 96

    # The planner's answer_tokens field is advisory metadata only. Neither
    # planner nor final-answer generation has an application-level token cap.

    # Small in-memory cache for repeated CAS requests.
    CACHE_SIZE = 256

    def __init__(self, router: ProviderRouter | None = None):
        # Share the application-wide router when one is supplied. This
        # keeps the persistent Wolfram session identical across
        # /api/reason, /api/compute, and the health/control endpoints.
        self.router = router or ProviderRouter()

        self._cache = OrderedDict()
        self._cache_lock = Lock()

    # =========================================================
    # PUBLIC API
    # =========================================================

    def solve(
        self,
        problem: str,
        explanation: bool | str = "detailed",
    ) -> dict[str, Any]:

        started = time.perf_counter()

        explanation = self._normalize_explanation_preference(explanation)

        if not problem or not problem.strip():
            return self._error("Problem is empty.")

        problem = problem.strip()

        # -----------------------------------------------------
        # 1. EXACT CACHE
        # -----------------------------------------------------

        # _cache_set stores only deterministic exact-computation results, so
        # semantic/local-LLM answers are never reused from this cache.
        cached = self._cache_get(
            problem,
            explanation,
        )

        if cached is not None:
            cached["cached"] = True
            cached["elapsed_ms"] = round(
                (time.perf_counter() - started) * 1000,
                2,
            )
            return cached


        # -----------------------------------------------------
        # 2. DETERMINISTIC SYMBOLIC FAST PATH
        # -----------------------------------------------------
        #
        # Unambiguous symbolic operations are already understood by the
        # operation detector and can be executed directly by Wolfram. In
        # result-only mode there is deliberately NO LLM call and the response
        # contains only the exact computational result.
        # -----------------------------------------------------

        explicit_operation = self._detect_explicit_operation(problem)

        # Result-only mode may also use the exact CAS path for clear natural-language calculus requests.
        if explanation == "off" and explicit_operation is None:
            explicit_operation = self._detect_result_only_operation(problem)

        if explicit_operation is not None:
            operation, arguments = explicit_operation
            tool_result = self._execute_tool(operation, arguments)

            if not tool_result.get("success"):
                return self._error(
                    tool_result.get("error") or "The computation failed."
                )

            if explanation == "off":
                response = {
                    "success": True,
                    "answer": str(tool_result.get("result") or "").strip(),
                    "result_only": True,
                    "engine": "wolfram",
                    "tool_calls": [],
                    "cached": False,
                }
            else:
                response = self._final_explanation(
                    problem=problem,
                    tool_results=[tool_result],
                    explanation=explanation,
                )

            response["elapsed_ms"] = round(
                (time.perf_counter() - started) * 1000,
                2,
            )
            self._cache_set(problem, response, explanation)
            return response

        # -----------------------------------------------------
        # 3. DIRECT EXPRESSION FAST PATH
        # -----------------------------------------------------
        #
        # Bare mathematical expressions are also exact CAS requests. Keep
        # these out of the planner entirely.
        # -----------------------------------------------------

        if self._is_simple_expression(problem):
            calculation = self.router.calculate(
                expression=normalize_expression(problem),
                provider="wolfram",
            )

            if not calculation.get("success"):
                return self._error(
                    calculation.get("error") or "The computation failed."
                )

            if explanation == "off":
                response = {
                    "success": True,
                    "answer": str(calculation.get("result") or "").strip(),
                    "result_only": True,
                    "engine": "wolfram",
                    "tool_calls": [],
                    "cached": False,
                }
            else:
                response = self._final_explanation(
                    problem=problem,
                    tool_results=[{"result": calculation.get("result")}],
                    explanation=explanation,
                )

            response["elapsed_ms"] = round(
                (time.perf_counter() - started) * 1000,
                2,
            )
            self._cache_set(problem, response, explanation)
            return response

        # -----------------------------------------------------
        # 4. DIRECT SEMANTIC FAST PATH
        # -----------------------------------------------------
        #
        # Short natural-language requests that contain no concrete
        # computational payload do not need the 0.5B planner first. The
        # planner is valuable when routing exact computation, but using it
        # for simple factual/conceptual requests adds a complete model
        # generation before the actual answer generation.
        #
        # This is deliberately structural rather than domain-specific:
        # there are no chemistry/physics/math subject words here.
        # -----------------------------------------------------

        if self._is_direct_semantic_request(problem):
            response = self._final_explanation(
                problem=problem,
                idea="",
                reasoning=[],
                tool_results=[],
                explanation=explanation,
                estimated_answer_tokens=None,
            )
            response["elapsed_ms"] = round(
                (time.perf_counter() - started) * 1000,
                2,
            )
            response["timings_ms"] = {
                "planner": 0.0,
                "wolfram": 0.0,
                "final_llm": response.pop("_final_llm_ms", 0.0),
            }
            return response

        # -----------------------------------------------------
        # 5. DIRECT QUANTITATIVE HANDOFF
        # -----------------------------------------------------
        if self._is_direct_quantitative_request(problem):
            response = self._solve_quantitative_direct(problem, explanation)
            response["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
            return response

        # -----------------------------------------------------
        # 6. MODEL-DRIVEN SEMANTIC PLANNING
        # -----------------------------------------------------

        response = self._solve_complex(
            problem,
            explanation=explanation,
        )

        response["elapsed_ms"] = round(
            (time.perf_counter() - started) * 1000,
            2,
        )

        if response.get("success"):
            self._cache_set(problem, response, explanation)

        return response

    def solve_follow_up(
        self,
        original_question: str,
        original_answer: str,
        history: list[dict[str, str]],
        follow_up: str,
    ) -> dict[str, Any]:
        """Answer a follow-up using the complete local question-session history."""
        original_question = original_question.strip()
        original_answer = original_answer.strip()
        follow_up = follow_up.strip()

        if not original_question or not follow_up:
            return self._error("Both the original question and follow-up are required.")

        turns: list[str] = [
            f"Original question:\n{original_question}",
            f"Original answer:\n{original_answer}",
        ]

        for index, item in enumerate(history, start=1):
            if not isinstance(item, dict):
                continue
            user_text = str(item.get("question") or "").strip()
            answer_text = str(item.get("answer") or "").strip()
            if user_text:
                turns.append(f"Follow-up {index} question:\n{user_text}")
            if answer_text:
                turns.append(f"Follow-up {index} answer:\n{answer_text}")

        turns.append(f"Current follow-up question:\n{follow_up}")

        prompt = """You are answering a follow-up question in an ongoing STEM problem-solving session.

Use the original question, original answer, and every previous follow-up turn below as authoritative local session context. Resolve references such as "this", "that result", "the value above", "what if", or "why" from that context. Do not reset the problem to a new unrelated problem unless the user explicitly asks for one.

You handle mathematics, physics, chemistry, and general science. Preserve all stated quantities, symbols, structures, assumptions, units, atom mappings, and prior results unless the current follow-up explicitly changes them.

For calculations, perform the required reasoning and show the actual equations/substitution needed for the answer. For chemistry, preserve the stated substrate and atom mapping and keep formulas, bonds, valence, formal charge, stoichiometry, and reaction identity consistent. Do not invent missing data.

Return only the answer to the current follow-up. Keep it very concise but sufficient: give the direct answer and only the minimum equation, calculation, or clarification needed to make it complete. Do not repeat the original solution unless a short reference is necessary. Do not add introductions, conclusions, or meta-commentary.

Scientific display contract:
- Put mathematical expressions in $...$.
- Put chemical formulas, structures, and reaction equations in $...$.
- Use explicit bonds (-, =, equiv), reaction arrows (\rightarrow), and formal charges where relevant.
- Use explicit lone-pair notation when lone pairs matter.
"""

        prompt = prompt + "\n\n" + "\n\n".join(turns)

        started = time.perf_counter()
        try:
            result = self.router.generate(
                prompt=prompt,
                temperature=0.0,
                system_prompt=(
                    "You are Nova's follow-up STEM solver. "
                    "Answer the current follow-up using the complete supplied session context. "
                    "Be correct, direct, and scientifically precise."
                ),
                enable_thinking=False,
                request_timeout=None,
            )
        except Exception as exc:
            return self._error(f"Follow-up processing failed: {exc}")

        if not result.get("success"):
            return self._error(result.get("error") or "Follow-up processing failed.")

        answer = str(result.get("result") or "").strip()
        if not answer:
            return self._error("The follow-up produced no answer.")

        return {
            "success": True,
            "answer": answer,
            "engine": result.get("engine") or "local_llm",
            "tool_calls": result.get("tool_calls", []),
            "cached": False,
            "result_only": False,
            "follow_up": True,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    @staticmethod
    def _normalize_explanation_preference(explanation: bool | str) -> str:
        if explanation is True:
            return "detailed"
        if explanation is False:
            return "off"
        value = str(explanation).strip().lower()
        return value if value in {"off", "concise", "detailed"} else "detailed"

    # =========================================================
    # FAST PROBLEM CLASSIFICATION
    # =========================================================

    def _is_simple_expression(self, problem: str) -> bool:
        """
        Return True only when the complete input is already a mathematical
        expression suitable for direct Wolfram evaluation.

        Classification is structural, not domain-specific. Natural-language
        text must never become a Wolfram expression merely because it contains
        a character that is also a valid mathematical variable.
        """
        text = problem.strip()

        if not text or len(text) > 120:
            return False

        # Bare expressions may contain known functions/constants and
        # single-letter variables. Multi-letter identifiers are accepted only
        # when they are known mathematical functions/constants.
        allowed_identifiers = set(FUNCTION_MAP) | set(CONSTANT_MAP)
        identifiers = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)

        if not identifiers:
            return bool(re.search(r"[0-9]", text))

        for identifier in identifiers:
            if identifier.lower() in allowed_identifiers:
                continue
            if len(identifier) == 1 and identifier.isalpha():
                continue
            return False

        # Require a mathematical atom. This also prevents punctuation-only
        # strings from reaching the CAS.
        return bool(re.search(r"[0-9A-Za-zπ∞√]", text))

    def _is_direct_semantic_request(self, problem: str) -> bool:
        """
        Identify short natural-language requests that do not contain a
        concrete computational payload.

        This classifier is intentionally domain-agnostic. It does not contain
        chemistry, physics, or mathematics vocabulary. Explicit deterministic
        operations and bare mathematical expressions have already been
        handled before this method is reached.

        The planner remains responsible for requests that contain numerical or
        symbolic data because those may require exact computation.
        """
        text = problem.strip()

        if not text or len(text) > 180:
            return False

        # Numerical data is a strong signal that the request may require
        # exact computation, so leave it for semantic planning/tool routing.
        if re.search(r"\d", text):
            return False

        # Operators commonly introduce an actual symbolic computation.
        if re.search(r"[=^*/+]", text):
            return False

        # A standalone subtraction sign is ambiguous in prose, so only treat
        # it as computational when it occurs between symbolic/numeric atoms.
        if re.search(r"[A-Za-z)]\s*-\s*[A-Za-z(]", text):
            return False

        # Explicit deterministic operations were already checked by solve().
        if self._detect_explicit_operation(text) is not None:
            return False

        # Require ordinary natural-language structure rather than punctuation
        # or an isolated identifier.
        words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", text)
        if len(words) < 2 or len(words) > 24:
            return False

        # A very long single sentence is more likely to contain a hidden
        # multi-step problem; let the planner inspect it.
        if len(text.split()) > 24:
            return False

        return True

    def _is_direct_quantitative_request(self, problem: str) -> bool:
        text = problem.strip()
        if not text or len(text) > 500 or not re.search(r"\d", text):
            return False
        if self._is_simple_expression(text) or self._detect_explicit_operation(text) is not None:
            return False
        return len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", text)) >= 3

    # =========================================================
    # COMPLEX COMPUTATIONAL PIPELINE
    # =========================================================

    def _solve_complex(
        self,
        problem: str,
        explanation: str = "detailed",
    ):

        planner_system = r"""
You are Nova's semantic STEM request planner.

Nova Standard Solver handles mathematics, physics, chemistry, and general scientific/conceptual questions. Classify the user's request by meaning, then decide whether exact symbolic/numeric computation is genuinely required.

Do not solve the request and do not write the final answer.

Return ONLY one JSON object with EXACTLY these fields:
type, intent, needs_wolfram, operation, arguments, answer_tokens.

The value of answer_tokens must be a positive integer chosen for this request.

Rules:
- "type" is always "plan".
- "needs_wolfram" is a boolean.
- Route by semantic intent, not by matching isolated subject words or operation keywords.
- "needs_wolfram" means that the final answer genuinely benefits from exact symbolic/numeric computation or exact mathematical transformation that can be represented by one of the allowed structured operations.
- Set "needs_wolfram" to false for factual recall, definitions, laws, units, formulas, nomenclature, identification, qualitative chemistry, conceptual physics, conceptual science, and ordinary explanations.
- A chemistry question is NOT a Wolfram request merely because it mentions a chemical compound, reaction, element, or formula. Factual formulas, nomenclature, identification, and qualitative chemistry should use needs_wolfram=false.
- A physics question is NOT a Wolfram request merely because it is physics. Conceptual laws, definitions, units, and qualitative explanations should use needs_wolfram=false.
- Use Wolfram for physics or chemistry only when the user's concrete problem requires exact symbolic/numeric computation, algebra, equation solving, calculus, or another supported exact operation and the required mathematical data are present.
- Derivations and proofs normally use needs_wolfram=false; use Wolfram only when an explicit concrete expression/equation requires exact symbolic computation.
- Simple arithmetic or a bare mathematical expression may use Wolfram when exact evaluation is the actual task.
- A semantic quantitative problem may use Wolfram for an exact arithmetic
  subcalculation while the local reasoning model remains responsible for the
  scientific interpretation and final solution.
- "operation" is null when exact symbolic/numeric computation is not genuinely needed.
- "arguments" is null when "operation" is null.
- If exact computation is needed, "operation" must be exactly one of:
  calculate, simplify, factor, expand, differentiate, integrate, solve, limit.
- Use "calculate" for concrete numerical arithmetic inside a semantic STEM
  problem. Its arguments must contain one mathematical expression only; do not
  return the final answer yourself.
- "arguments" must be a machine-readable object appropriate to that operation.
  For "calculate", use {"expression":"<mathematical expression>"}.
- Wolfram arguments must contain only mathematical expressions/values from the user's request.
- Never use natural-language instructions as computation arguments.
- Never invent a formula, value, variable, condition, or assumption.
- For derivations, proofs, conceptual explanations, and reasoning, set
  "needs_wolfram" to false and use "operation":null and "arguments":null unless
  the user explicitly supplies a concrete expression/equation that needs exact
  symbolic computation.
- A request to "derive", "prove", or "show that" a formula is normally a
  reasoning task, not a Wolfram task. Do not invent a computation for it.
- Use Wolfram only when a concrete expression/equation from the user's request
  is available and exact symbolic/numeric computation or verification is genuinely useful.
- If no computation is needed, "operation" MUST be null and "arguments" MUST be null.
- If computation is needed, return one structured operation and its arguments;
  do not return a tool name, provider name, array wrapper, or natural-language
  instruction in either field.
- Explicit concrete computation examples:
  - "Differentiate x^3 + 5x" requires exact symbolic computation:
    {"type":"plan","intent":"differentiate","needs_wolfram":true,"operation":"differentiate","arguments":{"expression":"x^3 + 5x","variable":"x"},"answer_tokens":60}
  - "Solve x^2 - 5x + 6 = 0" requires exact equation solving:
    {"type":"plan","intent":"solve","needs_wolfram":true,"operation":"solve","arguments":{"equation":"x^2 - 5x + 6 = 0","variable":"x"},"answer_tokens":60}
  - A conceptual law/definition question is semantic reasoning:
    {"type":"plan","intent":"conceptual","needs_wolfram":false,"operation":null,"arguments":null,"answer_tokens":80}
  - A request for a known chemical formula or name is factual chemistry:
    {"type":"plan","intent":"factual_chemistry","needs_wolfram":false,"operation":null,"arguments":null,"answer_tokens":40}
  - A quantitative semantic problem may use exact arithmetic verification:
    {"type":"plan","intent":"quantitative","needs_wolfram":true,"operation":"calculate","arguments":{"expression":"9*10^9*(2*10^-6)*(3*10^-6)/(0.5^2)"},"answer_tokens":80}
- Do not classify an explicit request to differentiate, integrate, factor, expand, simplify, solve, or take a limit as pure reasoning when the user supplied the concrete expression/equation needed for that exact operation.
- "answer_tokens" is the planner's estimate of the completion tokens actually
  required for the final answer. It is advisory metadata only and is never a
  hard generation ceiling.
- For explanation "off", estimate ONLY the required result that should be shown
  to the user. Do not budget for explanations, introductions, commentary,
  headings, derivations, or summaries. An overall-reaction request should budget
  for the reaction equation; a formula request for the formula; a name request
  for the name; a value request for the value.
- For "concise" and "detailed", estimate the complete response including the  mathematics and prose required by that explanation mode.
- Always use the smallest sufficient estimate for this specific request.
- Never inflate the estimate merely because the topic is difficult.

The planner is a router and computation-request builder, not the final solver.
For a semantic quantitative problem, it may construct the exact arithmetic
expression that the final solver needs Wolfram to evaluate, but it must not
write the final user answer. When information needed for a formula or calculation is missing, leave Wolfram unused rather than guessing it.

Important: do not infer that every math/physics/chemistry request needs Wolfram. The deciding question is whether exact computation is genuinely required for this particular request.
"""

        planner_started = time.perf_counter()

        planner = self.router.plan(
            prompt=json.dumps(
                {
                    "explanation_preference": explanation,
                    "user_request": problem,
                },
                ensure_ascii=False,
            ),
            system_prompt=planner_system,
            # The lightweight planner must use ordinary JSON-object output
            # rather than constrained JSON decoding. Constrained decoding has
            # previously caused severe stalls on this model. The reasoning
            # service already parses and validates the returned object, so the
            # response-format constraint is redundant here.
            max_tokens=self.PLANNER_TOKENS,
            compact=True,
        )

        planner_ms = round(
            (time.perf_counter() - planner_started) * 1000,
            2,
        )

        if not planner["success"]:
            response = self._error(planner["error"])
            response["timings_ms"] = {
                "planner": planner_ms,
                "wolfram": 0.0,
                "final_llm": 0.0,
            }
            return response

        plan = self._parse_json(planner["result"])

        # Lightweight local planner models can occasionally preserve an older
        # top-level type label (for example "reasoning") even when they emit
        # the current structured plan fields correctly. Recover only when the
        # complete current contract is otherwise present; never turn arbitrary
        # planner output into a plan.
        if isinstance(plan, dict) and plan.get("type") != "plan":
            compatible_fields = {
                "intent",
                "needs_wolfram",
                "operation",
                "arguments",
            }
            if compatible_fields.issubset(plan):
                plan["type"] = "plan"

        if not isinstance(plan, dict) or plan.get("type") != "plan":
            # The small planner is advisory. Never turn malformed planner JSON
            # into a user-visible failure when the request itself can be routed
            # safely. Recover exact operations deterministically; otherwise
            # continue as a semantic reasoning request.
            recovered = self._detect_explicit_operation(problem)
            if recovered is not None:
                operation, arguments = recovered
                plan = {
                    "type": "plan",
                    "intent": "solve",
                    "needs_wolfram": True,
                    "operation": operation,
                    "arguments": arguments,
                    "answer_tokens": None,
                }
            else:
                plan = {
                    "type": "plan",
                    "intent": "solve",
                    "needs_wolfram": False,
                    "operation": None,
                    "arguments": None,
                    "answer_tokens": None,
                }

        # Normalize harmless scalar variants emitted by small local models
        # before validating the routing contract.
        if isinstance(plan.get("needs_wolfram"), str):
            flag = plan["needs_wolfram"].strip().lower()
            if flag in {"true", "yes", "1"}:
                plan["needs_wolfram"] = True
            elif flag in {"false", "no", "0"}:
                plan["needs_wolfram"] = False

        if not isinstance(plan.get("intent"), str):
            plan["intent"] = str(plan.get("intent") or "solve")

        required_fields = {
            "intent": str,
            "needs_wolfram": bool,
            "operation": (str, type(None)),
            "arguments": (dict, type(None)),
        }

        for field, expected_type in required_fields.items():
            if field not in plan or not isinstance(plan[field], expected_type):
                recovered = self._detect_explicit_operation(problem)
                if field in {"needs_wolfram", "operation", "arguments"} and recovered is not None:
                    recovered_operation, recovered_arguments = recovered
                    plan["needs_wolfram"] = True
                    plan["operation"] = recovered_operation
                    plan["arguments"] = recovered_arguments
                    continue
                # Invalid optional routing metadata should not prevent the
                # semantic solver from answering. Fall back to no exact tool.
                if field == "needs_wolfram":
                    plan["needs_wolfram"] = False
                    continue
                if field == "operation":
                    plan["operation"] = None
                    continue
                if field == "arguments":
                    plan["arguments"] = None
                    continue
                return self._error(
                    f"Planner returned an invalid plan field: {field}."
                )

        # The new planner contract remains operation + arguments.
        operation = plan.get("operation")
        arguments = plan.get("arguments")

        legacy_calls = plan.get("wolfram")
        if operation is None and plan["needs_wolfram"] and isinstance(legacy_calls, list):
            if len(legacy_calls) == 1 and isinstance(legacy_calls[0], dict):
                operation = legacy_calls[0].get("operation")
                arguments = legacy_calls[0].get("arguments")

        # If a legacy planner emitted a provider/tool-name string such as
        # "wolfram", recover only from an unambiguous explicit operation in the
        # user's request. Do not use this as the primary routing mechanism.
        if operation is None and plan["needs_wolfram"]:
            recovered = self._detect_explicit_operation(problem)
            if recovered is not None:
                operation, arguments = recovered

        # Semantic planning remains primary, but validate the planner against
        # an unambiguous explicit exact-computation request. Small local models
        # can occasionally under-classify these requests as pure reasoning.
        # This is a recovery guard, not the routing architecture: it only
        # upgrades a false negative when the existing structured detector can
        # identify a concrete supported operation and arguments.
        #
        # Some local planner responses violate the contract by returning a
        # valid operation + arguments while incorrectly setting needs_wolfram
        # to false. If the user's request independently matches the same
        # explicit supported operation, promote the request back to exact
        # computation instead of discarding the planner's operation below.
        if not plan["needs_wolfram"]:
            recovered = self._detect_explicit_operation(problem)
            if recovered is not None:
                recovered_operation, recovered_arguments = recovered
                if (
                    isinstance(recovered_operation, str)
                    and isinstance(recovered_arguments, dict)
                    and (
                        operation is None
                        or operation == recovered_operation
                    )
                ):
                    plan["needs_wolfram"] = True
                    operation = recovered_operation
                    arguments = recovered_arguments

        if plan["needs_wolfram"]:
            if not isinstance(operation, str) or not operation.strip() or not isinstance(arguments, dict):
                # The planner is advisory. If it cannot produce a usable
                # exact-computation contract, fall back to semantic solving
                # instead of exposing "invalid plan" to the user.
                recovered = self._detect_explicit_operation(problem)
                if recovered is not None:
                    operation, arguments = recovered
                else:
                    plan["needs_wolfram"] = False
                    operation = None
                    arguments = None
        else:
            operation = None
            arguments = None

        calls = []
        if operation is not None:
            calls.append({
                "operation": operation,
                "arguments": arguments,
            })

        # The answer-size estimate is advisory. Normalize the small planner's
        # occasional JSON-string integer, but never let an unusable estimate
        # prevent an otherwise valid mathematical request from being solved.
        # A missing/invalid estimate means the final model may terminate
        # naturally rather than receiving an artificial fallback cap.
        answer_tokens = plan.get("answer_tokens")
        if isinstance(answer_tokens, str) and re.fullmatch(r"\d+", answer_tokens.strip()):
            answer_tokens = int(answer_tokens.strip())
            plan["answer_tokens"] = answer_tokens

        if type(answer_tokens) is not int or answer_tokens <= 0:
            answer_tokens = None
            plan["answer_tokens"] = None

        calls = calls[:self.MAX_TOOL_CALLS]

        if not plan["needs_wolfram"]:
            calls = []

        if self._contains_undefined_function_call(calls):
            return self._error("Planner produced an unsupported function call.")

        tool_results = []
        wolfram_ms_total = 0.0

        for call in calls:
            validated = self._validate_tool_call({
                "tool": call.get("operation"),
                "arguments": call.get("arguments"),
            })

            if not validated["valid"]:
                return self._error(validated["error"])

            tool_started = time.perf_counter()

            result = self._execute_tool(
                validated["tool"],
                validated["arguments"],
            )

            wolfram_ms_total += round(
                (time.perf_counter() - tool_started) * 1000,
                2,
            )

            if not result.get("success"):
                return self._error(
                    "Wolfram verification failed: "
                    + str(result.get("error"))
                )

            tool_results.append({
                "tool": validated["tool"],
                "arguments": validated["arguments"],
                "result": result.get("result"),
            })

        # -----------------------------------------------------
        # EXPLANATION OFF
        # -----------------------------------------------------
        #
        # If Wolfram produced an exact result, return it directly.
        # Otherwise the final Qwen model still owns the answer. The planner
        # never supplies final solution text.
        # -----------------------------------------------------

        if explanation == "off" and tool_results:
            response = self._cas_result_only(tool_results=tool_results)
            response["timings_ms"] = {
                "planner": planner_ms,
                "wolfram": wolfram_ms_total,
                "final_llm": 0.0,
            }
            return response

        # -----------------------------------------------------
        # ONE FINAL LLM CALL
        # -----------------------------------------------------

        response = self._final_explanation(
            problem=problem,
            idea="",
            reasoning=[],
            tool_results=tool_results,
            explanation=explanation,
            estimated_answer_tokens=answer_tokens,
        )
        response["timings_ms"] = {
            "planner": planner_ms,
            "wolfram": wolfram_ms_total,
            "final_llm": response.pop("_final_llm_ms", 0.0),
        }
        return response

    def _solve_quantitative_direct(self, problem: str, explanation: str):
        """Extract one exact calculation with the main reasoning model, then verify it."""
        extraction = self.router.generate(
            prompt=problem,
            system_prompt=(
                "Extract the single exact numerical calculation required by this STEM problem. "
                "Return ONLY JSON with fields needs_calculation and expression. "
                "If calculation is required, expression must contain only numbers, arithmetic "
                "operators, parentheses, and scientific notation. Do not return the final answer "
                "or units. Do not invent missing data. Interpret the complete problem before "
                "constructing the expression. Preserve every stated quantity, exponent, "
                "denominator, and mathematical operation. Convert every stated scale prefix "
                "or unit multiplier into its numeric factor instead of dropping it. Preserve "
                "powers on quantities exactly, including squared or cubed denominators. Keep "
                "grouping explicit with parentheses. You may introduce a required universal "
                "constant only when the calculation requires it, but never invent a measured "
                "value or condition. Before returning JSON, silently recompute the expression "
                "and check its numerical scale and operation structure against the original "
                "problem."
            ),
            temperature=0.0,
            enable_thinking=False,
            response_format={"type": "json_object"},
            request_timeout=None,
        )
        if not extraction.get("success"):
            return self._error(extraction.get("error") or "Quantitative processing failed.")

        parsed = self._parse_json(extraction.get("result"))
        expression = parsed.get("expression") if isinstance(parsed, dict) else None
        if not isinstance(parsed, dict) or parsed.get("needs_calculation") is not True:
            return self._final_explanation(
                problem=problem, idea="", reasoning=[], tool_results=[], explanation=explanation,
                estimated_answer_tokens=None,
            )
        if not isinstance(expression, str) or not self._is_safe_numeric_expression(expression.strip()):
            return self._error("The reasoning model did not produce a safe calculation expression.")

        started = time.perf_counter()
        result = self._execute_tool("calculate", {"expression": expression.strip()})
        wolfram_ms = round((time.perf_counter() - started) * 1000, 2)
        if not result.get("success"):
            return self._error("Wolfram verification failed: " + str(result.get("error")))

        response = self._final_explanation(
            problem=problem,
            idea="",
            reasoning=[],
            tool_results=[{
                "tool": "calculate",
                "arguments": {"expression": expression.strip()},
                "result": result.get("result"),
            }],
            explanation=explanation,
            estimated_answer_tokens=None,
        )
        response["timings_ms"] = {
            "planner": 0.0,
            "wolfram": wolfram_ms,
            "final_llm": response.pop("_final_llm_ms", 0.0),
        }
        return response

    def _detect_result_only_operation(self, problem: str):
        text = problem.strip()
        patterns = [
            (r"^(?:the )?derivative of\s+(.+?)(?:\s+with respect to\s+([A-Za-z]))?\s*\.?$", "differentiate"),
            (r"^find (?:the )?derivative of\s+(.+?)(?:\s+with respect to\s+([A-Za-z]))?\s*\.?$", "differentiate"),
            (r"^(?:the )?integral of\s+(.+?)(?:\s+(?:with respect to|wrt)\s+([A-Za-z]))?\s*\.?$", "integrate"),
            (r"^integrate\s+(.+?)(?:\s+(?:with respect to|wrt)\s+([A-Za-z]))?\s*\.?$", "integrate"),
        ]
        for pattern, operation in patterns:
            match = re.match(pattern, text, re.IGNORECASE)
            if not match:
                continue
            expression = normalize_expression(match.group(1).strip().strip(chr(96)))
            variable = (match.group(2) or "x").strip()
            if expression and re.search(r"[A-Za-z]", expression):
                return operation, {"expression": expression, "variable": variable}
        return None
    @staticmethod
    def _is_safe_numeric_expression(expression: str) -> bool:
        """Validate an extracted numeric expression without subject-specific rules."""
        text = expression.strip()
        if not text or len(text) > 300:
            return False
        # Only numeric literals, scientific notation, arithmetic operators,
        # parentheses, decimal points, and whitespace are executable.
        if re.search(r"[^0-9eE+\-*/().^%\s]", text):
            return False
        if not re.search(r"\d", text):
            return False
        # Alphabetic characters are permitted only as the e/E in scientific
        # notation and must be attached to a numeric mantissa/exponent.
        if re.search(r"(?<!\d)[eE](?![+\-]?\d)", text):
            return False
        if re.search(r"[A-DF-Za-df-z]", text):
            return False
        return True

    def _detect_explicit_operation(self, problem: str):
        """Detect an unambiguous symbolic operation and build safe arguments."""

        text = problem.strip()

        # DIFFERENTIATE
        match = re.match(
            r"(?is)^\s*(?:differentiate\s+|find\s+(?:the\s+)?derivative\s+(?:of\s+)?|the\s+derivative\s+of\s+)(.+?)\s*[.!?]?\s*$",
            text,
        )
        if match:
            expression = match.group(1).strip()
            expression = normalize_expression(expression.strip("`").strip())
            expression = re.sub(
                r"(?is)\s+with\s+respect\s+to\s+([a-zA-Z])\s*$",
                "",
                expression,
            ).strip()
            expression = re.sub(r"[.!?]+$", "", expression).strip()
            variable = "x"
            variable_match = re.search(
                r"(?is)\s+(?:with\s+respect\s+to|wrt)\s+([a-zA-Z])\s*$",
                match.group(1).strip(),
            )
            if variable_match:
                variable = variable_match.group(1)
            return ("differentiate", {
                "expression": expression,
                "variable": variable,
            })

        # INTEGRATE
        match = re.match(
            r"(?is)^\s*(?:integrate|find\s+(?:the\s+)?integral\s+(?:of\s+)?)(.+?)\s*[.!?]?\s*$",
            text,
        )
        if match:
            expression = match.group(1).strip()
            variable = "x"
            variable_match = re.search(
                r"(?is)\s+with\s+respect\s+to\s+([a-zA-Z])\s*$",
                expression,
            )
            if variable_match:
                variable = variable_match.group(1)
                expression = expression[:variable_match.start()].strip()
            expression = re.sub(r"[.!?]+$", "", expression).strip()
            expression = normalize_expression(expression)
            return ("integrate", {
                "expression": expression,
                "variable": variable,
            })

        # FACTOR / EXPAND / SIMPLIFY
        for operation, pattern in (
            ("factor", r"(?is)^\s*(?:factor|factorize)\s+(.+?)\s*[.!?]?\s*$"),
            ("expand", r"(?is)^\s*expand\s+(.+?)\s*[.!?]?\s*$"),
            ("simplify", r"(?is)^\s*simplify\s+(.+?)\s*[.!?]?\s*$"),
        ):
            match = re.match(pattern, text)
            if match:
                expression = re.sub(r"[.!?]+$", "", match.group(1).strip()).strip()
                return (operation, {"expression": expression})

        # SOLVE
        match = re.match(
            r"(?is)^\s*(?:solve|find\s+(?:the\s+)?(?:roots|solutions))\s+(.+?)\s*[.!?]?\s*$",
            text,
        )
        if match:
            equation = re.sub(r"[.!?]+$", "", match.group(1).strip()).strip()
            equation = re.sub(r"\s*=\s*", " == ", equation)
            if "==" not in equation:
                equation = f"{equation} == 0"
            return ("solve", {
                "equation": equation,
                "variable": "x",
            })

        # LIMIT
        match = re.match(
            r"(?is)^\s*(?:limit|evaluate\s+(?:the\s+)?limit)\s+(?:of\s+)?(.+?)\s*[.!?]?\s*$",
            text,
        )
        if match:
            expression = match.group(1).strip()
            limit_match = re.match(
                r"(?is)^(.+?)\s+(?:as\s+)?([a-zA-Z])\s*(?:approaches|->|→)\s*([^\s,.!?]+)\s*$",
                expression,
            )
            if limit_match:
                return ("limit", {
                    "expression": limit_match.group(1).strip(),
                    "variable": limit_match.group(2),
                    "value": limit_match.group(3),
                })

        return None

    def _validate_tool_call(self, call: dict):

        if not isinstance(call, dict):
            return {
                "valid": False,
                "error": "Tool call is not an object.",
            }

        tool = call.get("tool")
        arguments = call.get("arguments")

        allowed = {
            "calculate",
            "simplify",
            "factor",
            "expand",
            "differentiate",
            "integrate",
            "solve",
            "limit",
        }

        if tool not in allowed:
            return {
                "valid": False,                "error": f"Unsupported tool: {tool}",
            }

        if not isinstance(arguments, dict):
            return {
                "valid": False,
                "error": "Tool arguments must be an object.",
            }

        # -----------------------------------------------------
        # Argument safety
        # -----------------------------------------------------

        for key, value in arguments.items():

            if not isinstance(value, str):
                continue

            if len(value) > 500:
                return {
                    "valid": False,
                    "error": "Tool argument is too long.",
                }

            lowered = value.lower()

            natural_language_patterns = [
                r"\bintegral_\d",
                r"\bfrom\s+\d+\s+to\s+\d+",
                r"\bwhere\b",
                r"\bgiven\b",
                r"\bsuppose\b",
                r"\bcontinuous\b",
                r"\bfunction\s+f\b",
                r"\bprove\b",
                r"\bfind\s+the\b",
            ]

            if any(
                re.search(pattern, lowered)
                for pattern in natural_language_patterns
            ):
                return {
                    "valid": False,
                    "error": (
                        "Rejected unsafe/natural-language "
                        f"Wolfram argument: {value}"
                    ),
                }

        # -----------------------------------------------------
        # Required arguments
        # -----------------------------------------------------

        if tool == "calculate":
            expression = arguments.get("expression")
            if not isinstance(expression, str) or not expression.strip():
                return {
                    "valid": False,
                    "error": "calculate requires an expression.",
                }

            # Keep this general and structural: calculate accepts the same
            # mathematical expression grammar as the direct CAS path, while
            # rejecting natural-language instructions and arbitrary Wolfram
            # code.
            if not self._is_simple_expression(expression):
                return {
                    "valid": False,
                    "error": "calculate requires a valid mathematical expression.",
                }

        if tool in {
            "simplify",
            "factor",
            "expand",
            "differentiate",
            "integrate",
        }:
            if not arguments.get("expression"):
                return {
                    "valid": False,
                    "error": f"{tool} requires an expression.",
                }

        if tool == "solve":

            if not arguments.get("equation"):
                return {
                    "valid": False,
                    "error": "solve requires an equation.",
                }

            if not arguments.get("variable"):
                return {
                    "valid": False,
                    "error": "solve requires a variable.",
                }

        if tool == "limit":

            for field in (
                "expression",
                "variable",
                "value",
            ):
                if not arguments.get(field):
                    return {
                        "valid": False,
                        "error": f"limit requires {field}.",
                    }

        return {
            "valid": True,
            "tool": tool,
            "arguments": arguments,
        }

    # =========================================================
    # UNKNOWN FUNCTION DETECTION
    # =========================================================

    def _contains_undefined_function_call(
        self,
        calls: list,
    ) -> bool:

        known_functions = {
            "sin",
            "cos",
            "tan",
            "cot",
            "sec",
            "csc",
            "asin",
            "acos",
            "atan",
            "sinh",
            "cosh",
            "tanh",
            "sqrt",
            "log",
            "ln",
            "exp",
            "abs",
            "floor",
            "ceil",
            "sign",
        }

        for call in calls:

            # The planner contract requires Wolfram entries to be structured
            # objects. A malformed entry must be rejected here rather than
            # crashing the FastAPI request with AttributeError.
            if not isinstance(call, dict):
                return True

            arguments = call.get(
                "arguments",
                {},
            )

            if not isinstance(arguments, dict):
                return True

            for value in arguments.values():

                if not isinstance(value, str):
                    continue

                # f(x), g(x), h(x), etc.
                matches = re.findall(
                    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                    value,
                )

                for name in matches:

                    if name.lower() not in known_functions:
                        return True

                # Explicit Wolfram Integrate containing
                # unknown parenthesized functions.
                if re.search(
                    r"\bIntegrate\s*\[.*"
                    r"[A-Za-z_][A-Za-z0-9_]*\s*\(",
                    value,
                    flags=re.IGNORECASE,
                ):
                    return True

        return False

    # =========================================================
    # TOOL EXECUTION
    # =========================================================

    def _execute_tool(
        self,
        tool: str,
        arguments: dict,
    ) -> dict:

        try:

            provider = self.router.get_provider("wolfram")

            if tool == "calculate":

                raw = provider.calculate(
                    normalize_expression(
                        arguments["expression"]
                    )
                )

                if raw.get("success"):
                    return {
                        "success": True,
                        "result": raw.get("result"),
                        "error": None,
                    }

                return {
                    "success": False,
                    "result": None,
                    "error": raw.get("error"),
                }

            elif tool == "simplify":

                raw = provider.engine.simplify(
                    normalize_expression(
                        arguments["expression"]
                    )
                )

            elif tool == "factor":

                raw = provider.engine.factor(
                    normalize_expression(
                        arguments["expression"]
                    )
                )

            elif tool == "expand":

                raw = provider.engine.expand(
                    normalize_expression(
                        arguments["expression"]
                    )
                )

            elif tool == "differentiate":

                raw = provider.engine.differentiate(
                    normalize_expression(
                        arguments["expression"]
                    ),
                    arguments.get("variable", "x"),
                )

            elif tool == "integrate":

                raw = provider.engine.integrate(
                    normalize_expression(
                        arguments["expression"]
                    ),
                    arguments.get("variable", "x"),
                )

            elif tool == "solve":

                raw = provider.engine.solve(
                    normalize_expression(
                        arguments["equation"]
                    ),
                    arguments.get("variable", "x"),
                )

            elif tool == "limit":

                raw = provider.engine.limit(
                    normalize_expression(
                        arguments["expression"]
                    ),
                    arguments.get("variable", "x"),
                    arguments["value"],
                )

            else:

                return {
                    "success": False,
                    "result": None,
                    "error": f"Unknown tool: {tool}",
                }

            if raw.success:

                return {
                    "success": True,
                    "result": raw.result,
                    "error": None,
                }

            return {
                "success": False,
                "result": None,
                "error": raw.error,
            }

        except Exception as exc:

            return {
                "success": False,
                "result": None,
                "error": str(exc),
            }

    # =========================================================
    # PURE REASONING RESULT
    # =========================================================

    def _final_from_reasoning(
        self,
        problem: str,
        plan: dict,
    ):

        reasoning = plan.get(
            "reasoning",
            [],
        )

        if not isinstance(reasoning, list):
            reasoning = []

        answer_parts = []

        idea = plan.get("idea", "")

        if isinstance(idea, str) and idea.strip():
            answer_parts.append(
                idea.strip()
            )

        for step in reasoning:

            if isinstance(step, str):
                step = step.strip()

                if step:
                    answer_parts.append(step)

        answer = "\n\n".join(
            answer_parts
        )

        if not answer:
            return self._error(
                "The reasoning model returned no solution."
            )

        return {
            "success": True,
            "answer": answer,
            "engine": "local_llm",
            "tool_calls": [],
            "cached": False,
        }

    # =========================================================
    # RESULT-ONLY CAS RESPONSE
    # =========================================================

    def _cas_result_only(
        self,
        tool_results: list,
    ):
        """
        Return only the exact symbolic result when explanation is off.

        No explanatory prose, tool metadata, steps, or verification
        details are exposed to the caller.
        """

        if not tool_results:
            return self._error(
                "No computational result was produced."
            )

        results = [
            item.get("result")
            for item in tool_results
            if isinstance(item, dict)
            and item.get("result") is not None
        ]

        if not results:
            return self._error(
                "No computational result was produced."
            )

        answer = "\n".join(
            str(result).strip()
            for result in results
            if str(result).strip()
        )

        if not answer:
            return self._error(
                "The computational result was empty."
            )

        return {
            "success": True,
            "answer": answer,
            "result_only": True,
        }

    # =========================================================
    # FINAL EXPLANATION AFTER CAS
    # =========================================================

    def _final_explanation(
        self,
        problem: str,
        idea: str = "",
        reasoning: list | None = None,
        tool_results: list | None = None,
        explanation: str = "detailed",
        estimated_answer_tokens: int | None = None,
    ):

        if explanation == "off":
            final_system = r"""
Return ONLY the exact result requested by the user. No explanation, heading,
lead-in, summary, justification, or commentary.

For chemistry, check the complete atom mapping, bond changes, valence, charge,
stoichiometry, regiochemistry, and product identity internally. Preserve the
user's substrate skeleton unless the requested transformation itself changes
it. Every atom/group in the displayed reaction must be chemically accounted
for. Never substitute a related reaction, intermediate, side product,
mechanism, or invented transformation. If the question describes a reaction
class or substrate class without naming a specific substituent, answer using a
symbolic representation of exactly that stated class. Do not invent extra
substituents, alter the carbon/atom skeleton, or silently replace the stated
substrate with a different one. For an overall reaction, show the net
transformation requested rather than a mechanism or side product. Before
outputting the result, internally verify atom counts, bond changes, valence,
formal charge, stoichiometry, and the identity of the product against the
original substrate.

For chemical structures and reactions, use unambiguous scientific notation.
When a structure is useful, use LaTeX inside $...$ with explicit single (-),
double (=), and triple (\equiv) bonds, explicit formal charges such as
$\mathrm{NH_4^+}$, and explicit lone-pair notation when relevant. Do not rely
on ambiguous plain-text superscripts or spacing.

For mathematics and physics, check the governing relation, dimensions/units,
and required substitution internally. Never invent missing data.

When explanation is off, an instruction such as "derive", "prove", "show",
"calculate", or "explain" does NOT authorize a derivation or explanatory
sentence in the response. Perform the reasoning internally and emit only the
requested final payload. For a requested formula, emit only the formula. For a
requested value, emit only the value with its unit when the unit is part of the
requested result. For a requested equation or reaction, emit only that equation
or reaction.

Output only the requested payload itself.
Never write a sentence such as "The answer is ...", "The result is ...",
"The formula is ...", or "The value is ...". Do not add a label, lead-in,
explanation, justification, or concluding sentence. If the requested result
is a formula, equation, value, name, unit, reaction, structure, or law, output
only that result.
"""
        elif explanation == "concise":
            final_system = r"""
You are Nova's concise STEM solution writer.

Return a concise STEP-BY-STEP solution, not merely the governing formula
and final result. Show the necessary derivation from the starting principle
to the requested result.

Rules:
- Use mathematics/equations as the main content, with very little prose.
- For a derivation or multi-step calculation, show each indispensable
  transformation in order. Do not jump directly from the starting formula
  to the final formula.
- Normally use 3-6 compact steps when the problem genuinely requires them.
  Each step may contain one short explanatory phrase followed by its equation.
- Include the governing relation, relevant substitution or differential
  relation, required integration/algebra/transformation, and final result
  whenever those stages are mathematically necessary.
- Keep prose extremely concise and only use it to explain what a step is doing.
  Do not write paragraphs, introductions, background, repetition, or filler.
- Do not restate quantities already given in the question unless needed in an
  equation.
- Do not omit a mathematically necessary intermediate step merely to make the
  answer shorter.
- Do not preview the final formula/result. State it once, at the end.
- Never stop at an intermediate expression and never invent missing data.
- Do not mention tools, AI, prompts, token budgets, or internal processing.
- Return ONLY the solution text.

For example, an integral derivation should have the structure:
"At distance $r$, $v=\omega r$."
"$d\varepsilon=Bv\,dr=B\omega r\,dr$."
"Integrating along the rod, $\varepsilon=\int_0^l B\omega r\,dr$."
"Therefore, $\varepsilon=\frac12B\omega l^2$."

The example illustrates the required level of derivation; do not copy it
unless it applies to the actual problem.
"""
        else:
            final_system = r"""
You are Nova's detailed STEM solution writer.
Write ONE complete, self-contained solution from beginning to end. The request may involve mathematics, physics, chemistry, or general science.
Explain HOW the result is obtained and show the important equations, transformations, scientific relationships, and rule applications when they are relevant.

For differentiation, explicitly apply the relevant rules. For quantitative physics or chemistry, show the necessary equations, substitutions, units, and calculations. For derivations and proofs, continue through every necessary step to the requested final formula, result, or conclusion. For factual or conceptual questions, answer the actual question clearly without inventing a calculation.

Be complete but disciplined: do not repeat the same idea, add unrelated
background, present alternative methods, or continue after the result is
established. Stop when the solution is complete.

Never stop at an intermediate setup or partial argument. Never invent missing
information or assumptions.

For a derivation or proof, do NOT state or preview the requested final formula,
result, or conclusion before the derivation is complete. Build the result
step by step using the necessary intermediate equations. State the requested
final formula, result, or conclusion exactly ONCE, as the final mathematical
statement of the solution. Do not repeat it in an introduction, summary, or
second final-answer section.

Do not mention tools, AI, prompts, token budgets, or internal processing.
Return ONLY the complete solution text.
"""

        # Use one common scientific-format contract across explanation modes.
        final_system += r"""
Scientific display contract:
- Put mathematical expressions in $...$ and larger derivations in $...$.
- Put chemical formulas, structures, and reaction equations in $...$ so the
  frontend can typeset element subscripts, explicit bonds, charges, and lone
  pairs unambiguously.
- For chemical structures use upright element symbols with explicit bond
  characters: - for single, = for double, \equiv for triple, and
  \rightarrow or \rightleftharpoons for reaction arrows.
- Show formal charges explicitly, e.g. $\mathrm{NH_4^+}$.
- When lone pairs are important, show them explicitly with LaTeX such as
  $\overset{\cdot\,\cdot}{\mathrm{O}}$ rather than relying on prose or
  invisible electron pairs.
- Keep chemical atom counts and charge balance consistent with the displayed
  structure. Do not use ambiguous plain-text superscripts for charges.
- Do not invent structural details that are not supported by the question.
""";

        # The planner supplies request-specific advisory generation guidance.
        # It is never a hard generation ceiling.
        if estimated_answer_tokens:
            final_system += (
                "\n\nInternal answer-size guidance: aim for approximately "
                f"{estimated_answer_tokens} completion tokens. "
                "Treat this as a target, not a hard limit. If more tokens are "
                "mathematically necessary to complete the requested solution, "
                "continue until the solution is complete. If fewer are "
                "sufficient, stop when the solution is complete."
            )

        payload = {
            "problem": problem,
            "idea": idea,
            "reasoning": reasoning,
            "verified_results": tool_results,
        }

        final_llm_started = time.perf_counter()

        response = self.router.generate(
            prompt=json.dumps(
                payload,
                ensure_ascii=False,
            ),
            system_prompt=final_system,
            # answer_tokens is advisory metadata only. Never pass it as a
            # provider max_tokens ceiling; a difficult STEM request may need
            # substantially more output than the planner estimates.
            temperature=0,
            enable_thinking=False,
            request_timeout=None,
        )

        final_llm_ms = round(
            (time.perf_counter() - final_llm_started) * 1000,
            2,
        )

        if not response["success"]:
            result = self._error(response["error"])
            result["_final_llm_ms"] = final_llm_ms
            return result

        answer = str(response["result"] or "").strip()

        if explanation == "off":
            answer = self._sanitize_result_only_answer(answer)

        if not answer:
            result = self._error(
                "The reasoning model returned no solution."
            )
            result["_final_llm_ms"] = final_llm_ms
            return result

        return {
            "success": True,
            "answer": answer,
            "engine": "local_llm+wolfram" if tool_results else "local_llm",
            "tool_calls": tool_results,
            "cached": False,
            "result_only": explanation == "off",
            "_final_llm_ms": final_llm_ms,
        }


    @staticmethod
    def _sanitize_result_only_answer(answer: str) -> str:
        """Remove explanatory prose when result-only generation adds it."""
        text = answer.strip()
        if not text:
            return text

        # Prefer the final fenced/displayed mathematical payload.
        blocks = re.findall(
            r"\`\`\`(?:math|latex)?\s*(.*?)\s*\`\`\`",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if blocks:
            return blocks[-1].strip()

        displays = re.findall(r"\$\$(.*?)\$\$", text, flags=re.DOTALL)
        if displays:
            return displays[-1].strip()

        displays = re.findall(r"\\\[(.*?)\\\]", text, flags=re.DOTALL)
        if displays:
            return displays[-1].strip()

        inline = re.findall(
            r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$",
            text,
            flags=re.DOTALL,
        )
        if inline:
            return "$" + inline[-1].strip() + "$"

        return text

    # =========================================================
    # JSON PARSER
    # =========================================================

    def _parse_json(self, text: str):

        if not isinstance(text, str):
            return None

        text = text.strip()

        try:
            return json.loads(text)

        except json.JSONDecodeError:
            pass

        # Remove markdown fences.
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

        try:
            return json.loads(text)

        except json.JSONDecodeError:
            pass

        # Recover the first JSON object if the model added
        # a tiny amount of surrounding text.
        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end > start:

            candidate = text[
                start:end + 1
            ]

            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        return None

    # =========================================================
    # CACHE
    # =========================================================

    def _cache_get(
        self,
        problem: str,
        explanation: str,
    ):

        key = (
            problem.strip().lower(),
            explanation,
        )

        with self._cache_lock:

            value = self._cache.get(key)

            if value is None:
                return None

            # LRU behavior.
            self._cache.move_to_end(key)

            # Return a copy so callers don't mutate cache.
            return json.loads(
                json.dumps(value)
            )

    def _cache_set(
        self,
        problem: str,
        result: dict,
        explanation: str,
    ):

        key = (
            problem.strip().lower(),
            explanation,
        )

        # Cache only deterministic exact-computation results.
        # Semantic/local-LLM answers must not be cached because they can be
        # malformed and should be regenerated with the current prompt/model.
        if not result.get("success"):
            return
        if result.get("engine") != "wolfram":
            return

        with self._cache_lock:

            self._cache[key] = json.loads(
                json.dumps(result)
            )

            self._cache.move_to_end(key)

            while len(self._cache) > self.CACHE_SIZE:
                self._cache.popitem(
                    last=False
                )
    # =========================================================
    # ERROR
    # =========================================================

    def _error(self, message: str):

        return {
            "success": False,
            "answer": None,
            "error": message,
            "engine": None,
            "tool_calls": [],
            "cached": False,
        }