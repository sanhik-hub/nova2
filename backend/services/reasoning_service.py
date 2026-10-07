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
        # 4. DIRECT QUANTITATIVE STEM HANDOFF
        #
        # Numerical STEM word problems are solved directly by the local
        # reasoning model. Domain alone never selects a CAS provider.
        # This keeps physics/chemistry calculations with the model that
        # understands the full semantic problem and its units.
        if self._is_direct_quantitative_request(problem):
            response = self._solve_quantitative_direct(problem, explanation)
            response["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
            return response

        # 5. DIRECT SEMANTIC FAST PATH
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
- For a quantitative problem that supplies concrete numerical values and explicitly asks for a calculated numerical result, exact arithmetic is genuinely required. Prefer a supported exact calculation operation for the arithmetic subcalculation while the local reasoning model remains responsible for the scientific interpretation and final solution. Do not route such a request as a formula-only or factual response.
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
        """Solve a numerical STEM word problem directly with the local model.

        This path deliberately does not invoke Wolfram. The model is given the
        complete problem so it can preserve units, prefixes, powers, signs,
        and the semantic relationship between quantities.
        """
        if explanation == "off":
            style = """
Return ONLY the final requested result. Do not include a heading, explanation,
derivation, or sentence such as "the answer is". Include the unit when the
requested result has a unit.
"""
        elif explanation == "concise":
            final_system = r"""
You are Nova's concise STEM solution writer.

Produce a compact, genuinely worked solution to the actual user problem.
This is the short version of a full solution: show the necessary
intermediate transformations, but remove teaching filler and background.

This is domain-general. Handle mathematics, physics, chemistry, and general
science according to the actual problem. Never use a subject-specific
template, hardcoded formula sequence, or fixed number of steps.

Required format:
- Begin immediately with the first necessary equation, relation,
  transformation, or factual inference. No title, introduction, "Solution",
  "Answer", or paragraph that merely restates the problem.
- Make the solution visually open and easy to scan. Use generous blank lines
  between distinct reasoning/equation groups.
- Put each important standalone equation in a display-math block using
  $$...$$. The frontend renders these blocks as large, centered,
  professionally typeset equations.
- Keep explanatory prose outside math blocks. Use a short sentence immediately
  before an equation group when it explains what the next equation represents
  or why the next transformation is being made.
- Use $...$ for short inline mathematical expressions inside prose.
- For continuous algebraic transformations, use a display block with aligned
  or gathered equations when appropriate.
- Use standard LaTeX mathematical notation inside the delimiters. Preserve
  meaningful subscripts, superscripts, fractions, roots, integrals, matrices,
  Greek symbols, units, charges, bonds, and reaction arrows.
- Never put equations inside backticks or code blocks.
- For chemistry, keep reaction and structure equations in display math when
  they benefit from clear visual separation, and preserve scientifically
  meaningful charges and bonds.
- Do not restate the question.
- Do not add background, alternative methods, generic explanations, a recap,
  or a second final-answer section.
- Do not stop after merely stating the governing formula.
- Do not replace necessary algebra, substitution, unit conversion, or logical
  transformation with phrases such as "using the formula" or "simplifying".
- End immediately after the requested result has been established.
- There is NO hard output/token limit. Never truncate a complete solution or
  remove necessary work just to make it shorter.

Example of the intended visual structure:

The linear velocity of a segment at distance $r$ from the axis is

$$
v = r\omega
$$

The emf across a small segment is

$$
dE = Bv\,dr
$$

Substituting $v=r\omega$:

$$
dE = B(r\omega)\,dr
$$

Integrating along the rod:

$$
E = \int_0^L B\omega r\,dr
$$

Therefore,

$$
\boxed{E=\frac{1}{2}B\omega L^2}
$$

This example defines formatting and solution density only. Do not copy its
subject matter into other problems. For every concise solution, the first
visible content must be the first necessary equation, relation,
transformation, or factual inference.

Return ONLY the complete solution text. No metadata or commentary.
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

                # Concise mode already defines its own plain-text display contract.
        # Keep the structured scientific display contract for other modes.
        if explanation != "concise":
            final_system += r"""
Scientific display contract:
- Put mathematical expressions in $...$ and larger derivations in $...$.
- Put chemical formulas, structures, and reaction equations in $...$ so the
  frontend can typeset element subscripts, explicit bonds, charges, and lone
  pairs unambiguously.
- For chemical structures use upright element symbols with explicit bond
  characters: - for single, = for double, \\equiv for triple, and
  \\rightarrow or \\rightleftharpoons for reaction arrows.
- Show formal charges explicitly, e.g. $\\mathrm{NH_4^+}$.
- When lone pairs are important, show them explicitly with LaTeX such as
  $\\overset{\\cdot\\,\\cdot}{\\mathrm{O}}$ rather than relying on
  prose or invisible electron pairs.
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