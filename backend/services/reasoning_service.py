import json
import re
import time
from collections import OrderedDict
from threading import Lock
from typing import Any

from backend.services.provider_router import ProviderRouter
from backend.services.expression_normalizer import normalize_expression
class ReasoningService:

    # =========================================================
    # PERFORMANCE CONFIG
    # =========================================================

    MAX_TOOL_CALLS = 2

    # The planner's answer_tokens is the per-request final-generation budget.
    # There is deliberately no hardcoded final-answer token ceiling here.
    PLANNER_TOKENS = 220

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
        # 1. FAST EXACT OPERATION PATH
        # -----------------------------------------------------
        #
        # Result-only mode should not spend planner/final-writer time on an
        # unambiguous exact symbolic operation. This uses the same validated
        # Wolfram execution path as the normal solver; only the routing step
        # is skipped because the user's requested operation is explicit.
        # -----------------------------------------------------
        if explanation == "off":
            direct_operation = self._detect_explicit_operation(problem)
            if direct_operation is not None:
                response = self._solve_direct_operation(
                    problem=problem,
                    operation=direct_operation,
                    explanation=explanation,
                )
                response["elapsed_ms"] = round(
                    (time.perf_counter() - started) * 1000,
                    2,
                )
                return response

        # -----------------------------------------------------
        # 2. MODEL-DRIVEN SEMANTIC PLANNING
        # -----------------------------------------------------
        #
        # Every non-cached request goes through the small planner. This is
        # deliberate: Standard Solver is a general STEM solver, so routing
        # must be based on the meaning of the request rather than a growing
        # collection of operation/domain keywords.
        # -----------------------------------------------------

        response = self._solve_complex(
            problem,
            explanation=explanation,
        )

        response["elapsed_ms"] = round(
            (time.perf_counter() - started) * 1000,
            2,
        )

        return response

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
        Return True only for inputs that are already mathematical
        expressions suitable for the direct Wolfram calculate() path.

        Explicit natural-language operations such as "Differentiate x^2"
        are intentionally excluded. Those requests must go through the
        planner so the planner can emit a structured Wolfram operation
        instead of sending natural language to Wolfram.
        """

        text = problem.strip()

        if len(text) > 120:
            return False

        lowered = text.lower()

        computational_markers = (
            "integrate ",
            "integral ",
            "differentiate ",
            "derivative ",
            "derive ",
            "solve ",
            "factor ",
            "factorize ",
            "expand ",
            "simplify ",
            "limit ",
            "find the derivative",
            "find derivative",
            "find the integral",
            "find integral",
            "find the roots",
            "find roots",
            "find the solutions",
            "find solutions",
            "evaluate the limit",
            "evaluate limit",
        )

        if any(marker in lowered for marker in computational_markers):
            return False

        semantic_phrases = (
            "prove",
            "show that",
            "show",
            "explain",
            "determine whether",
            "determine if",
            "why",
            "suppose",
            "given",
            "continuous",
            "function",
            "sequence",
            "probability",
            "geometry",
            "physics",
            "chemistry",
            "maximum",
            "minimum",
            "area",
            "volume",
            "find the",
            "let ",
            "such that",
            "if ",
        )

        if any(phrase in lowered for phrase in semantic_phrases):
            return False

        words = re.findall(r"[A-Za-z]+", text)

        if len(words) > 8:
            return False

        if not re.search(
            r"[0-9xXyYzZπ√+\-*/^=<>()[\]{}]",
            text,
        ):
            return False

        return True

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
- A chemistry question is NOT a Wolfram request merely because it mentions a chemical compound, reaction, element, or formula. For example, a request for the formula of nitrobenzene should use needs_wolfram=false.
- A physics question is NOT a Wolfram request merely because it is physics. Conceptual questions such as Newton's laws, SI units, or qualitative explanations should use needs_wolfram=false.
- Use Wolfram for physics or chemistry only when the user's concrete problem requires exact symbolic/numeric computation, algebra, equation solving, calculus, or another supported exact operation and the required mathematical data are present.
- Derivations and proofs normally use needs_wolfram=false; use Wolfram only when an explicit concrete expression/equation requires exact symbolic computation.
- Simple arithmetic or a bare mathematical expression may use Wolfram when exact evaluation is the actual task.
- "operation" is null when exact symbolic/numeric computation is not genuinely needed.
- "arguments" is null when "operation" is null.
- If exact computation is needed, "operation" must be exactly one of:
  simplify, factor, expand, differentiate, integrate, solve, limit.
- "arguments" must be a machine-readable object appropriate to that operation.
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
  - "What is Newton's second law?" is conceptual:
    {"type":"plan","intent":"law","needs_wolfram":false,"operation":null,"arguments":null,"answer_tokens":80}
  - "Formula of nitrobenzene" is factual chemistry:
    {"type":"plan","intent":"formula","needs_wolfram":false,"operation":null,"arguments":null,"answer_tokens":40}
- Do not classify an explicit request to differentiate, integrate, factor, expand, simplify, solve, or take a limit as pure reasoning when the user supplied the concrete expression/equation needed for that exact operation.
- "answer_tokens" is the planner's estimate of the completion tokens actually
  required for the final answer. It is an advisory generation budget.
- For explanation "off", estimate ONLY the required result that should be shown
  to the user. Do not budget for explanations, introductions, commentary,
  headings, derivations, or summaries. An overall-reaction request should budget
  for the reaction equation; a formula request for the formula; a name request
  for the name; a value request for the value.
- For "concise" and "detailed", estimate the complete response including the
  mathematics and prose required by that explanation mode.
- Always use the smallest sufficient estimate for this specific request.
- Never inflate the estimate merely because the topic is difficult.

The planner is a router, not a solver. When information needed for a formula or calculation is missing, leave Wolfram unused rather than guessing it.

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
            max_tokens=self.PLANNER_TOKENS,
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
            return self._error("Planner returned an invalid plan.")

        required_fields = {
            "intent": str,
            "needs_wolfram": bool,
            "operation": (str, type(None)),
            "arguments": (dict, type(None)),
        }

        for field, expected_type in required_fields.items():
            if field not in plan or not isinstance(plan[field], expected_type):
                return self._error(
                    f"Planner returned an invalid plan field: {field}."
                )

        # A previous planner prompt used a nested "wolfram" field. Accept that
        # legacy shape only as a narrow recovery path so a stale/partially
        # updated local model cannot turn a valid request into a hard failure.
        # The new contract remains operation + arguments.
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
            if not isinstance(operation, str) or not operation.strip():
                return self._error(
                    "Planner requested exact computation but did not provide "
                    "a valid operation."
                )
            if not isinstance(arguments, dict):
                return self._error(
                    "Planner requested exact computation but did not provide "
                    "structured arguments."
                )
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

    def _detect_explicit_operation(self, problem: str):
        """Detect an unambiguous symbolic operation and build safe arguments."""

        text = problem.strip()

        # DIFFERENTIATE
        match = re.match(
            r"(?is)^\s*(?:differentiate\s+|find\s+(?:the\s+)?derivative\s+(?:of\s+)?)(.+?)\s*[.!?]?\s*$",
            text,
        )
        if match:
            expression = match.group(1).strip()
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
            r"(?is)^\s*(?:limit|evaluate\s+(?:the\s+)?limit)\s+(.+?)\s*[.!?]?\s*$",
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

    def _solve_direct_operation(
        self,
        problem: str,
        operation,
        explanation: str = "detailed",
    ):
        """Execute one validated symbolic operation with a planner-derived final budget."""

        tool, arguments = operation

        validation = self._validate_tool_call({
            "tool": tool,
            "arguments": arguments,
        })

        if not validation["valid"]:
            return self._error(validation["error"])

        tool_started = time.perf_counter()

        result = self._execute_tool(
            validation["tool"],
            validation["arguments"],
        )

        wolfram_ms = round(
            (time.perf_counter() - tool_started) * 1000,
            2,
        )

        if not result.get("success"):
            return self._error(
                "Wolfram verification failed: "
                + str(result.get("error"))
            )

        tool_result = {
            "tool": validation["tool"],
            "arguments": validation["arguments"],
            "result": result.get("result"),
        }

        if explanation == "off":
            response = self._cas_result_only(
                tool_results=[tool_result],
            )
            response["timings_ms"] = {
                "planner": 0.0,
                "wolfram": wolfram_ms,
                "final_llm": 0.0,
            }
            return response

        budget = self._plan_answer_tokens(
            problem=problem,
            explanation=explanation,
        )

        if not budget["success"]:
            return self._error(budget["error"])

        response = self._final_explanation(
            problem=problem,
            idea="",
            reasoning=[],
            tool_results=[tool_result],
            explanation=explanation,
            estimated_answer_tokens=budget["answer_tokens"],
        )
        response["timings_ms"] = {
            "planner": budget["planner_ms"],
            "wolfram": wolfram_ms,
            "final_llm": response.pop("_final_llm_ms", 0.0),
        }
        return response

    # =========================================================
    # PLANNER-DRIVEN FINAL ANSWER BUDGET
    # =========================================================

    def _plan_answer_tokens(
        self,
        problem: str,
        explanation: str,
    ) -> dict[str, Any]:
        """Ask the small planner for a semantic final-answer budget."""

        planner_system = r"""
You are Nova's final-answer budget planner.

Do not solve the problem and do not write any solution text.
Estimate only how many completion tokens the final mathematical answer
actually needs.

Return ONLY one JSON object with EXACTLY these fields:
type, answer_tokens.

The value of answer_tokens must be a positive integer chosen for this request.

Rules:
- "type" is always "answer_budget".
- "answer_tokens" must be a realistic positive integer.
- Estimate the complete final response, not just its final expression.
- Account for every equation, substitution, transformation, derivation/proof
  step, and the amount of prose required by the explanation mode.
- The estimate must be sufficient for the final writer to finish the requested
  mathematics without truncation, while remaining as small as reasonably
  possible for this specific request.
- Do not use a fixed budget based only on the explanation label.
- Do not output a trivially small estimate for a request that requires a
  derivation, proof, or worked explanation.
- Do not inflate the estimate merely because the problem is difficult.
- This is a generation budget estimate, not a claim about exact output length.
"""

        started = time.perf_counter()

        result = self.router.plan(
            prompt=json.dumps(
                {
                    "explanation_preference": explanation,
                    "user_request": problem,
                },
                ensure_ascii=False,
            ),
            system_prompt=planner_system,
        )

        elapsed = round((time.perf_counter() - started) * 1000, 2)

        if not result["success"]:
            return {
                "success": False,
                "error": result["error"],
                "planner_ms": elapsed,
            }

        plan = self._parse_json(result["result"])

        if not isinstance(plan, dict) or plan.get("type") != "answer_budget":
            return {
                "success": True,
                "answer_tokens": None,
                "planner_ms": elapsed,
            }

        answer_tokens = plan.get("answer_tokens")

        if isinstance(answer_tokens, str) and re.fullmatch(r"\d+", answer_tokens.strip()):
            answer_tokens = int(answer_tokens.strip())

        if type(answer_tokens) is not int or answer_tokens <= 0:
            # The estimate is advisory. Preserve the solved request and let
            # llama.cpp stop naturally when the model reaches EOS.
            return {
                "success": True,
                "answer_tokens": None,
                "planner_ms": elapsed,
            }

        return {
            "success": True,
            "answer_tokens": answer_tokens,
            "planner_ms": elapsed,
        }

    # =========================================================
    # TOOL VALIDATION
    # =========================================================

    def _validate_tool_call(self, call: dict):

        if not isinstance(call, dict):
            return {
                "valid": False,
                "error": "Tool call is not an object.",
            }

        tool = call.get("tool")
        arguments = call.get("arguments")

        allowed = {
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
                "valid": False,
                "error": f"Unsupported tool: {tool}",
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

            if tool == "simplify":

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
        idea: str,
        reasoning: list,
        tool_results: list,
        explanation: str = "detailed",
        estimated_answer_tokens: int | None = None,
    ):

        if explanation == "off":
            final_system = r"""
Return ONLY the requested result. Do the minimum internal reasoning needed for
accuracy, but expose no reasoning.

No explanation, lead-in, heading, summary, definition, justification,
background, alternatives, or commentary. Do not write "The answer is" or
similar text.

Output rules:
- overall reaction/equation -> reaction/equation only
- formula -> formula only
- value/expression -> value/expression only
- name -> name only
- multiple explicitly requested results -> only those results

For chemistry, first internally identify exactly what the user requested
(reaction, product, formula, structure, name, oxidation state, or condition).
For an overall-reaction request, reproduce the requested transformation itself;
do not substitute a related reaction, reagent, product, or mechanism. Check
reactants, reagents/conditions, stoichiometry, regiochemistry, and product
identity internally before emitting the result.
For physics and mathematics, check the governing relationship and any required
substitution internally before emitting the requested result.
Never invent missing data or assumptions.

Emit only the final requested result in standard mathematical/scientific
notation.
"""
        elif explanation == "concise":
            final_system = r"""
You are Nova's concise STEM solution writer.

Write ONE compact but complete worked solution. The request may involve mathematics, physics, chemistry, or general science. "Concise" means remove
unnecessary prose, not the mathematical work required to justify the answer.

For a calculation, show the essential setup and transformations leading to the
result. For a derivation or proof, ALWAYS show the mathematical chain from the
starting principle to the requested formula/result, with the key equation at
each necessary step and a brief explanation of what each step does. Do not
replace a derivation with a sentence saying that it follows from integration,
differentiation, substitution, or another operation.

Use short explanatory sentences around the mathematics. Preserve necessary
definitions, substitutions, limits/integrals, algebraic transformations, and
the final result. Omit optional background, repeated explanations, alternative
methods, long summaries, and filler.

The answer must be complete enough that a student can follow HOW the result was
obtained without needing an omitted derivation. Never stop at an intermediate
expression. Never invent missing information.

For a derivation or proof, do NOT state or preview the requested final formula,
result, or conclusion before the derivation is complete. Build the result
step by step using the necessary intermediate equations. State the requested
final formula, result, or conclusion exactly ONCE, at the end of the solution.
Do not repeat it in an introduction, summary, or second final-answer section.

Do not mention tools, AI, prompts, token budgets, or internal processing.
Return ONLY the solution text.
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

        # The planner supplies a request-specific advisory generation budget.
        # It is passed through to the model when valid; there is deliberately
        # no service-level fixed final-answer token cap.
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
            # Result-only mode must not spend completion tokens on prose.
            # Use the planner's request-specific result budget only in this
            # mode; concise/detailed remain uncapped at the service level.
            # In result-only mode the planner's estimate is an actual
            # completion budget: there is no reason to let the model continue
            # generating prose after the requested result is complete.
            max_tokens=estimated_answer_tokens if explanation == "off" else None,
            temperature=0,
            enable_thinking=False,
        )

        final_llm_ms = round(
            (time.perf_counter() - final_llm_started) * 1000,
            2,
        )

        if not response["success"]:
            result = self._error(response["error"])
            result["_final_llm_ms"] = final_llm_ms
            return result

        answer = response["result"].strip()

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

        # Don't cache failures.
        if not result.get("success"):
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


