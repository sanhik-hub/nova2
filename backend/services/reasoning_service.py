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
        # 2. VERY FAST DIRECT CAS PATH
        # -----------------------------------------------------

        if explanation == "off" and self._is_simple_expression(problem):

            result = self.router.calculate(
                query=normalize_expression(problem),
                provider="wolfram",
            )

            if explanation == "off" and result["success"]:
                response = {
                    "success": True,
                    "answer": result.get("result"),
                }
            else:
                response = {
                    "success": result["success"],
                    "answer": result.get("result"),
                    "error": result.get("error"),
                    "engine": "wolfram",
                    "tool_calls": [],
                    "cached": False,
                }

            if result["success"]:
                self._cache_set(
                    problem,
                    response,
                    explanation,
                )

            response["elapsed_ms"] = round(
                (time.perf_counter() - started) * 1000,
                2,
            )

            return response

        # -----------------------------------------------------
        # 3. EXPLICIT SYMBOLIC OPERATION PATH
        # -----------------------------------------------------
        #
        # Natural-language requests such as "Differentiate x^2 + 3x"
        # are not simple expressions, but their requested operation is
        # already unambiguous. Route them directly to the structured
        # Wolfram path. This avoids wasting a planner generation and,
        # with explanation="off", returns the exact CAS result directly.
        # -----------------------------------------------------

        explicit_operation = self._detect_explicit_operation(problem)

        if explicit_operation is not None:
            response = self._solve_direct_operation(
                problem,
                explicit_operation,
                explanation=explanation,
            )

            response["elapsed_ms"] = round(
                (time.perf_counter() - started) * 1000,
                2,
            )

            if response.get("success"):
                self._cache_set(
                    problem,
                    response,
                    explanation,
                )

            return response

        # -----------------------------------------------------
        # 4. MODEL-DRIVEN REQUEST PLANNING
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

    def _normalize_explanation_preference(self, explanation: bool | str) -> str:
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
You are Nova's semantic mathematical request planner.

Analyze the user's request. Do not solve it and do not write the final answer.

Return ONLY one JSON object with EXACTLY these fields:
type, intent, needs_wolfram, wolfram, answer_tokens.

The value of answer_tokens must be a positive integer chosen for this request.

Rules:
- "type" is always "plan".
- "needs_wolfram" is a boolean.
- "wolfram" is null unless exact symbolic computation is genuinely needed.
- If "wolfram" is used, it must be an array containing ONLY these operation
  names: simplify, factor, expand, differentiate, integrate, solve, limit.
- Wolfram arguments must be machine-readable mathematical expressions.
- Never use natural-language instructions as Wolfram arguments.
- Never invent a formula, value, variable, condition, or assumption.
- For derivations, proofs, conceptual explanations, and reasoning, set
  "needs_wolfram" to false and "wolfram" to null unless the user explicitly
  supplies a concrete expression/equation that needs exact symbolic computation.
- A request to "derive", "prove", or "show that" a formula is normally a
  reasoning task, not a Wolfram task. Do not invent a Wolfram call for it.
- Use Wolfram only when a concrete expression/equation from the user's request
  is available and exact symbolic computation or verification is genuinely useful.
- If no Wolfram call is needed, the value MUST be null, not an empty string,
  natural-language instruction, or placeholder.
- If Wolfram is needed, every array entry MUST be an object with exactly
  "operation" and "arguments", where "arguments" is an object.
- Example no-tool plan:
  {"type":"plan","intent":"derive","needs_wolfram":false,"wolfram":null,"answer_tokens":180}
- Example tool plan:
  {"type":"plan","intent":"differentiate","needs_wolfram":true,"wolfram":[{"operation":"differentiate","arguments":{"expression":"x^2+3x","variable":"x"}}],"answer_tokens":60}
- "answer_tokens" is the planner's estimate of the completion tokens actually
  required for the final answer. It is an advisory generation budget, not a
  global preset.
- Estimate the complete final response, not just the final expression or the
  number of mathematical results.
- Count the mathematics the writer must actually emit: required equations,
  substitutions, transformations, derivation/proof steps, and the concise prose
  needed to make those steps understandable.
- For a derivation or proof, the estimate must cover the complete mathematical
  chain, not merely the final formula and a one-sentence explanation.
- The estimate must be a realistic sufficient budget for the whole answer.
  Do not output a trivially small estimate when the request requires a
  derivation, proof, or worked explanation.
- Use the smallest sufficient estimate for this specific request. Do not
  inflate it merely because the topic is difficult.
- If the request is genuinely short, the estimate should also be short.
- If the request needs a substantial derivation, proof, or multi-part solution,
  the estimate should grow accordingly, without relying on a fixed table of
  budgets.
- The requested explanation mode is supplied separately by the caller; use it
  when estimating answer_tokens.

The planner is a router, not a solver. When information needed for a formula is
missing, leave Wolfram unused rather than guessing it.
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

        if not isinstance(plan, dict) or plan.get("type") != "plan":
            return self._error("Planner returned an invalid plan.")

        required_fields = {
            "intent": str,
            "needs_wolfram": bool,
            "wolfram": (list, type(None)),
        }

        for field, expected_type in required_fields.items():
            if field not in plan or not isinstance(plan[field], expected_type):
                return self._error(
                    f"Planner returned an invalid plan field: {field}."
                )

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

        calls = plan.get("wolfram") or []
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
You are Nova's answer writer.
Return ONLY the direct answer to the user's mathematical request.
Do not add explanation, derivation, commentary, or internal processing details.
Do not invent missing information or assumptions.
"""
        elif explanation == "concise":
            final_system = r"""
You are Nova's concise mathematical solution writer.

Write ONE compact but complete worked solution. "Concise" means remove
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
You are Nova's detailed mathematical solution writer.
Write ONE complete, self-contained mathematical solution from beginning to end.
Explain HOW the result is obtained and show the important equations,
transformations, and rule applications.

For differentiation, explicitly apply the relevant rules. For derivations and
proofs, continue through every necessary step to the requested final formula,
result, or conclusion.

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

        # The planner's estimate is guidance, not a hard generation cutoff.
        # Passing it as an internal target lets Qwen stay concise without
        # allowing a bad estimate to truncate a mathematically incomplete
        # answer. llama.cpp is allowed to terminate naturally at EOS.
        if type(estimated_answer_tokens) is int and estimated_answer_tokens > 0:
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
            max_tokens=None,
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
            "engine": "local_llm+wolfram",
            "tool_calls": tool_results,
            "cached": False,
            "_final_llm_ms": final_llm_ms,
        }

    # =========================================================
    # FINAL GENERATION BUDGET
    # =========================================================

    # The planner estimate is intentionally advisory. Final generation uses
    # natural EOS termination rather than converting the estimate into a hard
    # max_tokens cutoff.

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



