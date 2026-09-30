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

    # Token budgets are selected adaptively. Small symbolic answers should
    # not reserve the same generation budget as full multi-step solutions.
    PLANNER_MIN_TOKENS = 96
    PLANNER_MAX_TOKENS = 220
    FINAL_MIN_TOKENS = 128
    FINAL_MAX_TOKENS = 12000

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
        # 3. MODEL-DRIVEN REQUEST PLANNING
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
    # COMPLEX COMPUTATIONAL PIPELINE
    # =========================================================

    def _solve_complex(
        self,
        problem: str,
        explanation: bool = True,
    ):

        planner_system = r"""
You are Nova's mathematical request planner.

Analyze the user's request semantically. Do not classify it by matching
specific trigger words. Determine the actual mathematical intent.

Return ONLY JSON:
{
  "type": "plan",
  "request_type": "symbolic_operation | derivation | proof | explanation | reasoning | computation | other",
  "needs_computation": true,
  "idea": "central mathematical idea",
  "reasoning": ["essential mathematical reasoning"],
  "calls": [{"tool": "simplify | factor | expand | differentiate | integrate | solve | limit", "arguments": {}}],
  "generation_tokens": null,
  "final_result": null
}

Rules:
- Choose the request_type semantically, not from keywords.
- Use exact computation only when genuinely useful or required.
- Tool arguments must be valid mathematical expressions, never prose.
- Never invent unspecified values, functions, geometry, conditions, or assumptions.
- For derivations, proofs, explanations and reasoning, put the essential
  mathematical chain in reasoning and leave calls empty unless computation
  is genuinely needed.
- generation_tokens is your estimate of the final solution's required output
  based on mathematical content and requested depth, not a fixed keyword or
  character-length rule.
- For explanation mode "off", put the result in final_result when it can be
  determined without a final explanatory generation.
- Do not write the polished final solution here.
- Maximum 2 tool calls.
- Return valid JSON only.
"""

        planner = self.router.generate(
            prompt=(f"Explanation preference: {explanation}\n\nUser request:\n{problem}"),
            system_prompt=planner_system,
            max_tokens=None,
            temperature=0,
            enable_thinking=False,
        )

        if not planner["success"]:
            return self._error(planner["error"])

        plan = self._parse_json(planner["result"])

        if plan is None:
            return self._error(
                "Local reasoning model returned an invalid plan."
            )

        # -----------------------------------------------------
        # PLANNED REASONING / COMPUTATION
        # -----------------------------------------------------

        if plan.get("type") != "plan":
            return self._error("Invalid reasoning plan type.")

        calls = plan.get("calls", [])
        if not isinstance(calls, list):
            calls = []
        calls = calls[:self.MAX_TOOL_CALLS]

        # -----------------------------------------------------
        # NEVER SEND UNKNOWN FUNCTIONS TO WOLFRAM
        # -----------------------------------------------------

        if self._contains_undefined_function_call(calls):
            calls = []

        # -----------------------------------------------------
        # EXECUTE VERIFIED TOOL CALLS
        # -----------------------------------------------------

        tool_results = []

        for call in calls:

            validated = self._validate_tool_call(call)

            if not validated["valid"]:
                return self._error(
                    validated["error"]
                )

            result = self._execute_tool(
                validated["tool"],
                validated["arguments"],
            )

            # CRITICAL:
            # Never give failed CAS output to Qwen.
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

        if explanation == "off":
            if tool_results:
                return self._cas_result_only(tool_results=tool_results)

            final_result = plan.get("final_result")
            if isinstance(final_result, str) and final_result.strip():
                return {"success": True, "answer": final_result.strip(), "engine": "local_llm", "tool_calls": [], "cached": False}

            return self._error("The planner could not determine a final result without explanation.")

        # -----------------------------------------------------
        # ONE FINAL LLM CALL
        # -----------------------------------------------------

        return self._final_explanation(
            problem=problem,
            idea=plan.get("idea", ""),
            reasoning=plan.get("reasoning", []),
            tool_results=tool_results,
            explanation=explanation,
            estimated_answer_tokens=plan.get("generation_tokens"),
        )

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

            arguments = call.get(
                "arguments",
                {},
            )

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

        if explanation == "concise":
            final_system = r"""
You are Nova's concise mathematical solution writer.
Write ONE self-contained solution. Use very little prose, but preserve all
important mathematical expressions, transformations, substitutions, and
essential rule applications. For derivations and proofs, reach the requested
final result. Never stop at an intermediate expression. Never invent missing
information. Do not mention tools, AI, prompts, or internal processing.
Return ONLY the solution text.
"""
        else:
            final_system = r"""
You are Nova's detailed mathematical solution writer.
Write ONE complete, self-contained mathematical solution from beginning to
end. Explain HOW the result is obtained and show important equations and
transformations. For differentiation, explicitly apply the relevant rules.
For derivations and proofs, continue to the requested final formula, result,
or conclusion. Never stop at an intermediate setup or partial argument.
Never invent missing information or assumptions. Do not mention tools, AI,
prompts, or internal processing. Return ONLY the complete solution text.
"""

        payload = {
            "problem": problem,
            "idea": idea,
            "reasoning": reasoning,
            "verified_results": tool_results,
        }

        response = self.router.generate(
            prompt=json.dumps(
                payload,
                ensure_ascii=False,
            ),
            system_prompt=final_system,
            max_tokens=self._final_token_budget(estimated_answer_tokens),
            temperature=0,
            enable_thinking=False,
        )

        if not response["success"]:
            return self._error(
                response["error"]
            )

        answer = response["result"].strip()

        if not answer:
            return self._error(
                "The reasoning model returned no solution."
            )

        return {
            "success": True,
            "answer": answer,
            "engine": "local_llm+wolfram",
            "tool_calls": tool_results,
            "cached": False,
        }

    # =========================================================
    # DYNAMIC FINAL GENERATION BUDGET
    # =========================================================

    def _final_token_budget(self, estimated_answer_tokens: int | None = None) -> int | None:
        if isinstance(estimated_answer_tokens, int) and estimated_answer_tokens > 0:
            return estimated_answer_tokens
        return None

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



