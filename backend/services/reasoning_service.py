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
                # A deterministic parser is only an optimization. If Wolfram
                # rejects its normalized arguments, hand the original request
                # to the main local reasoning model instead of exposing the CAS
                # error directly to the user. The model can reformulate the
                # request and the Wolfram result is still verified afterward.
                response = self._solve_llm_wolfram(
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

        direct_expression = (
            problem
            if self._is_simple_expression(problem)
            else self._extract_numeric_calculation(problem)
        )

        if direct_expression is not None:
            calculation = self.router.calculate(
                expression=normalize_expression(direct_expression),
                provider="wolfram",
            )

            if not calculation.get("success"):
                # Bare-expression evaluation is also only a fast path. Let the
                # main reasoning model reformulate anything Wolfram cannot
                # evaluate directly, rather than surfacing the raw CAS error.
                response = self._solve_llm_wolfram(
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
        # 4. LLM + EXACT CAS FOR NATURAL-LANGUAGE COMPUTATION
        # -----------------------------------------------------
        # Numerical STEM word problems require semantic formulation before
        # exact computation. Let the local reasoning model translate the
        # request into a self-contained Wolfram expression, then let Wolfram
        # perform the actual calculation. This is deliberately separate from
        # the lightweight planner used for general routing.
        # -----------------------------------------------------
        if self._is_complex_computational_request(problem):
            response = self._solve_llm_wolfram(
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

        # -----------------------------------------------------
        # 5. DIRECT DERIVATION / PROOF FAST PATH
        # -----------------------------------------------------
        # Equation-backed derivations and proofs do not need the planner to
        # decide whether exact tool execution is useful. They are already
        # explicit reasoning tasks, so send them directly to the final
        # reasoning model. Classification is generic and domain-agnostic.
        if self._is_direct_derivation_request(problem):
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
        # 6. DIRECT SEMANTIC FAST PATH
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
        allowed_identifiers = (
            set(FUNCTION_MAP)
            | set(CONSTANT_MAP)
            | {
                "N", "Sqrt", "Sin", "Cos", "Tan", "Solve", "With",
                "Exp", "Log", "Abs", "Min", "Max",
            }
        )
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

    def _has_explicit_operation_intent(self, problem: str) -> bool:
        """Detect a supported operation request that needs model normalization.

        This is intentionally structural. It does not inspect subject/domain
        vocabulary and does not normalize punctuation or connective phrases.
        Clean forms are handled by the deterministic fast path; malformed or
        natural-language forms go to the main local reasoning model.
        """
        text = problem.strip()
        if not text or len(text) > 1200:
            return False
        return bool(
            re.match(
                r"(?is)^\s*(?:differentiate|integrate|factor(?:ize)?|expand|simplify|solve|limit)\b",
                text,
            )
        )

    def _is_complex_computational_request(self, problem: str) -> bool:
        """
        Detect a natural-language request that contains an explicit
        computational payload without depending on a subject/domain word.
        """
        text = problem.strip()
        if not text or len(text) > 1200:
            return False
        if self._is_simple_expression(text):
            return False
        if self._detect_explicit_operation(text) is not None:
            return False

        # An explicit computational operation whose deterministic parser could
        # not safely normalize is handed to the main local LLM. The model
        # returns a structured operation/arguments object; Python validates it
        # before anything reaches Wolfram.
        if self._has_explicit_operation_intent(text):
            return True

        if not re.search(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text):
            return False

        computational_intent = re.search(
            r"\b(?:calculate|compute|determine|evaluate|find|obtain|work\s+out|how\s+much|how\s+many|what\s+is|what\s+are)\b",
            text,
            flags=re.IGNORECASE,
        )
        return bool(computational_intent)

    def _is_direct_quantitative_request(self, problem: str) -> bool:
        text = problem.strip()
        if not text or len(text) > 500 or not re.search(r"\d", text):
            return False
        if self._is_simple_expression(text) or self._detect_explicit_operation(text) is not None:
            return False
        return len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", text)) >= 3

    def _is_direct_derivation_request(self, problem: str) -> bool:
        """Identify explicit derivation/proof tasks with symbolic structure."""
        text = problem.strip()
        if not text or len(text) > 700:
            return False
        if self._is_simple_expression(text) or self._detect_explicit_operation(text) is not None:
            return False
        intent = re.search(
            r"\b(?:derive|derivation|prove|proof|demonstrate|demonstration|deduce|deduction|show\s+(?:that|how)|obtain)\b",
            text,
            flags=re.IGNORECASE,
        )
        if not intent:
            return False
        has_symbolic_structure = bool(
            re.search(
                r"(?:[A-Za-z]\s*[A-Za-z]?\s*=|[A-Za-z0-9)]\s*[+\-*/^]\s*[A-Za-z0-9(]|[∫∑√∞∂Δωπ]|\b(?:d[A-Za-z]+|[A-Za-z]+\s*\([^)]*\))\b)",
                text,
            )
        )
        if not has_symbolic_structure:
            return False
        return len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", text)) >= 3

    # =========================================================
    # LLM + EXACT CAS COMPUTATIONAL PIPELINE
    # =========================================================

    def _solve_llm_wolfram(self, problem: str, explanation: str = "detailed"):
        """
        Use the main local reasoning model to translate a computational request
        into a validated structured Wolfram tool call.

        The model may return either a supported operation plus structured
        arguments or a calculate operation containing a self-contained Wolfram
        expression. Python owns validation and execution; Wolfram remains the
        source of truth for computed results.
        """
        formulation_system = r"""
You are Nova's STEM computational formulation engine.

The user supplied a computational STEM request. Your ONLY job is to translate
the request into one structured tool call for Wolfram. Do not solve it in
prose and do not return numerical answers yourself.

First reason internally about:
- every quantity explicitly given;
- every quantity explicitly requested;
- a concise, distinct machine-readable label for each requested output;
- the governing mathematical/scientific relationships;
- units and necessary conversions;
- initial conditions, geometry, constraints, and signs;
- whether the request is a direct symbolic operation or a natural-language
  quantitative problem.

Return ONLY one JSON object in exactly this form:
{"operation":"...","arguments":{...},"requested_outputs":["outputLabel1","outputLabel2"]}

requested_outputs is required. Include one short lowerCamelCase label for EVERY distinct output the user asks for, in the same order as the request. Do not include intermediate values unless the user requests them. For a single-output request, include exactly one label.

Supported operations and argument shapes:
- {"operation":"calculate","arguments":{"expression":"..."}}
- {"operation":"simplify","arguments":{"expression":"..."}}
- {"operation":"factor","arguments":{"expression":"..."}}
- {"operation":"expand","arguments":{"expression":"..."}}
- {"operation":"differentiate","arguments":{"expression":"...","variable":"x"}}
- {"operation":"integrate","arguments":{"expression":"...","variable":"x"}}
- {"operation":"integrate_definite","arguments":{"expression":"...","variable":"x","lower":"...","upper":"..."}}
- {"operation":"solve","arguments":{"equation":"...","variable":"x"}}
- {"operation":"limit","arguments":{"expression":"...","variable":"x","value":"..."}}

Choose the operation that actually matches the user's request. For a
natural-language quantitative STEM problem, use calculate and construct ONE
self-contained Wolfram Language expression that computes EVERY explicitly
requested result.

For multiple requested results, a calculate expression may return a list of
immediate rules, for example:
{timeOfFlight -> N[...], maximumHeight -> N[...], horizontalRange -> N[...], impactSpeed -> N[...]}

Use simple lowerCamelCase result labels. NEVER use snake_case or underscores
in Wolfram symbols: underscore is Wolfram's pattern syntax.

Completeness invariant: before emitting the JSON, make a private checklist of
EVERY quantity, value, equation, product, condition, or other output explicitly
requested by the user. A calculate call is invalid if it computes only one part
of a multi-part request. For multiple requested outputs, return one expression
that constructs a result for every requested output. Do not stop at an
intermediate Solve, equation, derivative, or other sub-result merely because
it answers one part. Reuse intermediate values with With, replacement rules,
or local symbols when needed.

For direct operations, translate the user's actual expression/equation into
valid Wolfram syntax. For example, a request such as
"solve : x^2-5x+4 = 0" should become:
{"operation":"solve","arguments":{"equation":"x^2 - 5*x + 4 == 0","variable":"x"}}

Do not invent data, formulas, assumptions, or unstated conditions. Preserve
the user's variables, quantities, units, constraints, initial conditions, and
requested outputs.

The arguments must contain only executable Wolfram content, never natural-
language instructions. Do not use strings inside Wolfram expressions,
external functions, files, URLs, or programming side effects.

Return no markdown, prose, or code fences outside the JSON object.
"""
        formulation_prompt = json.dumps(
            {
                "user_request": problem,
                "explanation_preference": explanation,
            },
            ensure_ascii=False,
        )

        # Prefer llama.cpp native tool calls when the running server exposes
        # a tool-aware Jinja template. Otherwise use llama.cpp's constrained
        # JSON output path. Both transports feed the same Python validator.
        formulation_schema = {
            "type": "object",
            "properties": {
                "requested_outputs": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "operation": {
                    "type": "string",
                    "enum": [
                        "calculate",
                        "simplify",
                        "factor",
                        "expand",
                        "differentiate",
                        "integrate",
                        "integrate_definite",
                        "solve",
                        "limit",
                    ],
                },
                "arguments": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"},
                        "equation": {"type": "string"},
                        "variable": {"type": "string"},
                        "lower": {"type": "string"},
                        "upper": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            },
            "required": ["operation", "arguments", "requested_outputs"],
            "additionalProperties": False,
        }

        wolfram_tool = {
            "type": "function",
            "function": {
                "name": "wolfram",
                "description": (
                    "Translate the user's computational STEM request into one "
                    "validated Wolfram operation. Preserve every explicitly "
                    "requested result."
                ),
                "parameters": formulation_schema,
            },
        }

        use_native_tools = self.router.supports_tool_calls()
        structured_kwargs = (
            {
                "tools": [wolfram_tool],
                "tool_choice": "required",
            }
            if use_native_tools
            else {
                "response_format": {
                    "type": "json_object",
                    "schema": formulation_schema,
                },
            }
        )

        llm_started = time.perf_counter()
        try:
            formulation = self.router.generate(
                prompt=formulation_prompt,
                system_prompt=formulation_system,
                temperature=0.0,
                request_timeout=None,
                **structured_kwargs,
            )
        except Exception as exc:
            return self._error(f"Computational formulation failed: {exc}")

        formulation_ms = round(
            (time.perf_counter() - llm_started) * 1000,
            2,
        )

        if not formulation.get("success"):
            return self._error(
                formulation.get("error") or "Computational formulation failed."
            )

        formulation_text = formulation.get("result")
        plan = self._parse_json(formulation_text)

        if not isinstance(plan, dict):
            return self._error(
                "The reasoning model did not produce a valid structured computational request."
            )

        operation = plan.get("operation")
        arguments = plan.get("arguments")
        requested_outputs = plan.get("requested_outputs")

        if (
            not isinstance(requested_outputs, list)
            or not requested_outputs
            or any(not isinstance(item, str) or not item.strip() for item in requested_outputs)
        ):
            requested_outputs = []

        # Backward-compatible recovery for a model that still emits the older
        # {"expression":"..."} contract. New generations are explicitly
        # instructed to use operation + arguments.
        if operation is None and isinstance(plan.get("expression"), str):
            operation = "calculate"
            arguments = {"expression": plan["expression"]}

        if not isinstance(operation, str) or not isinstance(arguments, dict):
            return self._error(
                "The reasoning model did not produce a valid structured computational request."
            )

        validated = self._validate_tool_call(
            {
                "tool": operation,
                "arguments": arguments,
            }
        )
        if not validated.get("valid"):
            return self._error(
                validated.get("error")
                or "The structured computational request was rejected."
            )

        operation = validated["tool"]
        arguments = validated["arguments"]

        wolfram_started = time.perf_counter()
        result = self._execute_tool(operation, arguments)
        wolfram_ms = round(
            (time.perf_counter() - wolfram_started) * 1000,
            2,
        )

        if not result.get("success"):
            # Repair the structured tool call using the actual Wolfram error.
            # This remains a main-LLM formulation pass; it never asks the model
            # to provide an unverified numerical answer.
            repair_system = r"""
You are repairing a structured Wolfram tool call generated from a STEM
problem.

Return ONLY one JSON object:
{"operation":"...","arguments":{...}}

Use only these operations:
calculate, simplify, factor, expand, differentiate, integrate,
integrate_definite, solve, limit.

Preserve the original problem and all requested quantities. Correct the
operation/arguments so Wolfram can execute the request. Do not answer the
problem yourself.

For calculate, the expression must be one executable Wolfram Language
expression. For solve, return an executable equation and variable. For limit,
return expression, variable, and value. Do not use natural-language
instructions, strings, external functions, files, URLs, or programming side
effects. Never use underscores in Wolfram symbols.
"""
            repair_prompt = json.dumps(
                {
                    "user_request": problem,
                    "previous_tool_call": {
                        "operation": operation,
                        "arguments": arguments,
                    },
                    "wolfram_error": str(result.get("error") or "unknown error"),
                },
                ensure_ascii=False,
            )
            repair_started = time.perf_counter()
            try:
                repaired = self.router.generate(
                    prompt=repair_prompt,
                    system_prompt=repair_system,
                    temperature=0.0,
                    request_timeout=None,
                    **structured_kwargs,
                )
            except Exception as exc:
                repaired = {"success": False, "error": str(exc)}
            formulation_ms += round(
                (time.perf_counter() - repair_started) * 1000,
                2,
            )

            repaired_plan = (
                self._parse_json(repaired.get("result"))
                if repaired.get("success")
                else None
            )
            repaired_operation = (
                repaired_plan.get("operation")
                if isinstance(repaired_plan, dict)
                else None
            )
            repaired_arguments = (
                repaired_plan.get("arguments")
                if isinstance(repaired_plan, dict)
                else None
            )

            if (
                not isinstance(repaired_operation, str)
                or not isinstance(repaired_arguments, dict)
            ):
                # Also accept the legacy expression shape during repair.
                if (
                    isinstance(repaired_plan, dict)
                    and isinstance(repaired_plan.get("expression"), str)
                ):
                    repaired_operation = "calculate"
                    repaired_arguments = {
                        "expression": repaired_plan["expression"]
                    }
                else:
                    return self._error(
                        "Wolfram verification failed: "
                        + str(result.get("error") or "unknown error")
                    )

            repair_validation = self._validate_tool_call(
                {
                    "tool": repaired_operation,
                    "arguments": repaired_arguments,
                }
            )
            if not repair_validation.get("valid"):
                return self._error(
                    repair_validation.get("error")
                    or "The repaired computational request was rejected."
                )

            repaired_operation = repair_validation["tool"]
            repaired_arguments = repair_validation["arguments"]

            wolfram_retry_started = time.perf_counter()
            result = self._execute_tool(
                repaired_operation,
                repaired_arguments,
            )
            wolfram_ms += round(
                (time.perf_counter() - wolfram_retry_started) * 1000,
                2,
            )
            if not result.get("success"):
                return self._error(
                    "Wolfram verification failed: "
                    + str(result.get("error") or "unknown error")
                )
            operation = repaired_operation
            arguments = repaired_arguments

        # A successful CAS evaluation can still be incomplete: e.g. it can
        # calculate time of flight while omitting the requested maximum height.
        # Check the labelled payload, and spend another model pass only when
        # one or more explicitly requested outputs are missing.
        if requested_outputs and not self._requested_outputs_present(
            requested_outputs, result.get("result")
        ):
            completion_system = formulation_system + r"""

COMPLETENESS REPAIR:
The previous expression executed, but its result did not contain every
requested output. Return a corrected single Wolfram operation that computes
all requested outputs in one result. Treat the supplied checklist as mandatory.
For a multi-output calculate expression, label every value with the requested
output symbols or equivalent lowerCamelCase labels. Do not return a Solve
result for only one intermediate variable when more quantities were requested.
"""
            completion_prompt = json.dumps(
                {
                    "user_request": problem,
                    "required_outputs": requested_outputs,
                    "previous_tool_call": {
                        "operation": operation,
                        "arguments": arguments,
                    },
                    "previous_wolfram_result": result.get("result"),
                    "explanation_preference": explanation,
                },
                ensure_ascii=False,
            )
            repair_started = time.perf_counter()
            try:
                completion = self.router.generate(
                    prompt=completion_prompt,
                    system_prompt=completion_system,
                    temperature=0.0,
                    request_timeout=None,
                    **structured_kwargs,
                )
            except Exception as exc:
                completion = {"success": False, "error": str(exc)}
            formulation_ms += round(
                (time.perf_counter() - repair_started) * 1000, 2
            )
            completion_plan = (
                self._parse_json(completion.get("result"))
                if completion.get("success") else None
            )
            if isinstance(completion_plan, dict):
                completion_operation = completion_plan.get("operation")
                completion_arguments = completion_plan.get("arguments")
                completion_validation = self._validate_tool_call(
                    {"tool": completion_operation, "arguments": completion_arguments}
                ) if isinstance(completion_operation, str) and isinstance(completion_arguments, dict) else {"valid": False}
                if completion_validation.get("valid"):
                    retry_started = time.perf_counter()
                    retry_result = self._execute_tool(
                        completion_validation["tool"],
                        completion_validation["arguments"],
                    )
                    wolfram_ms += round(
                        (time.perf_counter() - retry_started) * 1000, 2
                    )
                    if (
                        retry_result.get("success")
                        and self._requested_outputs_present(
                            requested_outputs, retry_result.get("result")
                        )
                    ):
                        operation = completion_validation["tool"]
                        arguments = completion_validation["arguments"]
                        result = retry_result
                    else:
                        return self._error(
                            "The computation executed, but the solver could not verify that every requested result was produced. Try asking for each requested quantity explicitly."
                        )
                else:
                    return self._error(
                        completion_validation.get("error")
                        or "The completeness repair produced an invalid computation."
                    )
            else:
                return self._error(
                    "The solver could not produce a complete structured computation for every requested result."
                )

        tool_results = [
            {
                "tool": operation,
                "arguments": arguments,
                "result": result.get("result"),
            }
        ]

        if explanation == "off":
            response = self._cas_result_only(tool_results)
            response["engine"] = "wolfram"
            response["tool_calls"] = []
            response["cached"] = False
            response["timings_ms"] = {
                "planner": formulation_ms,
                "wolfram": wolfram_ms,
                "final_llm": 0.0,
            }
            response["llm_passes"] = 1
            return response

        response = self._final_explanation(
            problem=problem,
            idea="",
            reasoning=[],
            tool_results=tool_results,
            explanation=explanation,
            estimated_answer_tokens=None,
        )
        response["timings_ms"] = {
            "planner": formulation_ms,
            "wolfram": wolfram_ms,
            "final_llm": response.pop("_final_llm_ms", 0.0),
        }
        response["llm_passes"] = 2
        return response

    # =========================================================
    # COMPLEX COMPUTATIONAL PIPELINE
    # =========================================================

    def _solve_complex(self, problem: str, explanation: str = "detailed"):
        """Run the semantic planner/tool pipeline followed by one final solver pass."""
        planner_system = r"""
You are Nova's semantic STEM request planner.

Classify the user's request by meaning. Do not solve it and do not write the final answer.

Return ONLY one JSON object with exactly:
type, intent, needs_wolfram, operation, arguments, answer_tokens

Rules:
- type must be "plan".
- needs_wolfram is boolean.
- Route by semantic intent, never by isolated subject words.
- Use Wolfram only when exact symbolic or numeric computation is genuinely useful
  and the request contains enough mathematical information.
- Conceptual, factual, qualitative, derivation, proof, nomenclature, and ordinary
  explanation requests normally use needs_wolfram=false.
- For exact computation, operation must be one of:
  calculate, simplify, factor, expand, differentiate, integrate, solve, limit.
- arguments must be a machine-readable object containing mathematical expressions
  and values from the user's request. Never put natural-language instructions in it.
- For calculate use {"expression":"..."}.
- A determinate numerical STEM word problem with enough numerical data for an
  exact calculation should normally use needs_wolfram=true and operation=calculate.
  Do NOT classify such a request as pure reasoning merely because it is written
  in natural language or because it is physics, chemistry, or another STEM domain.
- For a numerical word problem, construct a self-contained Wolfram Language
  expression that computes the requested quantities from the stated data. You
  may derive the necessary equations internally while constructing that
  expression, but never invent unstated data or assumptions.
- When several requested numerical quantities can be evaluated together, return
  one calculate expression that produces all of them as a compact list or rule
  set. Prefer explicit numerical substitution and standard mathematical
  functions such as Sqrt, Sin, Cos, Tan, Pi, Exp, Log, N, Solve, or With when
  they are genuinely required.
- Do not use calculate for factual or qualitative requests with no determinate
  numerical computation.
- Never invent missing data, formulas, or assumptions.
- answer_tokens is advisory metadata only, never a hard output ceiling.

For explanation "off", estimate the complete requested result payload, including every quantity,
formula, equation, reaction, or other result explicitly requested by the user. For concise/detailed,
estimate the complete answer. Never use answer_tokens to omit requested results.
"""
        planner_started = time.perf_counter()
        try:
            planner = self.router.plan(
                prompt=json.dumps(
                    {"explanation_preference": explanation, "user_request": problem},
                    ensure_ascii=False,
                ),
                system_prompt=planner_system,
                max_tokens=self.PLANNER_TOKENS,
                request_timeout=None,
            )
        except Exception as exc:
            planner = {"success": False, "error": str(exc)}

        planner_ms = round((time.perf_counter() - planner_started) * 1000, 2)
        plan = self._parse_json(planner.get("result")) if planner.get("success") else None

        # Planner output is advisory. Never expose malformed planner JSON as a
        # user-visible routing failure.
        if not isinstance(plan, dict):
            recovered = self._detect_explicit_operation(problem)
            if recovered is not None:
                operation, arguments = recovered
                plan = {
                    "type": "plan",
                    "intent": "exact",
                    "needs_wolfram": True,
                    "operation": operation,
                    "arguments": arguments,
                    "answer_tokens": None,
                }
            else:
                plan = {
                    "type": "plan",
                    "intent": "reasoning",
                    "needs_wolfram": False,
                    "operation": None,
                    "arguments": None,
                    "answer_tokens": None,
                }

        if plan.get("type") != "plan":
            plan["type"] = "plan"

        needs_wolfram = plan.get("needs_wolfram")
        if isinstance(needs_wolfram, str):
            needs_wolfram = needs_wolfram.strip().lower() in {"true", "yes", "1"}
        if not isinstance(needs_wolfram, bool):
            needs_wolfram = False

        operation = plan.get("operation")
        arguments = plan.get("arguments")

        # Explicit supported operations are a deterministic recovery path.
        recovered = self._detect_explicit_operation(problem)
        if recovered is not None and not needs_wolfram:
            operation, arguments = recovered
            needs_wolfram = True

        if not needs_wolfram or not isinstance(operation, str) or not isinstance(arguments, dict):
            needs_wolfram = False
            operation = None
            arguments = None

        tool_results = []
        wolfram_ms = 0.0

        if needs_wolfram:
            validated = self._validate_tool_call({
                "tool": operation,
                "arguments": arguments,
            })
            if validated.get("valid"):
                tool_started = time.perf_counter()
                result = self._execute_tool(operation, arguments)
                wolfram_ms = round((time.perf_counter() - tool_started) * 1000, 2)
                if result.get("success"):
                    tool_results.append({
                        "tool": operation,
                        "arguments": arguments,
                        "result": result.get("result"),
                    })
                else:
                    # The lightweight planner is not the final computational
                    # authority. If its Wolfram call fails, immediately retry
                    # through the main local reasoning + Wolfram formulation
                    # path, which can reinterpret the original request.
                    fallback = self._solve_llm_wolfram(
                        problem,
                        explanation=explanation,
                    )
                    return fallback
            else:
                # Invalid advisory routing falls back to semantic solving.
                needs_wolfram = False
                operation = None
                arguments = None

        answer_tokens = plan.get("answer_tokens")
        if isinstance(answer_tokens, str) and answer_tokens.strip().isdigit():
            answer_tokens = int(answer_tokens.strip())
        if type(answer_tokens) is not int or answer_tokens <= 0:
            answer_tokens = None

        if explanation == "off" and tool_results:
            response = self._cas_result_only(tool_results)
            response["timings_ms"] = {
                "planner": planner_ms,
                "wolfram": wolfram_ms,
                "final_llm": 0.0,
            }
            response["llm_passes"] = 0
            return response

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
            "wolfram": wolfram_ms,
            "final_llm": response.pop("_final_llm_ms", 0.0),
        }
        response["llm_passes"] = 1
        return response

    def _extract_scaled_numeric_values(problem: str) -> list[float]:
        """Find explicitly prefixed numeric quantities without knowing their domain."""
        prefixes = {
            "da": 1e1, "h": 1e2, "k": 1e3, "M": 1e6, "G": 1e9,
            "T": 1e12, "P": 1e15, "E": 1e18, "Z": 1e21, "Y": 1e24,
            "R": 1e27, "Q": 1e30,
            "d": 1e-1, "c": 1e-2, "m": 1e-3,
            "µ": 1e-6, "μ": 1e-6, "u": 1e-6, "n": 1e-9, "p": 1e-12,
            "f": 1e-15, "a": 1e-18, "z": 1e-21, "y": 1e-24,
            "r": 1e-27, "q": 1e-30,
        }
        # Require at least one unit character after the prefix. This avoids
        # treating ordinary values such as "0.5 m" as milli-prefixed numbers.
        normalized = problem.replace("Μ", "μ").replace("µ", "μ")
        pattern = re.compile(
            r"(?P<number>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
            r"\s*(?P<prefix>da|μ|[kMGTPEZYRQdcmunpfazyr])"
            r"(?P<unit>[A-Za-zΩ]+)",
            re.VERBOSE,
        )
        values: list[float] = []
        for match in pattern.finditer(normalized):
            prefix = match.group("prefix")
            unit = match.group("unit")
            # kg is the SI base unit rather than a kilo-prefixed gram quantity.
            # Do not force a conversion merely because its spelling contains k.
            if prefix == "k" and unit == "g":
                continue
            try:
                base = float(match.group("number"))
            except ValueError:
                continue
            values.append(base * prefixes[prefix])
        return values

    @staticmethod
    def _numeric_factor_present(expression: str, value: float) -> bool:
        """Check whether an extracted expression explicitly retains a numeric factor."""
        text = re.sub(r"\s+", "", expression).lower()
        if value == 0:
            return bool(re.search(r"(?<![\d.])0(?:\.0*)?(?![\d.])", text))

        # Accept ordinary scientific notation and explicit 10^n notation.
        scientific = format(value, ".15g").lower()
        if scientific in text:
            return True

        coefficient = abs(value)
        exponent = 0
        if coefficient:
            mantissa, exp_text = f"{coefficient:.15e}".split("e")
            exponent = int(exp_text)
            mantissa = mantissa.rstrip("0").rstrip(".")
            variants = {
                mantissa + "e" + str(exponent),
                mantissa + "*10^" + str(exponent),
                mantissa + "*10**" + str(exponent),
                "(" + mantissa + "*10^" + str(exponent) + ")",
                "(" + mantissa + "*10**" + str(exponent) + ")",
            }
            if any(v in text for v in variants):
                return True

        # Also accept the exact decimal representation for modest values.
        decimal = f"{value:.15f}".rstrip("0").rstrip(".")
        return decimal in text

    @staticmethod
    def _format_numeric_factor(value: float) -> str:
        return f"{value:.15g}"

    def _detect_result_only_operation(self, problem: str):
        text = problem.strip()
        definite = re.match(
            r"(?is)^\s*(?:∫|\\int)\s*_?\{?([^,{}]+)\}?\s*\^\s*\{?([^}\s]+)\}?\s*(.*?)\s+d\s*([A-Za-z])\s*[.!?]?\s*$",
            text,
        )
        if definite:
            lower, upper, expression, variable = definite.groups()
            expression = normalize_expression(expression.strip())
            if expression:
                return "integrate_definite", {"expression": expression, "variable": variable, "lower": normalize_expression(lower.strip()), "upper": normalize_expression(upper.strip())}

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
    def _extract_numeric_calculation(self, problem: str) -> str | None:
        """Extract only a complete arithmetic expression wrapped in a generic request."""
        match = re.match(
            r"(?is)^\s*(?:(?:please\s+)?(?:calculate|compute|evaluate|work\s+out)\s+|(?:what\s+is|what's)\s+|find\s+(?:the\s+)?value\s+of\s+)(.+?)\s*[?.!]*\s*$",
            problem,
        )
        if not match:
            return None

        expression = match.group(1).strip()
        if self._is_safe_numeric_expression(expression):
            return expression
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

    @staticmethod
    def _is_clean_symbolic_argument(value: str, allow_equals: bool = False) -> bool:
        """Accept standalone symbolic input, not prose or chained commands."""
        text = value.strip()
        if not text or len(text) > 500 or ";" in text or chr(96) in text:
            return False
        allowed = r"A-Za-z0-9_+\-*/^%().,\[\]{}\s"
        if allow_equals:
            allowed += "="
        if re.search(rf"[^{allowed}]", text):
            return False
        return bool(re.search(r"[A-Za-z0-9]", text))

    def _detect_explicit_operation(self, problem: str):
        """Detect an unambiguous symbolic operation and build safe arguments."""

        text = problem.strip()

        # DIFFERENTIATE
        match = re.match(
            r"(?is)^\s*(?:differentiate\s*:?[\s]+|find\s+(?:the\s+)?derivative\s+(?:of\s+)?:?[\s]+|the\s+derivative\s+of\s*:?[\s]+)(.+?)\s*[.!?]?\s*$",
            text,
        )
        if match:
            expression = match.group(1).strip()
            expression = expression.strip("`").strip()
            # If written as y = ..., differentiate the right-hand side.
            expression = re.sub(r"(?is)^y\s*=\s*", "", expression).strip()
            if not self._is_clean_symbolic_argument(expression, allow_equals=True):
                return None
            expression = normalize_expression(expression)
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

        # DEFINITE INTEGRAL notation, e.g. ∫_0^1 x^2 ln(x) dx.
        definite = re.match(
            r"(?is)^\s*(?:integrate\s+)?(?:∫|\\int)\s*_?\{?([^,{}]+)\}?\s*\^\s*\{?([^}\s]+)\}?\s*(.*?)\s+d\s*([A-Za-z])\s*[.!?]?\s*$",
            text,
        )
        if definite:
            lower, upper, expression, variable = definite.groups()
            expression = normalize_expression(expression.strip())
            if expression:
                return ("integrate_definite", {"expression": expression, "variable": variable, "lower": normalize_expression(lower.strip()), "upper": normalize_expression(upper.strip())})

        # INTEGRATE
        match = re.match(
            r"(?is)^\s*(?:integrate|find\s+(?:the\s+)?integral\s+(?:of\s+)?):?[\s]+(.+?)\s*[.!?]?\s*$",
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
            if not self._is_clean_symbolic_argument(expression):
                return None
            expression = normalize_expression(expression)
            return ("integrate", {
                "expression": expression,
                "variable": variable,
            })

        # FACTOR / EXPAND / SIMPLIFY
        for operation, pattern in (
            ("factor", r"(?is)^\s*(?:factor|factorize)\s*:?[\s]+(.+?)\s*[.!?]?\s*$"),
            ("expand", r"(?is)^\s*expand\s*:?[\s]+(.+?)\s*[.!?]?\s*$"),
            ("simplify", r"(?is)^\s*simplify\s*:?[\s]+(.+?)\s*[.!?]?\s*$"),
        ):
            match = re.match(pattern, text)
            if match:
                expression = re.sub(r"[.!?]+$", "", match.group(1).strip()).strip()
                if not self._is_clean_symbolic_argument(expression):
                    continue
                return (operation, {"expression": expression})

        # SOLVE
        match = re.match(
            r"(?is)^\s*(?:solve|find\s+(?:the\s+)?(?:roots|solutions))\s*:?[\s]*(.+?)\s*[.!?]?\s*$",
            text,
        )
        if match:
            raw_equation = re.sub(r"[.!?]+$", "", match.group(1).strip()).strip()
            # Clean symbolic solve requests stay on the exact fast path.
            # Anything that cannot be safely parsed is handled by the main
            # local LLM structured-tool path in solve().
            if not self._is_clean_symbolic_argument(raw_equation, allow_equals=True):
                return None
            equation = re.sub(r"\s*=\s*", " == ", raw_equation)
            if "==" not in equation:
                equation = f"{equation} == 0"
            return ("solve", {
                "equation": equation,
                "variable": "x",
            })

        # LIMIT
        match = re.match(
            r"(?is)^\s*(?:limit|evaluate\s+(?:the\s+)?limit)\s*:?[\s]+(?:of\s+)?(.+?)\s*[.!?]?\s*$",
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

    @staticmethod
    def _is_planner_calculate_expression(expression: str) -> bool:
        """
        Validate a planner-produced Wolfram calculation expression.
        """
        text = expression.strip()
        if not text or len(text) > 2000:
            return False

        # Reject strings and general programming/network/file syntax.
        if re.search(r"""["'\\#&?|:$]""", text):
            return False

        identifiers = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)
        blocked = {
            "Run", "RunProcess", "StartProcess", "SystemOpen",
            "Import", "Export", "Get", "Needs", "Put", "PutAppend",
            "OpenRead", "OpenWrite", "Read", "ReadString", "Write",
            "URLRead", "URLExecute", "URLDownload", "ExternalEvaluate",
            "CloudEvaluate", "CloudDeploy", "APIFunction",
            "NotebookOpen", "NotebookPut", "FrontEndExecute",
            "ToExpression", "Symbol", "Unset", "Clear", "ClearAll",
            "SetDelayed", "RuleDelayed",
        }
        if any(identifier in blocked for identifier in identifiers):
            return False

        # Allow arithmetic, comparisons, function calls, With/local
        # assignments, lists, and immediate rules.
        if re.search(r"[^A-Za-z0-9_+\-*/^%().,\[\]{}=<>;@π∞√\s]", text):
            return False

        return bool(re.search(r"\d|[A-Za-z]", text))

    @staticmethod
    def _is_safe_structured_argument(value: str, allow_equals: bool = False) -> bool:
        text = value.strip()
        if not text or len(text) > 500:
            return False
        if re.search(r"""["'\\#$&?|:]""", text):
            return False
        allowed = r"A-Za-z0-9_+\-*/^%().,\[\]{}\sπ∞√"
        if allow_equals:
            allowed += "="
        if re.search(rf"[^{allowed}]", text):
            return False
        return bool(re.search(r"[A-Za-z0-9]", text))

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
            "integrate_definite",
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

            if not self._is_planner_calculate_expression(expression):
                return {
                    "valid": False,
                    "error": "calculate requires a valid computational expression.",
                }

        # Direct structured operations carry Wolfram fragments rather than
        # arbitrary natural-language text. Keep the grammar conservative before
        # execution while still allowing the normal symbolic syntax.
        symbolic_fields = {
            "expression",
            "equation",
            "variable",
            "lower",
            "upper",
            "value",
        }
        for key, value in arguments.items():
            if key not in symbolic_fields or not isinstance(value, str):
                continue

            # calculate expressions already pass the dedicated Wolfram
            # calculation validator above. That validator intentionally allows
            # richer executable syntax such as immediate rules (->), local
            # assignments, comparisons, lists, and function calls needed to
            # return several computed quantities in one expression.
            if tool == "calculate" and key == "expression":
                continue

            if not self._is_safe_structured_argument(
                value,
                allow_equals=(tool == "solve" and key == "equation"),
            ):
                return {
                    "valid": False,
                    "error": f"{tool} argument '{key}' contains invalid Wolfram syntax.",
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

            elif tool == "integrate_definite":
                raw = provider.engine.integrate_definite(
                    normalize_expression(arguments["expression"]),
                    arguments.get("variable", "x"),
                    normalize_expression(arguments["lower"]),
                    normalize_expression(arguments["upper"]),
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
Return ONLY the complete result payload requested by the user. No explanation, heading,
lead-in, summary, justification, or commentary.

If the user requests multiple quantities, parts, conditions, equations, products, values,
or other outputs, you MUST return ALL of them. Treat the original question as a checklist
and internally verify that every explicit requested output is present before responding.
Never answer only the last or easiest part.

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
requested result. For multiple requested values, emit every requested value,
each clearly separated but without explanatory prose. For a requested equation
or reaction, emit only that equation or reaction.

Output only the requested payload itself.
Never write a sentence such as "The answer is ...", "The result is ...",
"The formula is ...", or "The value is ...". Do not add a label, lead-in,
explanation, justification, or concluding sentence. If several outputs are
requested, output all of them in a compact structured form, using mathematical
notation or line separation as appropriate. Do not omit any requested output.
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
- Keep scientific notation transport-safe outside math delimiters: never emit
  raw Unicode subscript/superscript digits, Greek letters, reaction arrows, or
  similar special symbols in prose. Put them in LaTeX instead, for example
  $\\mathrm{CH_3CH_2OH}$, $\\omega$, or $\\rightarrow$.
- Never emit mojibake or replacement characters such as â, Â, Ã, Ï, ð, or �.
  If a scientific symbol is needed, express it in LaTeX or unambiguous ASCII;
  never output a damaged Unicode fragment.
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
- Before writing the answer, perform a private adversarial verification pass:
  independently recompute the requested result from the original problem,
  challenge the most likely assumption or error, and reconcile any discrepancy
  before emitting the solution.
- For scientific transformations, verify the stated starting entities,
  transformations, intermediate states, final entities, quantities, units,
  conservation relationships, charges, and structural constraints against the
  original request. Do not infer a result merely because it sounds plausible;
  apply the governing rule that actually applies.

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
 
Transport-safe notation: use LaTeX for scientific symbols in displayed or inline math. Do not emit raw Unicode subscripts, superscripts, Greek letters, reaction arrows, or mojibake/replacement characters in prose; use ASCII or LaTeX instead.

For differentiation, explicitly apply the relevant rules. For quantitative physics or chemistry, show the necessary equations, substitutions, units, and calculations. For derivations and proofs, continue through every necessary step to the requested final formula, result, or conclusion. For factual or conceptual questions, answer the actual question clearly without inventing a calculation.

Be complete but disciplined: do not repeat the same idea, add unrelated
background, present alternative methods, or continue after the result is
established. Stop when the solution is complete.

Never stop at an intermediate setup or partial argument. Never invent missing
information or assumptions.

Before emitting the solution, perform a private adversarial verification pass.
Recompute the requested result independently from the original problem,
challenge the most likely assumption or tempting but incorrect interpretation,
and reconcile any discrepancy. For scientific transformations, verify the
starting entities, transformations, intermediate states, final entities,
quantities, units, conservation relationships, charges, and structural
constraints against the original request.

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

        # Preserve the COMPLETE generated result payload.
        #
        # The result-only prompt may contain several independent requested
        # values, each in its own display/inline math block. Never select only
        # the final block: doing so silently drops every earlier requested
        # result.
        blocks = re.findall(
            r"\`\`\`(?:math|latex)?\s*(.*?)\s*\`\`\`",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if blocks:
            return "\\n".join(
                block.strip()
                for block in blocks
                if block.strip()
            )

        displays = re.findall(r"\$\$(.*?)\$\$", text, flags=re.DOTALL)
        if displays:
            return "\\n".join(
                display.strip()
                for display in displays
                if display.strip()
            )

        displays = re.findall(r"\\\[(.*?)\\\]", text, flags=re.DOTALL)
        if displays:
            return "\\n".join(
                display.strip()
                for display in displays
                if display.strip()
            )

        inline = re.findall(
            r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$",
            text,
            flags=re.DOTALL,
        )
        if inline:
            return "\\n".join(
                "$" + value.strip() + "$"
                for value in inline
                if value.strip()
            )

        return text

    # =========================================================
    # JSON PARSER
    # =========================================================

    @staticmethod
    def _extract_json_string_field(text: Any, field: str) -> str | None:
        """Recover one JSON string field from lightly wrapped model output."""
        if not isinstance(text, str) or not text.strip():
            return None

        value = text.strip()
        value = re.sub(r"^<think>.*?</think>\s*", "", value, flags=re.I | re.S)

        start = value.find("{")
        end = value.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(value[start:end + 1])
                if isinstance(parsed, dict) and isinstance(parsed.get(field), str):
                    return parsed[field].strip()
            except json.JSONDecodeError:
                pass

        match = re.search(
            rf'"{re.escape(field)}"\s*:\s*("(?:\\.|[^"\\])*")',
            value,
            flags=re.S,
        )
        if match:
            try:
                parsed_value = json.loads(match.group(1))
                if isinstance(parsed_value, str):
                    return parsed_value.strip()
            except json.JSONDecodeError:
                pass

        return None

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

    @staticmethod
    def _requested_outputs_present(requested_outputs: list[str], result: Any) -> bool:
        """Check that each requested output label appears in a computed result."""
        if not isinstance(result, str) or not result.strip():
            return False
        normalized_result = re.sub(r"[^a-z0-9]", "", result.lower())
        if not normalized_result:
            return False
        for output in requested_outputs:
            label = re.sub(r"[^a-z0-9]", "", output.lower())
            if not label or label not in normalized_result:
                return False
        return True

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