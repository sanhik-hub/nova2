import json
import logging
import re
import time
from collections import OrderedDict
from threading import Lock
from typing import Any

from backend.services.provider_router import ProviderRouter

logger = logging.getLogger(__name__)
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

    # Structured tool calls should be compact. This budget applies only to
    # the machine-readable Wolfram request, never to user-facing solutions.
    STRUCTURED_TOOL_TOKENS = 256

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

    @staticmethod
    def _normalize_generated_calculate_expression(
        operation: str,
        arguments: dict[str, Any],
        requested_outputs: list[str],
    ) -> dict[str, Any]:
        """Normalize a simple assignment-plus-list draft into labelled rules."""
        if operation != "calculate" or not isinstance(arguments, dict):
            return arguments
        expression = arguments.get("expression")
        if not isinstance(expression, str) or ";" not in expression:
            return arguments

        parts: list[str] = []
        start = 0
        stack: list[str] = []
        pairs = {")": "(", "]": "[", "}": "{"}
        in_string = False
        escaped = False
        for index, char in enumerate(expression):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char in "([{":
                stack.append(char)
            elif char in ")]}":
                if stack and stack[-1] == pairs[char]:
                    stack.pop()
            elif char == ";" and not stack:
                parts.append(expression[start:index].strip())
                start = index + 1
        parts.append(expression[start:].strip())
        if len(parts) < 2 or any(not part for part in parts):
            return arguments

        assignments: dict[str, str] = {}
        for part in parts[:-1]:
            match = re.fullmatch(r"([A-Za-z][A-Za-z0-9]*)\s*=\s*(.+)", part, re.DOTALL)
            if not match:
                return arguments
            symbol, value = match.groups()
            if symbol in assignments:
                return arguments
            assignments[symbol] = value.strip()

        list_match = re.fullmatch(
            r"(?:\{\s*(.*?)\s*\}|List\s*\[\s*(.*?)\s*\])",
            parts[-1],
            re.DOTALL,
        )
        if not list_match:
            return arguments
        symbols_text = list_match.group(1) if list_match.group(1) is not None else list_match.group(2)
        symbols = [item.strip() for item in symbols_text.split(",")]
        if (
            not symbols
            or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", item) for item in symbols)
            or len(symbols) != len(assignments)
            or set(symbols) != set(assignments)
        ):
            return arguments
        if requested_outputs:
            normalized_requested = {
                re.sub(r"[^a-z0-9]", "", item.lower()) for item in requested_outputs
            }
            normalized_symbols = {
                re.sub(r"[^a-z0-9]", "", item.lower()) for item in symbols
            }
            if normalized_requested != normalized_symbols:
                return arguments

        normalized = dict(arguments)
        normalized["expression"] = "List[" + ", ".join(
            f"Rule[{symbol}, {assignments[symbol]}]" for symbol in symbols
        ) + "]"
        return normalized

    @staticmethod
    def _answer_template_meets_mode(
        template: str | None,
        problem: str,
        requested_outputs: list[str],
        explanation: str,
    ) -> bool:
        """Validate template structure without requiring a fixed equation count."""
        if explanation == "off":
            return True
        if not isinstance(template, str) or not template.strip() or not requested_outputs:
            return False

        placeholders = re.findall(r"\{\{([A-Za-z][A-Za-z0-9_]*)\}\}", template)
        if "{{" in template and not placeholders:
            return False

        normalize_label = lambda value: re.sub(r"[^a-z0-9]", "", value.lower())
        expected = {normalize_label(label) for label in requested_outputs}
        supplied = {normalize_label(label) for label in placeholders}
        if not expected or supplied != expected:
            return False

        # For numerical word problems, every requested value must be attached
        # to a mathematical relation. One relation may contain several equal
        # signs or intermediate steps; counting a fixed number per output
        # incorrectly rejects valid templates that combine steps compactly.
        if re.search(r"\d", problem):
            relation_pattern = r"=|≈|\\(?:approx|to|rightarrow|Rightarrow|propto)|→|∝"
            for placeholder in placeholders:
                marker = "{{" + placeholder + "}}"
                position = template.find(marker)
                prefix = template[:position]
                # Limit the check to this output's clause so a relation for an
                # earlier result cannot make a later value-only label look valid.
                boundaries = list(re.finditer(r";|\n|(?<=[.!?])\s+|\}\}", prefix))
                clause_start = boundaries[-1].end() if boundaries else 0
                preceding = template[clause_start:position]
                if not re.search(relation_pattern, preceding):
                    return False

        without_placeholders = re.sub(
            r"\{\{[A-Za-z][A-Za-z0-9_]*\}\}", "", template
        )
        if explanation == "detailed":
            meaningful_words = re.findall(
                r"[A-Za-zÀ-ÖØ-öø-ÿ]{3,}", without_placeholders
            )
            # Detailed answers need some explanatory derivation, but the
            # threshold is intentionally modest and independent of output count.
            if len(meaningful_words) < 4:
                return False
        return True

    @classmethod
    def _augment_answer_template_with_calculation(
        cls,
        template: str | None,
        requested_outputs: list[str],
        expression: str | None,
        problem: str,
        explanation: str,
    ) -> str | None:
        """Add verified computation expressions when a numeric template omits them.

        This keeps a model-authored template in one pass when its prose is useful
        but it forgot to show the calculation. Only expressions present in the
        already validated Wolfram call are inserted; computed values are still
        supplied exclusively by the Wolfram result during rendering.
        """
        if (
            explanation not in {"concise", "detailed"}
            or not re.search(r"\d", problem)
            or not isinstance(template, str)
            or not template.strip()
            or not isinstance(expression, str)
            or not requested_outputs
        ):
            return template

        formulas = cls._extract_labelled_wolfram_values(expression)
        if not formulas:
            return template

        normalize_label = lambda value: re.sub(r"[^a-z0-9]", "", value.lower())
        formula_by_label = {
            normalize_label(label): value.strip()
            for label, value in formulas.items()
        }
        placeholders = re.findall(r"\{\{([A-Za-z][A-Za-z0-9_]*)\}\}", template)
        expected = {normalize_label(label) for label in requested_outputs}
        supplied = {normalize_label(label) for label in placeholders}
        if not placeholders or supplied != expected:
            return template

        # Prepare all additions first. Refuse partial augmentation if even one
        # output lacks a genuine expression (rather than a bare numeric value).
        additions: dict[str, str] = {}
        relation_pattern = r"=|≈|\\(?:approx|to|rightarrow|Rightarrow|propto)|→|∝"
        for placeholder in placeholders:
            label = normalize_label(placeholder)
            formula = formula_by_label.get(label)
            if not formula or not re.search(r"[+*/^()]|-(?!>)", formula):
                continue
            additions[placeholder] = formula

        if set(map(normalize_label, additions)) != expected:
            return template

        augmented = template
        for placeholder in placeholders:
            marker = "{{" + placeholder + "}}"
            position = augmented.find(marker)
            if position < 0:
                return template
            # Evaluate each output against the original template, so adding a
            # relation for an earlier output cannot mask a missing relation
            # for a later output joined with "and".
            original_position = template.find(marker)
            prefix = template[:original_position]
            boundaries = list(re.finditer(r";|\n|(?<=[.!?])\s+|\}\}", prefix))
            clause_start = boundaries[-1].end() if boundaries else 0
            preceding = template[clause_start:original_position]
            if not re.search(relation_pattern, preceding):
                augmented = augmented.replace(
                    marker, additions[placeholder] + " = " + marker, 1
                )
        return augmented

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
Convert the user's computational STEM request into one executable Wolfram Language
tool call. Return only the JSON object required by the schema. Do not write a solution or explain in prose; do not put explanations inside an expression.

For natural-language quantitative problems, use operation "calculate". For multiple
requested outputs, return one List of Rule expressions, with one exact lowerCamelCase
label per requested output in the original order. Substitute all supplied numerical
values directly; do not use assignments, intermediate symbols, or unrelated equations.
Use explicit multiplication (*), valid Wolfram function syntax, and Degree for angles
stated in degrees. Preserve signs, units, conditions, and the actual requested outputs.
Never invent data or assumptions.

For concise/detailed modes, include a complete answer_template in this same JSON
response. Use {{label}} placeholders for every requested output, matching the labels
exactly. Include the governing relation, necessary substitution, and units appropriate
to the selected explanation_mode. Do not guess computed values. For explanation off,
omit answer_template.

Use the appropriate supported operation for symbolic requests. Preserve the user's
expression and variable for differentiation, integration, limits, and equation solving.
Return one balanced executable Wolfram expression with no trailing markup, assignments,
natural-language instructions, external functions, files, URLs, or side effects.
Use ASCII Wolfram symbols and no underscores in symbols. Output only JSON.
"""
        formulation_prompt = json.dumps(
            {
                "user_request": problem,
                "explanation_mode": explanation,
            },
            ensure_ascii=False,
        )

        # Use llama.cpp's constrained JSON grammar for the formulation contract.
        # Native tool-call templates can produce syntactically valid tool calls
        # whose expression field contains derivation prose; constrained JSON
        # keeps the outer contract stable while the existing Wolfram validator
        # remains responsible for rejecting unsafe or non-executable expressions.
        formulation_schema = {
            "type": "object",
            "properties": {
                "requested_outputs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Every output explicitly requested, in request order.",
                },
                "answer_template": {
                    "type": "string",
                    "description": (
                        "For concise/detailed modes, a complete solution template. "
                        "Use {{label}} placeholders for all requested outputs; do not "
                        "guess or include computed numerical values. Omit in result-only mode."
                    ),
                },
                "operation": {
                    "type": "string",
                    "enum": (
                        ["calculate"]
                        if self._is_direct_quantitative_request(problem)
                        else [
                            "calculate",
                            "simplify",
                            "factor",
                            "expand",
                            "differentiate",
                            "integrate",
                            "integrate_definite",
                            "solve",
                            "limit",
                        ]
                    ),
                    "description": (
                        "Use calculate for natural-language numerical word problems. "
                        "A multi-output numerical request must compute every requested "
                        "quantity in one labelled List of Rule expressions."
                    ),
                },
                "arguments": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": (
                                "Executable Wolfram Language syntax only. "
                                "For calculate, return one balanced executable "
                                "expression only, not explanation or derivation prose. "
                                "Use ASCII variable names, no With/local assignments, "
                                "and directly substitute supplied numeric values in "
                                "numerical word problems. For multiple outputs, use "
                                "one Wolfram list of labelled rules. No trailing markup."
                            ),
                        },
                        "equation": {"type": "string"},
                        "variable": {"type": "string"},
                        "lower": {"type": "string"},
                        "upper": {"type": "string"},
                        "value": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            },
            "required": ["operation", "arguments", "requested_outputs"],
            "additionalProperties": False,
        }

        structured_kwargs = {
            "response_format": {
                "type": "json_object",
                "schema": formulation_schema,
            },
        }

        llm_started = time.perf_counter()
        try:
            formulation = self.router.generate(
                prompt=formulation_prompt,
                system_prompt=formulation_system,
                # Explanation modes include a user-facing template in this
                # generation, so do not apply the compact tool-JSON budget to
                # that content. None delegates to the provider's configured
                # transport fallback.
                max_tokens=(
                    self.STRUCTURED_TOOL_TOKENS
                    if explanation == "off"
                    else None
                ),
                temperature=0.0,
                enable_thinking=False,
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
        answer_template = plan.get("answer_template")
        if not isinstance(answer_template, str) or not answer_template.strip():
            answer_template = None

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

        multi_output_numeric_problem = (
            len(requested_outputs) >= 2
            and self._is_direct_quantitative_request(problem)
        )
        arguments = self._normalize_generated_calculate_expression(
            operation, arguments, requested_outputs
        )
        if multi_output_numeric_problem and operation != "calculate":
            validated = {
                "valid": False,
                "error": (
                    "A numerical word problem with multiple requested outputs "
                    "must use one calculate expression containing all labelled results."
                ),
            }
        else:
            validated = self._validate_tool_call(
                {
                    "tool": operation,
                    "arguments": arguments,
                }
            )
        logger.warning(
            "structured_tool_initial operation=%r arguments=%r valid=%s validation_error=%r",
            operation, arguments, validated.get("valid"), validated.get("error"),
        )
        if not validated.get("valid"):
            # Treat invalid generated syntax as a formulation failure and let
            # the existing repair pass produce a new candidate. The invalid
            # expression is never sent to Wolfram.
            result = {
                "success": False,
                "error": validated.get("error")
                or "The structured computational request was rejected.",
            }
            wolfram_ms = 0.0
        else:
            operation = validated["tool"]
            arguments = validated["arguments"]

            wolfram_started = time.perf_counter()
            result = self._execute_tool(operation, arguments)
            wolfram_ms = round(
                (time.perf_counter() - wolfram_started) * 1000,
                2,
            )
            logger.warning(
                "structured_tool_initial_wolfram operation=%r arguments=%r success=%s error=%r result=%r",
                operation, arguments, result.get("success"), result.get("error"),
                str(result.get("result"))[:1000],
            )

        # Result-only mode is deliberately single-pass: never invoke the model
        # again to repair a rejected expression or a failed Wolfram evaluation.
        # Deterministic normalization/validation and CAS verification still run.
        if explanation == "off" and not result.get("success"):
            return self._error(
                "Single-pass computation failed; no second model generation was "
                "attempted. " + str(result.get("error") or "Wolfram verification failed.")
            )

        repair_attempted = False
        if not result.get("success"):
            # Repair the structured tool call using the actual Wolfram error.
            # This remains a main-LLM formulation pass; it never asks the model
            # to provide an unverified numerical answer.
            repair_attempted = True
            repair_system = r"""
You are repairing a structured Wolfram tool call generated from a STEM
problem.

Return ONLY one JSON object matching this structure:
{"operation":"...","arguments":{...},"requested_outputs":["..."]}

Include every output requested by the original problem in requested_outputs.
Use only these operations:
calculate, simplify, factor, expand, differentiate, integrate,
integrate_definite, solve, limit.

Preserve the original problem and every requested quantity. The
requested_outputs field in the repair payload is authoritative.

For a numerical word problem with two or more requested outputs, you MUST use
operation "calculate" and return exactly one executable Wolfram List of Rule
expressions, one Rule per requested output in the original order. Do not switch
such a request to "solve", because solving one equation usually omits other
requested quantities. Substitute the given numerical values directly into each
right-hand side; do not use assignments, intermediate symbols, or a bare list
of unrelated equations. Keep the exact requested labels.

For other request types, preserve the appropriate operation unless it is
actually invalid for the user's request. For calculate, the expression must be
one executable Wolfram Language expression. For solve, return an executable
equation and variable. For limit, return expression, variable, and value.
Do not use natural-language instructions, strings, external functions, files,
URLs, or programming side effects. Never use underscores in Wolfram symbols.
"""
            repair_prompt = json.dumps(
                {
                    "user_request": problem,
                    "requested_outputs": requested_outputs,
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
                    max_tokens=self.STRUCTURED_TOOL_TOKENS,
                    temperature=0.0,
                    enable_thinking=False,
                    request_timeout=None,
                    **structured_kwargs,
                )
            except Exception as exc:
                repaired = {"success": False, "error": str(exc)}
            formulation_ms += round(
                (time.perf_counter() - repair_started) * 1000,
                2,
            )

            logger.warning(
                "structured_tool_repair_generation success=%s raw_result=%r generation_error=%r",
                repaired.get("success"), str(repaired.get("result"))[:1500], repaired.get("error"),
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

            if multi_output_numeric_problem and repaired_operation != "calculate":
                logger.warning(
                    "structured_tool_repair_rejected_wrong_operation operation=%r required='calculate'",
                    repaired_operation,
                )
                return self._error(
                    "The repair selected an incompatible operation for a numerical "
                    "problem with multiple requested outputs. No partial symbolic "
                    "result was accepted; the repair must calculate all requested "
                    "quantities in one labelled expression."
                )

            repair_validation = self._validate_tool_call(
                {
                    "tool": repaired_operation,
                    "arguments": repaired_arguments,
                }
            )
            logger.warning(
                "structured_tool_repair_validation operation=%r arguments=%r valid=%s validation_error=%r",
                repaired_operation, repaired_arguments, repair_validation.get("valid"),
                repair_validation.get("error"),
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
            logger.warning(
                "structured_tool_repair_wolfram operation=%r arguments=%r success=%s error=%r result=%r",
                repaired_operation, repaired_arguments, result.get("success"), result.get("error"),
                str(result.get("result"))[:1000],
            )
            if not result.get("success"):
                return self._error(
                    "Wolfram verification failed: "
                    + str(result.get("error") or "unknown error")
                )
            operation = repaired_operation
            arguments = repaired_arguments

        # In result-only mode, never make a second model call to repair missing
        # labels or incomplete results. Reject the unverified result explicitly.
        if explanation == "off" and requested_outputs:
            result_text = result.get("result")
            outputs_present = self._requested_outputs_present(
                requested_outputs, result_text
            )
            unresolved_values = bool(
                re.search(r"\d", problem)
                and self._requested_outputs_are_unresolved_symbols(
                    requested_outputs, result_text
                )
            )
            if not outputs_present or unresolved_values:
                return self._error(
                    "Single-pass computation did not verify every requested output; "
                    "no second model generation was attempted."
                )

        # A successful CAS evaluation can still be incomplete. If the one
        # allowed repair pass did not recover every requested output, fail here
        # with a diagnostic instead of launching another very slow generation.
        # This bounds recovery to one repair pass without imposing a global
        # timeout or token limit on the local model.
        if (
            repair_attempted
            and requested_outputs
            and not self._requested_outputs_present(
                requested_outputs, result.get("result")
            )
        ):
            logger.warning(
                "structured_repair_incomplete requested_outputs=%r operation=%r arguments=%r result=%r",
                requested_outputs, operation, arguments, str(result.get("result"))[:1000],
            )
            return self._error(
                "Wolfram repair returned an incomplete result for the requested quantities. "
                "The solver stopped after one repair attempt rather than starting another slow generation."
            )

        # A successful CAS evaluation can still be incomplete. For an initially
        # successful but incomplete computation, retain the existing completeness
        # recovery path; the failed-computation repair path above is bounded.
        if requested_outputs and not self._requested_outputs_present(
            requested_outputs, result.get("result")
        ):
            completion_system = formulation_system + r"""

COMPLETENESS REPAIR — STRICT OUTPUT CONTRACT:
The previous Wolfram result was incomplete. The JSON field required_outputs
in the user payload is the authoritative list of labels. Return a new
calculate expression that computes every item in that list in one evaluation.

For two or more requested outputs, the expression MUST use the Wolfram
List and Rule constructors, with left-hand-side labels exactly matching every
required_outputs entry in the same order, for example:
List[Rule[outputOne, expressionOne], Rule[outputTwo, expressionTwo]]
Use this constructor form rather than curly-brace list syntax because the
application normalizes curly braces in ordinary mathematical input.
The example illustrates output structure only; derive the actual formulas and
values from the original request.

Do NOT use assignments (=), semicolon-separated expressions, equality tests
(==), local variables, intermediate-only results, or a result containing just
one requested quantity. Do not return a rule for an unrequested label. Use
direct expressions on the right-hand side and the exact requested labels on
the left-hand side. Preserve the original problem's physical relationships,
numbers, signs, and units. The output must be a single executable Wolfram
expression, not a derivation or explanation.

Every right-hand side must be a complete expression that Wolfram can evaluate
from the data in the original request. Do not refer to placeholder symbols,
abbreviations, or intermediate variable names such as t or h from the previous
tool call. If the original request provides numerical data for a numerical
word problem, substitute those values directly into every requested result.
Never return a labelled rule whose value is merely an unresolved symbol.
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
                    max_tokens=self.STRUCTURED_TOOL_TOKENS,
                    temperature=0.0,
                    enable_thinking=False,
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
                        and not (
                            re.search(r"\d", problem)
                            and self._requested_outputs_are_unresolved_symbols(
                                requested_outputs, retry_result.get("result")
                            )
                        )
                    ):
                        operation = completion_validation["tool"]
                        arguments = completion_validation["arguments"]
                        result = retry_result
                    else:
                        logger.warning(
                            "requested_outputs verification failed: required=%r "
                            "initial_result=%r retry_expression=%r retry_result=%r",
                            requested_outputs,
                            result.get("result"),
                            completion_validation["arguments"].get("expression"),
                            retry_result.get("result"),
                        )
                        return self._error(
                            "The computation executed, but the solver could not verify that every requested result was produced. Try asking for each requested quantity explicitly."
                        )
                else:
                    return self._error(
                        completion_validation.get("error")                        or "The completeness repair produced an invalid computation."
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

        # The initial LLM generation also authored this template. Fill its
        # requested-output placeholders exclusively from verified Wolfram
        # values. Invalid/missing templates retain the established LLM fallback.
        answer_template = self._augment_answer_template_with_calculation(
            answer_template,
            requested_outputs,
            arguments.get("expression"),
            problem,
            explanation,
        )
        template_is_appropriate = self._answer_template_meets_mode(
            answer_template, problem, requested_outputs, explanation
        )
        rendered_answer = (
            self._render_verified_answer_template(
                answer_template, requested_outputs, result.get("result")
            )
            if template_is_appropriate else None
        )
        if rendered_answer is not None:
            logger.info("Using one-pass deterministic answer rendering.")
            return {
                "success": True,
                "answer": rendered_answer,
                "engine": "local_llm+wolfram",
                "tool_calls": tool_results,
                "cached": False,
                "result_only": False,
                "timings_ms": {
                    # Keep the legacy key for existing consumers. This path's
                    # first generation is the main-model formulation call, not
                    # the lightweight semantic planner used by _solve_complex.
                    "planner": formulation_ms,
                    "formulation": formulation_ms,
                    "wolfram": wolfram_ms,
                    "final_llm": 0.0,
                },
                "llm_passes": 1,
                "answer_rendering": "deterministic_template",
            }

        placeholder_names = re.findall(
            r"\{\{([A-Za-z][A-Za-z0-9_]*)\}\}",
            answer_template or "",
        )
        logger.warning(
            "One-pass answer template unavailable; using LLM fallback "
            "(mode=%s, structurally_valid=%s, requested_outputs=%r, placeholders=%r, "
            "wolfram_result_labels=%r).",
            explanation,
            template_is_appropriate,
            requested_outputs,
            placeholder_names,
            sorted(self._extract_labelled_wolfram_values(result.get("result")).keys()),
        )
        response = self._final_explanation(
            problem=problem,
            idea="",
            reasoning=[],
            tool_results=tool_results,
            explanation=explanation,
            estimated_answer_tokens=None,
        )
        response["timings_ms"] = {
            # Keep the legacy key for existing consumers.
            "planner": formulation_ms,
            "formulation": formulation_ms,
            "wolfram": wolfram_ms,
            "final_llm": response.pop("_final_llm_ms", 0.0),
        }
        response["llm_passes"] = 2
        response["answer_rendering"] = "llm_fallback"
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
                # Include the rejected expression in the local API diagnostic so
                # formulation failures can be distinguished from Wolfram errors.
                # Rejected text is never executed; the conservative validator
                # remains the gatekeeper.
                return {
                    "valid": False,
                    "error": (
                        "calculate requires a valid computational expression; "
                        f"rejected expression={expression!r}"
                    ),
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

    @staticmethod
    def _assignment_sequence_to_labelled_expression(expression: str) -> str | None:
        """Convert a pure multi-output assignment sequence into labelled rules.

        Generated Wolfram calls must not depend on assignment side effects or
        return only the final assignment's value. When every top-level
        semicolon-separated statement is a simple symbol assignment, convert
        it to a List of Rules so each computed value remains labelled.
        """
        if not isinstance(expression, str):
            return None
        # A trailing semicolon is valid Wolfram syntax. Do not let it create
        # an empty statement that prevents labelled-result normalization.
        expression = expression.strip()
        if expression.endswith(";"):
            expression = expression[:-1].rstrip()
        if ";" not in expression:
            return None

        statements = []
        start = 0
        stack = []
        in_string = False
        escaped = False
        pairs = {")": "(", "]": "[", "}": "{"}

        for index, char in enumerate(expression):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char in "([{":
                stack.append(char)
            elif char in ")]}":
                if not stack or stack[-1] != pairs[char]:
                    return None
                stack.pop()
            elif char == ";" and not stack:
                statements.append(expression[start:index].strip())
                start = index + 1

        if in_string or stack:
            return None
        statements.append(expression[start:].strip())

        assignment_pattern = re.compile(
            r"^([A-Za-z][A-Za-z0-9_]*)\s*=(?!=)\s*(.+)$",
            re.DOTALL,
        )
        assignments: dict[str, str] = {}
        for statement in statements:
            match = assignment_pattern.fullmatch(statement)
            if not match:
                break
            name, value = match.groups()
            assignments[name] = value.strip()
        else:
            # Backwards-compatible handling for assignment-only sequences.
            if len(assignments) < 2:
                return None
            rules = []
            for name, value in assignments.items():
                label = ReasoningService._wolfram_label(name)
                if not label:
                    return None
                rules.append(f"Rule[{label}, {normalize_expression(value)}]")
            return "List[" + ", ".join(rules) + "]"

        # Handle model output that initializes givens, assigns requested
        # quantities, and ends with a list of those quantity symbols.
        final_list = statements[-1].strip()
        if not (final_list.startswith("{") and final_list.endswith("}")):
            return None
        output_names = [item.strip() for item in final_list[1:-1].split(",")]
        if len(output_names) < 2 or any(
            not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name)
            or name not in assignments
            for name in output_names
        ):
            return None

        def resolve(name: str, trail: set[str] | None = None) -> str | None:
            trail = set() if trail is None else set(trail)
            if name in trail:
                return None
            trail.add(name)
            value = assignments.get(name)
            if value is None:
                return None
            for dependency in sorted(assignments, key=len, reverse=True):
                if dependency == name:
                    continue
                if re.search(
                    rf"(?<![A-Za-z0-9_$]){re.escape(dependency)}(?![A-Za-z0-9_$])",
                    value,
                ):
                    resolved = resolve(dependency, trail)
                    if resolved is None:
                        return None
                    value = re.sub(
                        rf"(?<![A-Za-z0-9_$]){re.escape(dependency)}(?![A-Za-z0-9_$])",
                        f"({resolved})",
                        value,
                    )
            return value

        rules = []
        for name in output_names:
            label = ReasoningService._wolfram_label(name)
            value = resolve(name)
            if not label or value is None:
                return None
            rules.append(f"Rule[{label}, {normalize_expression(value)}]")
        return "List[" + ", ".join(rules) + "]"

    @staticmethod
    def _wolfram_label(name: str) -> str | None:
        """Convert a generated label to a simple valid Wolfram symbol."""
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            return None
        if "_" in name:
            parts = [part for part in name.split("_") if part]
            if not parts:
                return None
            name = parts[0].lower() + "".join(
                part[:1].upper() + part[1:] for part in parts[1:]
            )
        return name if re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", name) else None

    def _execute_tool(
        self,
        tool: str,
        arguments: dict,
    ) -> dict:

        try:

            provider = self.router.get_provider("wolfram")

            if tool == "calculate":

                expression = arguments["expression"]
                labelled_expression = self._assignment_sequence_to_labelled_expression(
                    expression
                )
                # Assignment sequences otherwise risk being normalized into
                # equality tests and returning only the final requested value.
                # Convert simple multi-output assignments to a pure List[Rule]
                # expression before evaluation; all other expressions retain
                # the existing normalization and safety path.
                normalized = (
                    labelled_expression
                    if labelled_expression is not None
                    else normalize_expression(expression)
                )
                raw = provider.calculate(normalized)

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
                    "result": raw.result,                    "error": None,
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

    # =========================================================    # RESULT-ONLY CAS RESPONSE
    # =========================================================

    @staticmethod
    def _extract_labelled_wolfram_values(result: Any) -> dict[str, str]:
        """Extract top-level labelled values without evaluating Wolfram syntax."""
        if not isinstance(result, str) or not result.strip():
            return {}

        values: dict[str, str] = {}
        rule_pattern = re.compile(
            r"\bRule\s*\[\s*([A-Za-z][A-Za-z0-9_]*)\s*,"
            r"|([A-Za-z][A-Za-z0-9_]*)\s*->\s*"
        )
        for match in rule_pattern.finditer(result):
            label = match.group(1) or match.group(2)
            start = match.end()
            depth = 0
            end = start
            while end < len(result):
                char = result[end]
                if char in "([{":
                    depth += 1
                elif char in ")]}":
                    if depth == 0:
                        break
                    depth -= 1
                elif char == "," and depth == 0:
                    break
                end += 1

            value = result[start:end].strip()
            if value:
                key = re.sub(r"[^a-z0-9]", "", label.lower())
                values[key] = value
        return values

    @classmethod
    def _render_verified_answer_template(
        cls,
        template: str | None,
        requested_outputs: list[str],
        result: Any,
    ) -> str | None:
        """Fill a model-authored template only with values present in CAS output."""
        if not isinstance(template, str) or not template.strip():
            return None
        if not isinstance(result, str) or not result.strip():
            return None

        placeholders = re.findall(r"\{\{([A-Za-z][A-Za-z0-9_]*)\}\}", template)
        if "{{" in template and not placeholders:
            return None

        if requested_outputs:
            expected = {
                re.sub(r"[^a-z0-9]", "", label.lower())
                for label in requested_outputs
            }
            supplied = {
                re.sub(r"[^a-z0-9]", "", label.lower())
                for label in placeholders
            }
            if not expected or not expected.issubset(supplied):
                return None
            values = cls._extract_labelled_wolfram_values(result)
            if not expected.issubset(values.keys()):
                return None
            for placeholder in placeholders:
                key = re.sub(r"[^a-z0-9]", "", placeholder.lower())
                if key not in values:
                    return None
                template = template.replace("{{" + placeholder + "}}", values[key])
        else:
            if placeholders != ["result"]:
                return None
            template = template.replace("{{result}}", result.strip())

        if "{{" in template or "}}" in template:
            return None
        return template.strip() or None

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

        results = [            item.get("result")
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

Give the shortest complete solution that is mathematically and scientifically
sound. Concise means minimal necessary working, not a miniature textbook
explanation. Adapt to the actual request across mathematics, physics, chemistry,
and general science; do not use subject-specific templates or fixed step counts.

Rules:
- Start with the first necessary equation, transformation, or direct answer.
  No title, greeting, restatement of the question, or introductory paragraph.
- Include only the essential relation, substitution/transformation, and result
  needed to establish the answer. Combine consecutive algebraic steps into one
  aligned equation block where practical.
- For numerical problems, show enough substitution and arithmetic to make the
  result auditable, with units. Do not separately explain a formula that is
  already clear from the equation.
- For symbolic work, show the transformations needed to reach the result; do
  not skip a non-obvious step merely to shorten the answer.
- For conceptual/factual questions, answer directly in a few sentences. Add
  detail only when it is needed to be correct or resolve ambiguity.
- Use prose only when an equation or transformation needs a brief explanation.
  No teaching filler, background, alternative methods, recap, or repeated result.
- Do not create separate "Final Answer" sections or restate a result already
  clearly established. Stop as soon as all requested outputs are answered.
- Use LaTeX for mathematical and scientific notation: $...$ inline and $$...$$
  for a necessary display equation. Preserve meaningful subscripts,
  superscripts, fractions, units, chemical bonds, and formal charges.
- Never emit mojibake or damaged scientific symbols. Use LaTeX or unambiguous
  ASCII rather than corrupted Unicode characters.
- Keep the output complete; there is no hard output/token limit and no
  truncation. Brevity must come from removing unnecessary material, not from
  cutting off necessary reasoning.
- Verify the result internally before answering, but do not narrate that check.

Return only the concise solution, with no metadata or commentary.
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

    @staticmethod
    def _requested_outputs_are_unresolved_symbols(
        requested_outputs: list[str], result: Any
    ) -> bool:
        """Reject numerical word-problem results whose labelled values are bare symbols."""
        if not isinstance(result, str) or not result.strip():
            return True
        # Support both ordinary InputForm rules (label -> value) and
        # explicit Wolfram constructors (Rule[label, value]). Keep the two
        # syntaxes separate: mixing their delimiters can misread the value
        # of one rule as the label of the next.
        if "Rule[" in result:
            rules = re.findall(
                r"\bRule\s*\[\s*([A-Za-z][A-Za-z0-9]*)\s*,\s*([A-Za-z][A-Za-z0-9]*)\s*\]",
                result,
            )
        else:
            rules = re.findall(
                r"\b([A-Za-z][A-Za-z0-9]*)\s*->\s*([A-Za-z][A-Za-z0-9]*)\b",
                result,
            )
        if not rules:
            return False
        values_by_label = {
            re.sub(r"[^a-z0-9]", "", label.lower()): value
            for label, value in rules
        }
        for output in requested_outputs:
            label = re.sub(r"[^a-z0-9]", "", output.lower())
            value = values_by_label.get(label)
            if value is not None and re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", value):
                return True
        return False

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