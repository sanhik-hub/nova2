import re
from typing import Optional

from rapidfuzz import fuzz

from .entities import ParsedQuery
from .intent import Intent


class LanguageParser:
    """
    Lightweight natural-language parser.

    The parser intentionally does not solve mathematics.
    Its job is:

        natural language
              ↓
        structured query

    Computation is handled later by SymPy/Wolfram.
    """

    INTENT_PATTERNS = {
        Intent.DIFFERENTIATE: [
            "differentiate",
            "differentiate with respect to",
            "find derivative",
            "find the derivative",
            "derivative of",
            "derivative",
            "differentation",
            "differentiation",
            "derive",
        ],

        Intent.INTEGRATE: [
            "integrate",
            "integration",
            "find integral",
            "find the integral",
            "integral of",
            "integral",
        ],

        Intent.LIMIT: [
            "limit",
            "find limit",
            "evaluate limit",
            "lim",
        ],

        Intent.SOLVE: [
            "solve",
            "solve for",
            "find solution",
            "find the solution",
            "solutions of",
            "solution of",
        ],

        Intent.ROOTS: [
            "roots",
            "find roots",
            "find the roots",
            "zeros",
            "find zeros",
            "find the zeros",
        ],

        Intent.SIMPLIFY: [
            "simplify",
            "simplify this",
            "simplify the expression",
        ],

        Intent.EXPAND: [
            "expand",
            "expand this",
            "expand the expression",
        ],

        Intent.FACTOR: [
            "factor",
            "factorise",
            "factorize",
            "factorise this",
            "factorize this",
        ],

        Intent.PLOT: [
            "plot",
            "plot this",
            "graph",
            "graph this",
            "draw graph",
            "show graph",
        ],

        Intent.MAXIMUM: [
            "maximum",
            "find maximum",
            "find the maximum",
            "max value",
            "greatest value",
        ],

        Intent.MINIMUM: [
            "minimum",
            "find minimum",
            "find the minimum",
            "min value",
            "smallest value",
        ],

        Intent.DETERMINANT: [
            "determinant",
            "find determinant",
            "det",
        ],

        Intent.INVERSE: [
            "inverse",
            "find inverse",
            "inverse matrix",
        ],

        Intent.MEAN: [
            "mean",
            "average",
            "arithmetic mean",
        ],

        Intent.MEDIAN: [
            "median",
            "find median",
        ],

        Intent.VARIANCE: [
            "variance",
            "find variance",
        ],

        Intent.STANDARD_DEVIATION: [
            "standard deviation",
            "std deviation",
            "standard dev",
        ],

        Intent.PROBABILITY: [
            "probability",
            "chance",
            "odds",
        ],

        Intent.EXPLAIN: [
            "explain",
            "explain this",
            "why",
            "how does this work",
            "what does this mean",
        ],

        Intent.ALTERNATIVE_METHOD: [
            "another way",
            "another method",
            "different method",
            "solve another way",
            "alternative method",
        ],

        Intent.GENERATE_MCQ: [
            "generate mcq",
            "make an mcq",
            "create an mcq",
            "give me an mcq",
            "make a multiple choice question",
            "create a multiple choice question",
        ],

        Intent.START_MCQ: [
            "start mcq",
            "start quiz",
            "start the quiz",
            "practice this",
            "practice",
            "quiz me",
        ],

        Intent.HINT: [
            "hint",
            "give me a hint",
            "help me",
            "give hint",
        ],

        Intent.SHOW_SOLUTION: [
            "show solution",
            "show the solution",
            "give me the solution",
            "show answer",
            "reveal answer",
        ],

        Intent.SIMILAR_PROBLEM: [
            "similar problem",
            "similar question",
            "another problem",
            "another question",
        ],

        Intent.HARDER_PROBLEM: [
            "harder problem",
            "harder question",
            "make it harder",
            "increase difficulty",
        ],

        Intent.EASIER_PROBLEM: [
            "easier problem",
            "easier question",
            "make it easier",
            "decrease difficulty",
        ],
    }

    def parse(self, text: str) -> ParsedQuery:

        original = text.strip()

        if not original:
            return ParsedQuery(original="", confidence=1.0)

        normalized = self._normalize_text(original)

        intent, confidence = self._detect_intent(normalized)

        query = ParsedQuery(
            original=original,
            intent=intent.value,
            confidence=confidence,
        )

        # ---------------------------------------------------------
        # Standard expression extraction
        # ---------------------------------------------------------
        self._extract_expression(query, normalized)

        # ---------------------------------------------------------
        # Reasoning / proof expression extraction
        #
        # Examples:
        #   prove that x^2 >= 0
        #   show that x^2 + 1 > 0
        #   why is x^2 nonnegative
        #   determine whether such a function can exist
        # ---------------------------------------------------------
        if intent == Intent.EXPLAIN and not query.expression:
            reasoning_text = normalized

            prefixes = [
                "prove that",
                "prove",
                "show that",
                "show why",
                "why does",
                "why is",
                "why are",
                "explain why",
                "determine whether",
                "determine if",
                "can there exist",
                "does there exist",
                "is it possible",
                "suppose",
                "assume",
                "given that",
            ]

            for prefix in prefixes:
                if reasoning_text.startswith(prefix):
                    expression = reasoning_text[len(prefix):].strip()

                    if expression:
                        query.expression = expression

                    break

        # ---------------------------------------------------------
        # Remaining entity extraction
        # ---------------------------------------------------------
        self._extract_variable(query, normalized)

        self._extract_bounds(query, normalized)

        self._extract_value(query, normalized)

        self._extract_conditions(query, normalized)

        self._detect_requested_output(query, normalized)

        return query

    # ---------------------------------------------------------
    # Normalization
    # ---------------------------------------------------------

    def _normalize_text(self, text: str) -> str:
        text = text.lower().strip()

        text = text.replace("²", "^2")
        text = text.replace("³", "^3")
        text = text.replace("⁴", "^4")

        text = re.sub(r"\s+", " ", text)

        return text

    # ---------------------------------------------------------
    # Intent detection
    # ---------------------------------------------------------

    def _detect_intent(self, text: str):
        """
        Detect the user's intent.

        Returns:
            (Intent, confidence)

        Important:
            This method returns an Intent enum, NOT intent.value.
            parse() converts the enum to its string value.
        """

        text_lower = text.lower().strip()

        # =========================================================
        # 1. EXPLICIT PROOF / REASONING
        # =========================================================
        #
        # These must be checked BEFORE fuzzy matching.
        #
        # Otherwise words such as "determine" can accidentally
        # become "determinant", etc.
        #
        reasoning_markers = [
            "prove",
            "prove that",
            "show that",
            "show why",
            "why does",
            "why is",
            "why are",
            "why can",
            "why cannot",
            "explain why",
            "determine whether",
            "determine if",
            "can there exist",
            "does there exist",
            "is there a",
            "is it possible",
            "suppose",
            "assume",
            "given that",
            "contradiction",
            "by contradiction",
            "if and only if",
            "iff",
        ]

        if any(marker in text_lower for marker in reasoning_markers):
            return Intent.EXPLAIN, 1.0

        # =========================================================
        # 2. FOLLOW-UP QUESTIONS
        # =========================================================
        #
        # These must beat fuzzy matching because a sentence such as
        # "what about at x=2" should not accidentally match another
        # mathematical operation.
        #
        follow_up_starts = [
            "what about",
            "what if",
            "how about",
            "and if",
            "then",
            "also",
            "now",
        ]

        if any(text_lower.startswith(start) for start in follow_up_starts):
            return Intent.FOLLOW_UP, 1.0

        # =========================================================
        # 3. EXPLICIT LEARNING / MCQ COMMANDS
        # =========================================================

        # Generate MCQ
        if (
            "generate an mcq" in text_lower
            or "generate a mcq" in text_lower
            or "generate mcq" in text_lower
            or "make an mcq" in text_lower
            or "make a mcq" in text_lower
            or "create an mcq" in text_lower
            or "create a mcq" in text_lower
            or "mcq from this" in text_lower
            or "mcq from it" in text_lower
        ):
            return Intent.GENERATE_MCQ, 1.0

        # Start MCQ / quiz
        if (
            "start mcq" in text_lower
            or "start an mcq" in text_lower
            or "start a quiz" in text_lower
            or "start quiz" in text_lower
            or "quiz me" in text_lower
            or "test me" in text_lower
            or "start step by step" in text_lower
            or "step by step mode" in text_lower
        ):
            return Intent.START_MCQ, 1.0

        # Next question
        if (
            text_lower == "next"
            or "next question" in text_lower
            or "next mcq" in text_lower
            or "give me the next question" in text_lower
            or "another question" in text_lower
        ):
            return Intent.NEXT_MCQ, 1.0

        # Hint
        if (
            text_lower == "hint"
            or "give me a hint" in text_lower
            or "give a hint" in text_lower
            or "need a hint" in text_lower
            or "help me with this" in text_lower
        ):
            return Intent.HINT, 1.0

        # Submit answer
        if (
            text_lower.startswith("my answer is")
            or text_lower.startswith("my answer:")
            or text_lower.startswith("answer is")
            or text_lower.startswith("i choose")
            or text_lower.startswith("i chose")
            or text_lower.startswith("option ")
            or text_lower.startswith("the answer is")
        ):
            return Intent.SUBMIT_ANSWER, 1.0

        # Show solution
        if (
            "show solution" in text_lower
            or "show the solution" in text_lower
            or "give me the solution" in text_lower
            or "show me the solution" in text_lower
            or "reveal solution" in text_lower
            or "reveal the solution" in text_lower
        ):
            return Intent.SHOW_SOLUTION, 1.0

        # Similar problem
        if (
            "similar problem" in text_lower
            or "give me a similar problem" in text_lower
            or "give another similar problem" in text_lower
            or "another problem like this" in text_lower
            or "another one like this" in text_lower
            or "another like this" in text_lower
        ):
            return Intent.SIMILAR_PROBLEM, 1.0

        # Harder
        if (
            "make it harder" in text_lower
            or "make this harder" in text_lower
            or "harder problem" in text_lower
            or "give me a harder problem" in text_lower
            or "increase the difficulty" in text_lower
        ):
            return Intent.HARDER_PROBLEM, 1.0

        # Easier
        if (
            "make it easier" in text_lower
            or "make this easier" in text_lower
            or "easier problem" in text_lower
            or "give me an easier problem" in text_lower
            or "decrease the difficulty" in text_lower
        ):
            return Intent.EASIER_PROBLEM, 1.0

        # =========================================================
        # 4. EXPLICIT EXPLANATION
        # =========================================================

        if (
            text_lower.startswith("explain")
            or text_lower.startswith("explain the")
            or text_lower.startswith("explain how")
            or text_lower.startswith("explain why")
            or "explain this" in text_lower
            or "explain it" in text_lower
            or "why did you" in text_lower
            or "why was" in text_lower
        ):
            return Intent.EXPLAIN, 1.0

        # =========================================================
        # 5. ALTERNATIVE METHOD
        # =========================================================

        if (
            "another method" in text_lower
            or "alternative method" in text_lower
            or "different method" in text_lower
            or "solve another way" in text_lower
            or "solve it another way" in text_lower
            or "different way" in text_lower
        ):
            return Intent.ALTERNATIVE_METHOD, 1.0

        # =========================================================
        # 6. DIRECT COMPUTATIONAL INTENTS
        # =========================================================
        #
        # These are checked explicitly before fuzzy matching.
        # This makes common mathematical commands reliable.
        # =========================================================

        # Differentiate
        if (
            "differentiate" in text_lower
            or "derivative" in text_lower
            or "differentiate with respect to" in text_lower
            or "find d/d" in text_lower
            or "find the derivative" in text_lower
            or "calculate the derivative" in text_lower
        ):
            return Intent.DIFFERENTIATE, 1.0

        # Integrate
        if (
            "integrate" in text_lower
            or "integral" in text_lower
            or "integration" in text_lower
            or "find the integral" in text_lower
            or "calculate the integral" in text_lower
        ):
            return Intent.INTEGRATE, 1.0

        # Limit
        if (
            "limit" in text_lower
            or "lim " in text_lower
            or "find the limit" in text_lower
        ):
            return Intent.LIMIT, 1.0

        # Solve
        if (
            text_lower.startswith("solve ")
            or text_lower.startswith("solve for")
            or "solve the equation" in text_lower
            or "solve equation" in text_lower
            or "find the solution" in text_lower
            or "solutions of" in text_lower
        ):
            return Intent.SOLVE, 1.0

        # Roots
        if (
            "roots of" in text_lower
            or "find the roots" in text_lower
            or "find roots" in text_lower
            or "root of" in text_lower
        ):
            return Intent.ROOTS, 1.0

        # Simplify
        if (
            "simplify" in text_lower
            or "simplification" in text_lower
        ):
            return Intent.SIMPLIFY, 1.0

        # Expand
        if (
            "expand" in text_lower
            or "expansion of" in text_lower
        ):
            return Intent.EXPAND, 1.0

        # Factor
        if (
            "factor" in text_lower
            or "factorise" in text_lower
            or "factorize" in text_lower
            or "factorisation" in text_lower
            or "factorization" in text_lower
        ):
            return Intent.FACTOR, 1.0

        # Plot
        if (
            "plot" in text_lower
            or "graph" in text_lower
            or "draw the graph" in text_lower
            or "show the graph" in text_lower
            or "plot the function" in text_lower
        ):
            return Intent.PLOT, 1.0

        # Maximum
        if (
            "maximum" in text_lower
            or "maxima" in text_lower
            or "maximize" in text_lower
            or "greatest value" in text_lower
        ):
            return Intent.MAXIMUM, 1.0

        # Minimum
        if (
            "minimum" in text_lower
            or "minima" in text_lower
            or "minimize" in text_lower
            or "least value" in text_lower
            or "smallest value" in text_lower
        ):
            return Intent.MINIMUM, 1.0

        # Determinant
        if (
            "determinant" in text_lower
            or "det of" in text_lower
            or "det(" in text_lower
        ):
            return Intent.DETERMINANT, 1.0

        # Matrix
        if (
            "matrix" in text_lower
            or "matrices" in text_lower
        ):
            return Intent.MATRIX, 1.0

        # Inverse
        if (
            "inverse matrix" in text_lower
            or "inverse of the matrix" in text_lower
            or "matrix inverse" in text_lower
        ):
            return Intent.INVERSE, 1.0

        # Mean
        if (
            "mean" in text_lower
            or "average" in text_lower
            or "arithmetic mean" in text_lower
        ):
            return Intent.MEAN, 1.0

        # Median
        if (
            "median" in text_lower
            or "middle value" in text_lower
        ):
            return Intent.MEDIAN, 1.0

        # Variance
        if (
            "variance" in text_lower
            or "var(" in text_lower
        ):
            return Intent.VARIANCE, 1.0

        # Standard deviation
        if (
            "standard deviation" in text_lower
            or "std deviation" in text_lower
            or "stdev" in text_lower
            or "std(" in text_lower
        ):
            return Intent.STANDARD_DEVIATION, 1.0

        # Probability
        if (
            "probability" in text_lower
            or "probability of" in text_lower
            or "chance of" in text_lower
            or "odds of" in text_lower
        ):
            return Intent.PROBABILITY, 1.0

        # =========================================================
        # 7. FALLBACK TO EXISTING FUZZY MATCHING
        # =========================================================
        #
        # Keep this section compatible with your existing
        # INTENT_PATTERNS and RapidFuzz implementation.
        # =========================================================

        best_intent = Intent.UNKNOWN
        best_score = 0.0

        for intent, patterns in self.INTENT_PATTERNS.items():

            for pattern in patterns:
                pattern_lower = pattern.lower()

                # Exact substring match
                if pattern_lower in text_lower:
                    score = 0.90
                else:
                    # Existing RapidFuzz logic
                    from rapidfuzz.fuzz import partial_ratio

                    score = partial_ratio(
                        pattern_lower,
                        text_lower,
                    ) / 100.0

                if score > best_score:
                    best_score = score
                    best_intent = intent

        # =========================================================
        # 8. FUZZY MATCH THRESHOLD
        # =========================================================

        if best_score >= 0.70:
            return best_intent, best_score

        return Intent.UNKNOWN, best_score
    # ---------------------------------------------------------
    # Expression extraction
    # ---------------------------------------------------------

    def _extract_expression(
        self,
        query: ParsedQuery,
        text: str,
    ) -> None:

        expression = None

        patterns = [
            r"(?:derivative of|differentiate)\s+(.+)",
            r"(?:integral of|integrate)\s+(.+)",
            r"(?:limit of|limit)\s+(.+)",
            r"(?:solve|solve for)\s+(.+)",
            r"(?:roots? of|zeros? of)\s+(.+)",
            r"(?:simplify)\s+(.+)",
            r"(?:expand)\s+(.+)",
            r"(?:factor(?:ise|ize)?)\s+(.+)",
            r"(?:plot|graph)\s+(.+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)

            if match:
                expression = match.group(1).strip()
                break

        if expression:
            # Remove common trailing natural-language clauses.
            expression = re.split(
                r"\s+(?:with respect to|where|at|from|between)\s+",
                expression,
                maxsplit=1,
            )[0].strip()

            query.expression = expression

    # ---------------------------------------------------------
    # Variable extraction
    # ---------------------------------------------------------

    def _extract_variable(
        self,
        query: ParsedQuery,
        text: str,
    ) -> None:

        patterns = [
            r"with respect to\s+([a-zA-Z]\w*)",
            r"respect to\s+([a-zA-Z]\w*)",
            r"wrt\s+([a-zA-Z]\w*)",
            r"d/d([a-zA-Z]\w*)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)

            if match:
                query.variable = match.group(1)
                return

        # Mathematical default.
        if query.intent in {
            Intent.DIFFERENTIATE.value,
            Intent.INTEGRATE.value,
            Intent.LIMIT.value,
            Intent.SOLVE.value,
        }:
            query.variable = "x"

    # ---------------------------------------------------------
    # Bounds
    # ---------------------------------------------------------

    def _extract_bounds(
        self,
        query: ParsedQuery,
        text: str,
    ) -> None:

        patterns = [
            r"from\s+(.+?)\s+to\s+(.+?)(?:\s|$)",
            r"between\s+(.+?)\s+and\s+(.+?)(?:\s|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)

            if match:
                query.lower_bound = match.group(1).strip()
                query.upper_bound = match.group(2).strip()
                return

    # ---------------------------------------------------------
    # Evaluation value
    # ---------------------------------------------------------

    def _extract_value(
        self,
        query: ParsedQuery,
        text: str,
    ) -> None:

        match = re.search(
            r"\bat\s+([a-zA-Z]\w*)\s*=\s*([^\s,]+)",
            text,
        )

        if match:
            query.variable = match.group(1)
            query.value = match.group(2)
            return

        match = re.search(
            r"\b([a-zA-Z]\w*)\s*=\s*([^\s,]+)",
            text,
        )

        if match:
            query.parameters[match.group(1)] = match.group(2)

    # ---------------------------------------------------------
    # Conditions
    # ---------------------------------------------------------

    def _extract_conditions(
        self,
        query: ParsedQuery,
        text: str,
    ) -> None:

        condition_patterns = [
            r"where\s+(.+)",
            r"given\s+(.+)",
            r"assuming\s+(.+)",
            r"subject to\s+(.+)",
        ]

        for pattern in condition_patterns:
            match = re.search(pattern, text)

            if match:
                query.conditions.append(match.group(1).strip())

    # ---------------------------------------------------------
    # Requested output
    # ---------------------------------------------------------

    def _detect_requested_output(
        self,
        query: ParsedQuery,
        text: str,
    ) -> None:

        if any(word in text for word in [
            "step",
            "steps",
            "step by step",
            "explain",
        ]):
            query.requested_output.append("steps")

        if any(word in text for word in [
            "graph",
            "plot",
        ]):
            query.requested_output.append("plot")

        if "answer only" in text:
            query.requested_output.append("answer")

        if "mcq" in text or "multiple choice" in text:
            query.requested_output.append("mcq")