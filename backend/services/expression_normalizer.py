import re
from dataclasses import dataclass


# ---------------------------------------------------------
# Normalization result
# ---------------------------------------------------------

@dataclass
class NormalizationResult:
    original: str
    normalized: str
    changed: bool
    confidence: str
    requires_llm: bool


# ---------------------------------------------------------
# Wolfram function mappings
# ---------------------------------------------------------

FUNCTION_MAP = {
    "sin": "Sin",
    "cos": "Cos",
    "tan": "Tan",
    "cot": "Cot",
    "sec": "Sec",
    "csc": "Csc",

    "asin": "ArcSin",
    "arcsin": "ArcSin",
    "acos": "ArcCos",
    "arccos": "ArcCos",
    "atan": "ArcTan",
    "arctan": "ArcTan",
    "acot": "ArcCot",
    "arcot": "ArcCot",
    "asec": "ArcSec",
    "arcsec": "ArcSec",
    "acsc": "ArcCsc",
    "arccsc": "ArcCsc",

    "sinh": "Sinh",
    "cosh": "Cosh",
    "tanh": "Tanh",
    "coth": "Coth",

    "asinh": "ArcSinh",
    "acosh": "ArcCosh",
    "atanh": "ArcTanh",

    "sqrt": "Sqrt",
    "exp": "Exp",
    "log": "Log",
    "ln": "Log",

    "abs": "Abs",
    "floor": "Floor",
    "ceil": "Ceiling",
    "ceiling": "Ceiling",
    "sign": "Sign",

    "factorial": "Factorial",
}


# ---------------------------------------------------------
# Mathematical constants
# ---------------------------------------------------------

CONSTANT_MAP = {
    "pi": "Pi",
    "π": "Pi",
    "∞": "Infinity",
    "infinity": "Infinity",
}


# ---------------------------------------------------------
# Unicode operators / symbols
# ---------------------------------------------------------

SYMBOL_MAP = {
    "×": "*",
    "·": "*",
    "÷": "/",

    "−": "-",
    "–": "-",
    "—": "-",

    "≤": "<=",
    "≥": ">=",
    "≠": "!=",

    "∈": "∈",
    "±": "+/-",

    "∞": "Infinity",

    "√": "sqrt",
}


# ---------------------------------------------------------
# Unicode superscripts
# ---------------------------------------------------------

SUPERSCRIPT_MAP = str.maketrans({
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "⁺": "+",
    "⁻": "-",
})


# ---------------------------------------------------------
# Basic symbol normalization
# ---------------------------------------------------------

def _normalize_symbols(expression: str) -> str:
    result = expression

    for source, target in SYMBOL_MAP.items():
        result = result.replace(source, target)

    return result


# ---------------------------------------------------------
# Superscript normalization
# ---------------------------------------------------------

def _normalize_superscripts(expression: str) -> str:
    """
    Examples:

        x²       -> x^2
        x³       -> x^3
        x⁻²      -> x^-2
        (x+1)²   -> (x+1)^2
    """

    if not any(char in expression for char in "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻"):
        return expression

    result = []
    i = 0

    while i < len(expression):
        char = expression[i]

        if char in "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻":
            superscript = []

            while i < len(expression) and expression[i] in "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻":
                superscript.append(expression[i])
                i += 1

            power = "".join(superscript).translate(SUPERSCRIPT_MAP)

            # Find previous mathematical atom.
            if result:
                if result[-1] == ")":
                    # (x+1)²
                    result.append("^" + power)
                else:
                    # x²
                    result.append("^" + power)
            else:
                result.append("^" + power)

            continue

        result.append(char)
        i += 1

    return "".join(result)


# ---------------------------------------------------------
# Function normalization
# ---------------------------------------------------------

def _normalize_functions(expression: str) -> str:
    result = expression

    function_names = sorted(
        FUNCTION_MAP.keys(),
        key=len,
        reverse=True,
    )

    # Functions may appear:
    # sin(x)
    # 2sin(x)
    # xsin(x)
    # sqrt(x)
    #
    # We require a function name followed by '('.
    pattern = re.compile(
        r"("
        + "|".join(map(re.escape, function_names))
        + r")\s*\(",
        re.IGNORECASE,
    )

    while True:
        match = pattern.search(result)

        if not match:
            break

        start = match.start()
        open_paren = match.end() - 1

        depth = 0
        close_paren = None

        for i in range(open_paren, len(result)):
            if result[i] == "(":
                depth += 1
            elif result[i] == ")":
                depth -= 1

                if depth == 0:
                    close_paren = i
                    break

        if close_paren is None:
            break

        name = match.group(1).lower()
        wolfram_name = FUNCTION_MAP[name]

        argument = result[open_paren + 1:close_paren]

        replacement = f"{wolfram_name}[{argument}]"

        result = (
            result[:start]
            + replacement
            + result[close_paren + 1:]
        )

    return result
# ---------------------------------------------------------
# Constant normalization
# ---------------------------------------------------------

def _normalize_constants(expression: str) -> str:
    result = expression

    result = re.sub(
        r"(?<![A-Za-z])π(?![A-Za-z])",
        "Pi",
        result,
    )

    result = re.sub(
        r"(?<![A-Za-z])pi(?![A-Za-z])",
        "Pi",
        result,
        flags=re.IGNORECASE,
    )

    result = re.sub(
        r"(?<![A-Za-z])infinity(?![A-Za-z])",
        "Infinity",
        result,
        flags=re.IGNORECASE,
    )

    result = result.replace("∞", "Infinity")

    # Mathematical e.
    result = re.sub(
        r"(?<![A-Za-z])e(?![A-Za-z])",
        "E",
        result,
    )

    return result


# ---------------------------------------------------------
# Bracket normalization
# ---------------------------------------------------------

def _normalize_brackets(expression: str) -> str:
    """
    Convert mathematical curly/angle grouping where appropriate.

    {x} -> (x)
    """

    result = expression

    result = result.replace("{", "(")
    result = result.replace("}", ")")

    return result


# ---------------------------------------------------------
# Implicit multiplication
# ---------------------------------------------------------

def _normalize_multiplication(expression: str) -> str:
    result = expression

    # Number followed by a variable:
    # 2x -> 2*x
    result = re.sub(
        r"(\d)([A-Za-z])",
        r"\1*\2",
        result,
    )

    # Number followed by an opening parenthesis:
    # 2(x+1) -> 2*(x+1)
    result = re.sub(
        r"(\d)\s*\(",
        r"\1*(",
        result,
    )

    # Closing parenthesis followed by opening parenthesis:
    # (x+1)(x-1) -> (x+1)*(x-1)
    result = re.sub(
        r"\)\s*\(",
        ")*(",
        result,
    )

    # Single variable followed by an opening parenthesis:
    # x(x+1) -> x*(x+1)
    result = re.sub(
        r"(?<![A-Za-z])([A-Za-z])\s*\(",
        r"\1*(",
        result,
    )

    # Number followed by a Wolfram function:
    # 2Sin[x] -> 2*Sin[x]
    result = re.sub(
        r"(\d)\s*([A-Z][A-Za-z]*)\[",
        r"\1*\2[",
        result,
    )

    # Variable followed directly by a Wolfram function:
    # xSin[x] -> x*Sin[x]
    result = re.sub(
        r"([a-zA-Z])(?=[A-Z][A-Za-z]*\[)",
        r"\1*",
        result,
    )

    # Variable followed by a function with whitespace:
    # x Sin[x] -> x*Sin[x]
    result = re.sub(
        r"([a-zA-Z])\s+([A-Z][A-Za-z]*)\[",
        r"\1*\2[",
        result,
    )

    return result
# ---------------------------------------------------------
# Equation / comparison normalization
# ---------------------------------------------------------

def _normalize_comparisons(expression: str) -> str:
    """
    Convert common comparison notation.

        x = 2     -> x == 2
        x ≤ 2     -> x <= 2
        x ≥ 2     -> x >= 2
        x ≠ 2     -> x != 2

    Existing == is preserved.
    """

    result = expression

    result = re.sub(
        r"(?<![<>=!])=(?!=)",
        "==",
        result,
    )

    return result


# ---------------------------------------------------------
# Whitespace
# ---------------------------------------------------------

def _normalize_whitespace(expression: str) -> str:
    result = re.sub(r"\s+", " ", expression)
    return result.strip()


# ---------------------------------------------------------
# Determine whether LLM is needed
# ---------------------------------------------------------

def _looks_like_mathematical_expression(expression: str) -> bool:
    """
    Conservative classifier.

    True means the input looks sufficiently mathematical that
    deterministic normalization can handle it.

    False means natural-language interpretation is probably needed.
    """

    if not expression:
        return False

    text = expression.strip()

    # Obvious natural-language indicators.
    natural_language_patterns = [
        r"\bfind\b",
        r"\bcalculate\b",
        r"\bsolve\b",
        r"\bevaluate\b",
        r"\bdetermine\b",
        r"\bprove\b",
        r"\bshow\b",
        r"\bexplain\b",
        r"\bwhat\b",
        r"\bwhy\b",
        r"\bhow\b",
        r"\bmaximum\b",
        r"\bminimum\b",
        r"\barea\b",
        r"\bvolume\b",
        r"\bdistance\b",
        r"\bvelocity\b",
        r"\bacceleration\b",
        r"\bprobability\b",
        r"\bderivative\b",
        r"\bintegral\b",
        r"\blimit\b",
        r"\bbetween\b",
        r"\bsubject to\b",
        r"\bwhere\b",
    ]

    lowered = text.lower()

    for pattern in natural_language_patterns:
        if re.search(pattern, lowered):
            return False

    # If there are alphabetic words separated by spaces,
    # it is probably natural language.
    words = re.findall(r"[A-Za-z]+", text)

    if len(words) > 3:
        return False

    return True


# ---------------------------------------------------------
# Main normalizer
# ---------------------------------------------------------

def normalize_expression_result(expression: str) -> NormalizationResult:
    if expression is None:
        return NormalizationResult(
            original="",
            normalized="",
            changed=False,
            confidence="low",
            requires_llm=True,
        )

    original = expression.strip()

    if not original:
        return NormalizationResult(
            original=expression,
            normalized="",
            changed=False,
            confidence="low",
            requires_llm=True,
        )

    result = original

    result = _normalize_symbols(result)
    result = _normalize_brackets(result)
    result = _normalize_superscripts(result)
    result = _normalize_constants(result)
    result = _normalize_functions(result)
    result = _normalize_comparisons(result)
    result = _normalize_multiplication(result)
    result = _normalize_whitespace(result)

    mathematical = _looks_like_mathematical_expression(original)

    return NormalizationResult(
        original=original,
        normalized=result,
        changed=(original != result),
        confidence="high" if mathematical else "low",
        requires_llm=not mathematical,
    )


def normalize_expression(expression: str) -> str:
    """
    Backwards-compatible helper.

    Existing code can continue doing:

        normalize_expression("sin(x)")

    and receive:

        Sin[x]
    """

    return normalize_expression_result(expression).normalized