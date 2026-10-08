from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ComplexityAssessment:
    level: str
    score: int
    thinking: bool
    signals: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "score": self.score,
            "thinking": self.thinking,
            "signals": list(self.signals),
        }


class ComplexityClassifier:
    """Deterministic, domain-agnostic complexity classifier."""

    THINKING_THRESHOLD = 6

    def analyze(self, problem: str) -> ComplexityAssessment:
        text = re.sub(r"\s+", " ", str(problem or "").strip())
        if not text:
            return ComplexityAssessment("trivial", 0, False, ("empty_request",))

        score = 0
        signals: list[str] = []
        words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", text)
        word_count = len(words)

        number_count = len(re.findall(
            r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text
        ))
        equation_count = len(re.findall(r"(?<![<>])=(?!=)", text))
        operator_count = len(re.findall(r"[+*/^]|(?<!\w)-(?!\w)", text))
        grouped_count = len(re.findall(r"\([^()]{1,80}\)", text))
        question_count = text.count("?")

        request_count = len(re.findall(
            r"\b(?:find|calculate|determine|derive|prove|show|evaluate|"
            r"compare|explain|obtain|solve|compute|identify|classify|"
            r"estimate|verify|check)\b",
            text,
            re.IGNORECASE,
        ))
        chain_count = len(re.findall(
            r"\b(?:then|after|next|followed by|given that|assuming that|"
            r"such that|whereas|while|if|when)\b",
            text,
            re.IGNORECASE,
        ))
        relation_count = len(re.findall(
            r"\b(?:because|therefore|hence|thus|so that|in order to|"
            r"depends on|as a result|provided that|subject to)\b",
            text,
            re.IGNORECASE,
        ))
        verification_count = len(re.findall(
            r"\b(?:verify|verification|validate|validation|check|checking|"
            r"substitut(?:e|ing)|confirm|confirming|test|testing)\b",
            text,
            re.IGNORECASE,
        ))
        constraint_count = len(re.findall(
            r"\b(?:constraint|constraints|condition|conditions|unknown|"
            r"unknowns|equation|equations|requirement|requirements)\b",
            text,
            re.IGNORECASE,
        ))
        unit_count = len(re.findall(
            r"(?<![A-Za-z])\d+(?:\.\d+)?\s*(?:[A-Za-zµμΩ]+)(?![A-Za-z])",
            text,
        ))
        symbolic_count = len(re.findall(
            r"(?:\b[A-Za-z]\b|[∫∑√∞∂Δπ]|\^|_|\$)", text
        ))

        if word_count > 28:
            score += 2
            signals.append("long_request")
        elif word_count > 16:
            score += 1
            signals.append("moderate_request_length")

        if number_count >= 4:
            score += 2
            signals.append("many_explicit_quantities")
        elif number_count >= 2:
            score += 1
            signals.append("multiple_explicit_quantities")

        if equation_count >= 2:
            score += 2
            signals.append("multiple_equations")
        elif equation_count == 1:
            score += 1
            signals.append("equation_constraint")

        if operator_count >= 5:
            score += 2
            signals.append("dense_symbolic_structure")
        elif operator_count >= 2:
            score += 1
            signals.append("symbolic_structure")

        if grouped_count >= 2:
            score += 1
            signals.append("nested_grouping")

        if request_count >= 3 or question_count >= 2:
            score += 2
            signals.append("multiple_requested_outputs")
        elif request_count >= 2:
            score += 1
            signals.append("multiple_actions")

        if chain_count >= 2:
            score += 2
            signals.append("multi_stage_dependencies")
        elif chain_count == 1:
            score += 1
            signals.append("conditional_or_sequential_dependency")

        if relation_count >= 2:
            score += 1
            signals.append("explicit_reasoning_relationships")

        # Explicit verification adds a separate reasoning stage and therefore
        # materially increases structural complexity.
        if verification_count >= 1:
            score += 2
            signals.append("explicit_verification")

        # Natural-language constraints/unknowns can represent coupled reasoning
        # even when no symbolic equations are present.
        if constraint_count >= 3:
            score += 1
            signals.append("multiple_constraints_or_unknowns")

        if unit_count >= 3:
            score += 1
            signals.append("multiple_quantities_with_units")

        if symbolic_count >= 6:
            score += 1
            signals.append("symbolic_density")

        if word_count <= 8 and request_count <= 1 and chain_count == 0 and equation_count <= 1:
            score = min(score, 2)

        if score >= self.THINKING_THRESHOLD:
            level, thinking = "high", True
        elif score >= 3:
            level, thinking = "moderate", False
        else:
            level, thinking = "simple", False

        return ComplexityAssessment(level, score, thinking, tuple(signals))
