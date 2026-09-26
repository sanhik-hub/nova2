from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class ParsedQuery:
    """
    Structured representation of a user's mathematical request.
    """

    original: str

    intent: str = "unknown"

    expression: Optional[str] = None
    variable: Optional[str] = None

    value: Optional[str] = None

    lower_bound: Optional[str] = None
    upper_bound: Optional[str] = None

    conditions: list[str] = field(default_factory=list)

    parameters: dict[str, str] = field(default_factory=dict)

    requested_output: list[str] = field(default_factory=list)

    confidence: float = 0.0

    ambiguous: bool = False

    alternatives: list[dict[str, Any]] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)