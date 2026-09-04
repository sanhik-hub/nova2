from pydantic import BaseModel, Field
from typing import Any, Optional


class ComputationRequest(BaseModel):
    query: str = Field(..., min_length=1)
    operation: str = "compute"
    provider: str = "wolfram"


class ExpressionRequest(BaseModel):
    expression: str = Field(..., min_length=1)
    variable: str = "x"
    provider: str = "wolfram"


class LimitRequest(BaseModel):
    expression: str = Field(..., min_length=1)
    variable: str = "x"
    value: str
    provider: str = "wolfram"


class ComputationResponse(BaseModel):
    success: bool
    operation: str
    input: str

    result: Optional[str] = None
    error: Optional[str] = None

    engine: Optional[str] = None

    steps: list[Any] = []
    plots: list[Any] = []
    related: list[Any] = []

    
class PlotRequest(BaseModel):
    expression: str = Field(..., min_length=1)
    variable: str = "x"
    xmin: float = -10
    xmax: float = 10
    provider: str = "wolfram"