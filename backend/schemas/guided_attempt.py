from typing import Optional
from pydantic import BaseModel, Field


class GuidedAttemptRequest(BaseModel):
    query: str = Field(..., min_length=1)
    include_hints: bool = True
    include_full_solution: bool = True
    step_count: int = Field(default=4, ge=3, le=6)


class GuidedOption(BaseModel):
    text: str


class GuidedStep(BaseModel):
    question: str
    options: list[GuidedOption] = Field(min_length=2, max_length=4)
    correct_index: int = Field(ge=0, le=3)
    hint: Optional[str] = None
    explanation: str


class GuidedAttemptResponse(BaseModel):
    success: bool
    query: str
    steps: list[GuidedStep] = []
    include_hints: bool = True
    include_full_solution: bool = True
    full_solution: Optional[str] = None
    error: Optional[str] = None
