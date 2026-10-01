from typing import Optional
from pydantic import BaseModel, Field


class GuidedAttemptRequest(BaseModel):
    query: str = Field(..., min_length=1)
    include_hints: bool = False
    include_full_solution: bool = False
    step_count: int | None = Field(default=None, ge=2, le=8)


class GuidedOption(BaseModel):
    text: str


class GuidedStep(BaseModel):
    question: str
    options: list[GuidedOption] = Field(min_length=2, max_length=4)
    correct_index: int = Field(ge=0, le=3)
    hint: Optional[str] = None


class GuidedAttemptResponse(BaseModel):
    success: bool
    query: str
    steps: list[GuidedStep] = []
    include_hints: bool = False
    include_full_solution: bool = False
    full_solution: Optional[str] = None
    error: Optional[str] = None
