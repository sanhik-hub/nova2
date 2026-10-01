from typing import Any, Optional

from .config import PlannerLLMConfig
from .runtime import PlannerLLMRuntime


class PlannerLLMProvider:
    name = "planner_llm"

    def __init__(self, config: Optional[PlannerLLMConfig] = None):
        self.config = config or PlannerLLMConfig()
        self.runtime = PlannerLLMRuntime(self.config)

    def plan(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        try:
            result = self.runtime.generate_json(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
            )
            return {
                "success": True,
                "result": result,
                "error": None,
                "engine": self.name,
            }
        except Exception as exc:
            return {
                "success": False,
                "result": None,
                "error": str(exc),
                "engine": self.name,
            }
