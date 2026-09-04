from typing import Any, Optional

from backend.providers.base import ComputationProvider

from .config import LocalLLMConfig
from .runtime import LocalLLMRuntime


class LocalLLMProvider(ComputationProvider):
    """
    Local AI provider.

    This provider is responsible for:
    - natural-language interpretation
    - mathematical problem decomposition
    - reasoning/planning
    - explanations
    - guided-learning assistance

    It is NOT the authoritative mathematical calculator.
    Wolfram Engine remains responsible for verified computation.
    """

    name = "local_llm"

    def __init__(
        self,
        config: Optional[LocalLLMConfig] = None,
    ):
        self.config = config or LocalLLMConfig()
        self.runtime = LocalLLMRuntime(self.config)

    def calculate(self, query: str) -> dict[str, Any]:
        """
        Provider interface compatibility.

        General computation should eventually go through the
        reasoning/orchestration layer rather than directly treating
        the LLM as a CAS.
        """
        return {
            "success": False,
            "result": None,
            "error": (
                "Local LLM is an AI reasoning provider, "
                "not a computational CAS."
            ),
            "engine": self.name,
        }

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> dict[str, Any]:
        try:
            result = self.runtime.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
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

    def is_available(self) -> bool:
        return (
            self.config.enabled
            and self.config.model_path is not None
        )