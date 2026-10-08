from typing import Any, Optional

from backend.providers.base import ComputationProvider

from .config import LocalLLMConfig
from .runtime import LocalLLMRuntime


class LocalLLMProvider(ComputationProvider):

    name = "local_llm"

    def __init__(
        self,
        config: Optional[LocalLLMConfig] = None,
    ):
        self.config = config or LocalLLMConfig()
        self.runtime = LocalLLMRuntime(self.config)

    def calculate(self, query: str) -> dict[str, Any]:
        return {
            "success": False,
            "result": None,
            "error": (
                "Local LLM is a reasoning provider, "
                "not the authoritative computational engine."
            ),
            "engine": self.name,
        }

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        enable_thinking: bool = False,
        response_format: Optional[dict[str, Any]] = None,
        request_timeout: Optional[int] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> dict[str, Any]:

        try:
            result = self.runtime.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
                enable_thinking=enable_thinking,
                response_format=response_format,
                request_timeout=request_timeout,
                tools=tools,
                tool_choice=tool_choice,
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
        if not self.config.enabled:
            return False

        try:
            self.runtime.load()
            return True
        except Exception:
            return False