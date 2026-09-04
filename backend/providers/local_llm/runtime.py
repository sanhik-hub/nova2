from typing import Optional

from .config import LocalLLMConfig


class LocalLLMRuntime:
    """
    Thin wrapper around the local LLM runtime.

    The actual llama.cpp implementation will be plugged in here.
    Keeping it isolated means we can change models/runtimes later
    without changing the rest of the backend.
    """

    def __init__(self, config: Optional[LocalLLMConfig] = None):
        self.config = config or LocalLLMConfig()
        self._model = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """
        Load the local model.

        The actual llama.cpp loading code will be added after
        we benchmark the appropriate model/runtime.
        """
        if not self.config.enabled:
            raise RuntimeError("Local LLM is disabled.")

        if self.loaded:
            return

        raise NotImplementedError(
            "Local LLM runtime is not installed yet."
        )

    def unload(self) -> None:
        """Release the model from memory."""
        self._model = None

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        if not self.loaded:
            self.load()

        raise NotImplementedError(
            "Local LLM generation is not implemented yet."
        )