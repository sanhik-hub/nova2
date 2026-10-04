from backend.providers.wolfram.provider import WolframProvider
from backend.providers.sympy.provider import SymPyProvider
from backend.providers.local_llm.provider import LocalLLMProvider
from backend.providers.planner_llm.provider import PlannerLLMProvider


class ProviderRouter:

    def __init__(self):

        self.providers = {
            "wolfram": WolframProvider(),
            "sympy": SymPyProvider(),
            "local_llm": LocalLLMProvider(),
            "planner_llm": PlannerLLMProvider(),
        }

    def get_provider(self, name: str):

        provider = self.providers.get(name)

        if provider is None:
            raise ValueError(
                f"Unknown provider: {name}"
            )

        return provider

    def plan(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int | None = None,
        compact: bool = False,
        request_timeout: int | None = None,
    ):
        return self.get_provider("planner_llm").plan(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            compact=compact,
            request_timeout=request_timeout,
        )

    def calculate(
        self,
        query: str | None = None,
        provider: str = "wolfram",
        expression: str | None = None,
    ):
        """Run an exact calculation through the selected provider.

        ``query`` is canonical; ``expression`` is a compatibility alias.
        """
        if query is None:
            query = expression
        if query is None or not str(query).strip():
            raise ValueError("A calculation query is required.")
        selected = self.get_provider(provider)
        return selected.calculate(str(query))

    def generate(
        self,
        prompt: str,
        max_tokens=None,
        temperature=None,
        system_prompt=None,
        enable_thinking=False,
        response_format=None,
        request_timeout=None,
    ):

        provider = self.get_provider("local_llm")

        return provider.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            enable_thinking=enable_thinking,
            response_format=response_format,
            request_timeout=request_timeout,
        )