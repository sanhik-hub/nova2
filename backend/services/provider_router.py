from backend.providers.wolfram.provider import WolframProvider
from backend.providers.sympy.provider import SymPyProvider
from backend.providers.local_llm.provider import LocalLLMProvider


class ProviderRouter:

    def __init__(self):

        self.providers = {
            "wolfram": WolframProvider(),
            "sympy": SymPyProvider(),
            "local_llm": LocalLLMProvider(),
        }

    def get_provider(self, name: str):

        provider = self.providers.get(name)

        if provider is None:
            raise ValueError(
                f"Unknown provider: {name}"
            )

        return provider

    def calculate(
        self,
        query: str,
        provider: str = "wolfram",
    ):

        selected = self.get_provider(provider)

        return selected.calculate(query)

    def generate(
        self,
        prompt: str,
        max_tokens=None,
        temperature=None,
        system_prompt=None,
        enable_thinking=False,
    ):

        provider = self.get_provider("local_llm")

        return provider.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            enable_thinking=enable_thinking,
        )