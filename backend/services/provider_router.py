from backend.providers.wolfram.provider import WolframProvider
from backend.providers.sympy.provider import SymPyProvider


class ProviderRouter:

    def __init__(self):
        self.providers = {
            "wolfram": WolframProvider(),
            "sympy": SymPyProvider(),
        }

    def get_provider(self, name: str):
        provider = self.providers.get(name)

        if provider is None:
            raise ValueError(f"Unknown provider: {name}")

        return provider

    def calculate(self, query: str, provider: str = "wolfram"):
        selected = self.get_provider(provider)
        return selected.calculate(query)