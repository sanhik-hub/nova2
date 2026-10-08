import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class LocalLLMConfig:

    enabled: bool = True

    model_path: str = os.getenv(
        "LOCAL_LLM_MODEL",
        "models/chemistry_test/base/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    )

    # llama.cpp server
    server_url: str = os.getenv(
        "LOCAL_LLM_SERVER",
        "http://127.0.0.1:8080",
    )

    # CPU-first defaults. A device-local benchmark profile may override
    # runtime tuning values without changing the repository configuration.
    n_gpu_layers: int = 0
    context_size: int = 32768
    threads: int = 6

    # Transport safety fallback for llama.cpp. This is not a solution-length
    # policy: callers may explicitly request a different value. It prevents a
    # malformed local generation from running indefinitely when a caller does
    # not provide a provider-level completion bound.
    max_tokens: int | None = 8192

    # Deterministic reasoning.
    temperature: float = 0.0

    batch_size: int = 128

    # Transport safety fallback for the local HTTP request. This is only a
    # connection/generation watchdog; it does not impose a solver-side time
    # budget when the caller supplies its own timeout.
    request_timeout: int | None = 300

    def __post_init__(self) -> None:
        profile_path = (
            Path(__file__).resolve().parents[3]
            / ".nova"
            / "llama_benchmark.json"
        )

        try:
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            tuned = profile.get("config", {})
        except (FileNotFoundError, OSError, json.JSONDecodeError, AttributeError):
            return

        if not isinstance(tuned, dict):
            return

        for field in (
            "n_gpu_layers",
            "context_size",
            "threads",
            "batch_size",
        ):
            value = tuned.get(field)
            if isinstance(value, int) and value >= 0:
                setattr(self, field, value)


def main() -> int:
    """Print the effective configuration for launch scripts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the effective configuration as JSON",
    )
    arguments = parser.parse_args()

    if not arguments.json:
        parser.print_help()
        return 0

    print(json.dumps(asdict(LocalLLMConfig())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
