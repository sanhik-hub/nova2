import argparse
import json
import os
from dataclasses import asdict, dataclass


@dataclass
class PlannerLLMConfig:
    enabled: bool = True
    model_path: str = os.getenv(
        "PLANNER_LLM_MODEL",
        "models/local/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
    )
    server_url: str = os.getenv(
        "PLANNER_LLM_SERVER",
        "http://127.0.0.1:8081",
    )
    n_gpu_layers: int = 0
    context_size: int = 4096
    threads: int = 6
    batch_size: int = 128
    temperature: float = 0.0
    request_timeout: int = 20


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args()

    if not arguments.json:
        parser.print_help()
        return 0

    print(json.dumps(asdict(PlannerLLMConfig())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
