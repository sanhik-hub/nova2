import os
from dataclasses import dataclass


@dataclass
class LocalLLMConfig:
    enabled: bool = True

    # Model path will be configured later.
    model_path: str = os.getenv(
        "LOCAL_LLM_MODEL",
        "models/local/model.gguf",
    )

    # CPU-only for your current machine.
    n_gpu_layers: int = 0

    # Keep this conservative initially.
    context_size: int = 4096

    # 0 lets the runtime choose.
    threads: int = 0

    # Number of tokens generated per request.
    max_tokens: int = 1024

    # Deterministic enough for mathematical planning.
    temperature: float = 0.2

    # Keep prompt processing manageable on the i3.
    batch_size: int = 128