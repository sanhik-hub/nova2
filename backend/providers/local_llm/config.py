import os
from dataclasses import dataclass


@dataclass
class LocalLLMConfig:

    enabled: bool = True

    model_path: str = os.getenv(
        "LOCAL_LLM_MODEL",
        "models/local/Qwen3-4B-Q4_K_M.gguf",
    )

    # llama.cpp server
    server_url: str = os.getenv(
        "LOCAL_LLM_SERVER",
        "http://127.0.0.1:8080",
    )

    # CPU-first configuration
    n_gpu_layers: int = 0
    context_size: int = 4096
    threads: int = 6

    # Keep generation short by default.
    max_tokens: int = 384

    # Deterministic reasoning.
    temperature: float = 0.0

    batch_size: int = 128

    # Never allow a request to hang indefinitely.
    request_timeout: int = 90