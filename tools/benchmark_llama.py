#!/usr/bin/env python3
"""
Benchmark llama.cpp configurations and optionally apply the fastest profile.

The benchmark never uses Nova's production ports. It starts llama-server on a
temporary localhost port, runs controlled inference tests, shuts the server
down, and then tries the next configuration.

Examples:
    python tools/benchmark_llama.py
    python tools/benchmark_llama.py --runs 2 --apply
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "local" / "Qwen3-4B-Q4_K_M.gguf"
DEFAULT_SERVER = ROOT / "engines" / "llama.cpp" / "llama-server.exe"
PROFILE_PATH = ROOT / ".nova" / "llama_benchmark.json"

SYSTEM_PROMPT = """You are Nova's detailed mathematical solution writer.
Write ONE complete, self-contained mathematical solution from beginning to
end. Explain HOW the result is obtained and show important equations and
transformations. For differentiation, explicitly apply the relevant rules.
For derivations and proofs, continue to the requested final formula, result,
or conclusion. Never stop at an intermediate setup or partial argument.
Never invent missing information or assumptions. Return ONLY the complete
solution text."""

TESTS = [
    "Differentiate x^2 + 3x and show the differentiation rule used.",
    "Derive the formula for the induced emf of a rotating rod of length L "
    "with angular velocity omega in a uniform magnetic field B perpendicular "
    "to the plane of rotation.",
    "Solve x^2 - 5x + 6 = 0 and explain the algebraic steps briefly.",
]


@dataclass(frozen=True)
class Config:
    kind: str
    threads: int
    batch_size: int
    context_size: int
    n_gpu_layers: int


def eprint(message: str) -> None:
    print(message, file=sys.stderr)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def http_json(
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 10,
) -> dict[str, Any]:
    if payload is None:
        request = urllib.request.Request(url, method="GET")
    else:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def discover_gpu(server: Path) -> dict[str, Any]:
    """Detect hardware exposed by this llama.cpp build without assuming CUDA."""
    devices: list[str] = []
    list_error = None

    try:
        completed = subprocess.run(
            [str(server), "--list-devices"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        output = (completed.stdout + "\n" + completed.stderr).strip()
        for line in output.splitlines():
            if re.search(
                r"\b(?:CUDA|Vulkan|SYCL|HIP|ROCm|Metal|CANN|OpenCL)\w*",
                line,
                re.IGNORECASE,
            ):
                devices.append(line.strip())
    except Exception as exc:
        list_error = str(exc)

    nvidia_smi = None
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            nvidia_smi = completed.stdout.strip().splitlines()
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    return {
        "gpu_backend_detected": bool(devices or nvidia_smi),
        "llama_devices": devices,
        "nvidia_smi": nvidia_smi,
        "list_error": list_error,
    }


def cpu_thread_candidates() -> list[int]:
    logical = os.cpu_count() or 4
    candidates = {
        max(1, logical // 4),
        max(1, logical // 2),
        max(1, logical - 2),
        logical,
    }
    return sorted(candidates)


def build_candidates(gpu_info: dict[str, Any]) -> list[Config]:
    candidates: list[Config] = []

    # Stage 1: controlled CPU thread sweep using the current production
    # context/batch values as the baseline.
    for threads in cpu_thread_candidates():
        candidates.append(Config("cpu-thread", threads, 128, 32768, 0))

    # Stage 2: controlled batch/context sweep around the best CPU thread count.
    # A temporary placeholder thread count is replaced after stage 1.
    for batch in (32, 64, 128):
        for context in (4096, 32768):
            candidates.append(Config("cpu-shape", 0, batch, context, 0))

    # GPU is tested only when the executable exposes a non-CPU backend or
    # NVIDIA hardware is visible.  999 means "offload all possible layers";
    # llama.cpp will reject a configuration it cannot load.
    if gpu_info["gpu_backend_detected"]:
        for batch in (32, 64, 128):
            for context in (4096, 32768):
                candidates.append(Config("gpu-shape", 0, batch, context, 999))

    return candidates


def wait_ready(base_url: str, process: subprocess.Popen[Any], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(
                f"{base_url}/health",
                timeout=2,
            ) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(0.5)
    return False


def start_server(
    server: Path,
    model: Path,
    config: Config,
) -> tuple[subprocess.Popen[Any], str, Path]:
    port = free_port()
    log_fd, log_name = tempfile.mkstemp(prefix="nova_llama_", suffix=".log")
    os.close(log_fd)
    log_file = Path(log_name)
    log_handle = open(log_file, "w", encoding="utf-8")

    command = [
        str(server),
        "-m",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(config.context_size),
        "--threads",
        str(config.threads),
        "--batch-size",
        str(config.batch_size),
        "--n-gpu-layers",
        str(config.n_gpu_layers),
        "-np",
        "1",
    ]

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
    )
    # The child owns the OS handle after launch; close our copy.
    log_handle.close()

    return process, f"http://127.0.0.1:{port}", log_file


def stop_server(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def benchmark_request(
    base_url: str,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    started = time.perf_counter()
    result = http_json(
        f"{base_url}/v1/chat/completions",
        payload,
        timeout=max(30, max_tokens * 2),
    )
    elapsed_ms = (time.perf_counter() - started) * 1000

    choices = result.get("choices", [])
    usage = result.get("usage", {})
    timings = result.get("timings", {})
    finish_reason = choices[0].get("finish_reason") if choices else None

    predicted_n = int(
        timings.get(
            "predicted_n",
            usage.get("completion_tokens", 0),
        )
        or 0
    )
    predicted_ms = float(timings.get("predicted_ms", 0) or 0)
    prompt_n = int(
        timings.get(
            "cache_n",
            usage.get("prompt_tokens", 0),
        )
        or 0
    )
    prompt_ms = float(timings.get("prompt_ms", 0) or 0)

    generation_tps = (
        predicted_n / (predicted_ms / 1000)
        if predicted_n > 0 and predicted_ms > 0
        else 0.0
    )
    total_tps = (
        predicted_n / (elapsed_ms / 1000)
        if predicted_n > 0 and elapsed_ms > 0
        else 0.0
    )

    return {
        "elapsed_ms": round(elapsed_ms, 2),
        "prompt_tokens": prompt_n,
        "prompt_ms": round(prompt_ms, 2),
        "completion_tokens": predicted_n,
        "generation_ms": round(predicted_ms, 2),
        "generation_tps": round(generation_tps, 3),
        "total_tps": round(total_tps, 3),
        "finish_reason": finish_reason,
        "truncated": finish_reason == "length",
    }


def run_config(
    server: Path,
    model: Path,
    config: Config,
    runs: int,
    max_tokens: int,
    startup_timeout: float,
    tests: list[str],
) -> dict[str, Any]:
    process, base_url, log_file = start_server(server, model, config)
    measurements: list[dict[str, Any]] = []

    try:
        if not wait_ready(base_url, process, startup_timeout):
            return {
                "config": asdict(config),
                "success": False,
                "error": "llama-server failed to become ready",
                "measurements": [],
            }

        # Warm up model execution once. This is excluded from the score.
        try:
            benchmark_request(
                base_url,
                "Return only: 2 + 2 = 4",
                min(max_tokens, 32),
            )
        except Exception as exc:
            return {
                "config": asdict(config),
                "success": False,
                "error": f"warmup failed: {exc}",
                "measurements": [],
            }

        for test_index, prompt in enumerate(tests):
            for run_index in range(runs):
                try:
                    measurement = benchmark_request(
                        base_url,
                        prompt,
                        max_tokens,
                    )
                    measurement["test"] = test_index + 1
                    measurement["run"] = run_index + 1
                    measurements.append(measurement)
                except Exception as exc:
                    return {
                        "config": asdict(config),
                        "success": False,
                        "error": (
                            f"test {test_index + 1}, run {run_index + 1}: "
                            f"{exc}"
                        ),
                        "measurements": measurements,
                    }

        valid = [m for m in measurements if m["completion_tokens"] > 0]
        tps = [m["generation_tps"] for m in valid if m["generation_tps"] > 0]
        prompt_tps = [
            m["prompt_tokens"] / (m["prompt_ms"] / 1000)
            for m in valid
            if m["prompt_tokens"] > 0 and m["prompt_ms"] > 0
        ]

        return {
            "config": asdict(config),
            "success": True,
            "measurements": measurements,
            "avg_generation_tps": round(statistics.mean(tps), 3) if tps else 0.0,
            "median_generation_tps": round(statistics.median(tps), 3) if tps else 0.0,
            "avg_prompt_tps": round(statistics.mean(prompt_tps), 3) if prompt_tps else 0.0,
            "truncated": any(m["truncated"] for m in measurements),
        }
    finally:
        stop_server(process)
        try:
            log_file.unlink(missing_ok=True)
        except OSError:
            pass


def replace_threads(config: Config, threads: int) -> Config:
    return Config(
        kind=config.kind,
        threads=threads,
        batch_size=config.batch_size,
        context_size=config.context_size,
        n_gpu_layers=config.n_gpu_layers,
    )


def score(
    result: dict[str, Any],
    *,
    allow_search_ceiling: bool = False,
) -> float:
    if not result.get("success"):
        return -1.0
    # During the short configuration search, reaching --search-tokens is
    # expected: the search is measuring inference speed, not solution
    # completeness.  During final verification, truncated output remains
    # ineligible because the selected configuration must handle Nova's real
    # workload without hitting the artificial verification ceiling.
    if result.get("truncated") and not allow_search_ceiling:
        return -1.0
    return float(result.get("avg_generation_tps", 0.0))


def choose_best(
    results: list[dict[str, Any]],
    *,
    allow_search_ceiling: bool = False,
) -> dict[str, Any] | None:
    successful = [
        r
        for r in results
        if score(r, allow_search_ceiling=allow_search_ceiling) >= 0
    ]
    if not successful:
        return None
    return max(
        successful,
        key=lambda r: (
            score(r, allow_search_ceiling=allow_search_ceiling),
            float(r.get("avg_prompt_tps", 0.0)),
        ),
    )


def apply_profile(best: dict[str, Any]) -> None:
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "version": 1,
        "source": "tools/benchmark_llama.py",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": best["config"],
        "avg_generation_tps": best.get("avg_generation_tps", 0.0),
        "avg_prompt_tps": best.get("avg_prompt_tps", 0.0),
    }
    PROFILE_PATH.write_text(
        json.dumps(profile, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark llama.cpp CPU/GPU configurations for Nova2."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.getenv("LOCAL_LLM_MODEL", DEFAULT_MODEL)),
        help="GGUF model path.",
    )
    parser.add_argument(
        "--server",
        type=Path,
        default=DEFAULT_SERVER,
        help="llama-server executable.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="Measured runs per benchmark prompt (default: 2).",
    )
    parser.add_argument(
        "--search-tokens",
        type=int,
        default=256,
        help="Completion ceiling for the fast configuration search (default: 256).",
    )
    parser.add_argument(
        "--verify-tokens",
        type=int,
        default=768,
        help="Completion ceiling for final workload verification (default: 768).",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=120,
        help="Maximum seconds allowed for each server startup.",
    )
    parser.add_argument(
        "--production-port",
        type=int,
        default=8080,
        help="Nova final-model port that must be free during benchmarking (default: 8080).",
    )
    parser.add_argument(
        "--planner-port",
        type=int,
        default=8081,
        help="Nova planner port that must be free during benchmarking (default: 8081).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the winning device-local profile to .nova/.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")
    if args.search_tokens < 32:
        raise SystemExit("--search-tokens must be at least 32")
    if args.verify_tokens < 64:
        raise SystemExit("--verify-tokens must be at least 64")
    if not args.model.is_absolute():
        args.model = (ROOT / args.model).resolve()
    if not args.server.is_absolute():
        args.server = (ROOT / args.server).resolve()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")
    if not args.server.exists():
        raise SystemExit(f"llama-server not found: {args.server}")

    for port, name in (
        (args.production_port, "final model"),
        (args.planner_port, "planner"),
    ):
        if port_in_use(port):
            raise SystemExit(
                f"Port {port} is in use by the Nova {name}. "
                "Stop the active llama-server before benchmarking so the "
                "benchmark does not suffer memory/contention overhead."
            )

    gpu_info = discover_gpu(args.server)
    candidates = build_candidates(gpu_info)

    logical = os.cpu_count() or 1
    print("Nova2 llama.cpp benchmark")
    print("=" * 72)
    print(f"OS              : {platform.platform()}")
    print(f"CPU logical     : {logical}")
    print(f"Model           : {args.model}")
    print(f"llama-server    : {args.server}")
    print(f"GPU detected    : {gpu_info['gpu_backend_detected']}")
    if gpu_info["llama_devices"]:
        print("llama devices   :")
        for device in gpu_info["llama_devices"]:
            print(f"  {device}")
    if gpu_info["nvidia_smi"]:
        print("NVIDIA hardware :")
        for device in gpu_info["nvidia_smi"]:
            print(f"  {device}")
    print(f"Measured runs   : {args.runs}")
    print(f"Search tokens   : {args.search_tokens}")
    print(f"Verify tokens   : {args.verify_tokens}")
    print()

    # Stage 1: CPU thread sweep.
    thread_results: list[dict[str, Any]] = []
    print("Stage 1/3: CPU thread sweep")
    for config in candidates:
        if config.kind != "cpu-thread":
            continue
        result = run_config(
            args.server,
            args.model,
            config,
            args.runs,
            args.search_tokens,
            args.startup_timeout,
            TESTS[:1],
        )
        thread_results.append(result)
        print_result(result, search_phase=True)

    # The search ceiling is deliberately not a completeness requirement.
    # We need a fastest configuration even though the short search request may
    # legitimately use all 256 tokens.
    best_thread_result = choose_best(
        thread_results,
        allow_search_ceiling=True,
    )
    best_threads = (
        int(best_thread_result["config"]["threads"])
        if best_thread_result
        else (os.cpu_count() or 4)
    )

    # Stage 2: CPU batch/context sweep using the measured best thread count.
    shape_results: list[dict[str, Any]] = []
    print()
    print(f"Stage 2/3: CPU batch/context sweep (threads={best_threads})")
    for candidate in candidates:
        if candidate.kind != "cpu-shape":
            continue
        config = replace_threads(candidate, best_threads)
        result = run_config(
            args.server,
            args.model,
            config,
            args.runs,
            args.search_tokens,
            args.startup_timeout,
            TESTS[:1],
        )
        shape_results.append(result)
        print_result(result, search_phase=True)

    best_cpu = choose_best(
        thread_results + shape_results,
        allow_search_ceiling=True,
    )

    # GPU candidates get the full CPU thread count first; if they win, the
    # selected GPU configuration is then verified with the full test suite.
    gpu_results: list[dict[str, Any]] = []
    if gpu_info["gpu_backend_detected"]:
        print()
        print(f"Stage 2/3: GPU batch/context sweep (threads={best_threads})")
        for candidate in candidates:
            if candidate.kind != "gpu-shape":
                continue
            config = replace_threads(candidate, best_threads)
            result = run_config(
                args.server,
                args.model,
                config,
                args.runs,
                args.search_tokens,
                args.startup_timeout,
                TESTS[:1],
            )
            gpu_results.append(result)
            print_result(result, search_phase=True)

    shortlist = [
        r
        for r in [
            best_cpu,
            choose_best(gpu_results, allow_search_ceiling=True),
        ]
        if r is not None
    ]

    # Stage 3: full verification. This prevents a configuration that wins on
    # one short request from being applied if it behaves badly on derivations
    # or algebra.
    verified_results: list[dict[str, Any]] = []
    print()
    print("Stage 3/3: full workload verification")
    for candidate_result in shortlist:
        config = Config(**candidate_result["config"])
        result = run_config(
            args.server,
            args.model,
            config,
            args.runs,
            args.verify_tokens,
            args.startup_timeout,
            TESTS,
        )
        verified_results.append(result)
        print_result(result)

    best = choose_best(verified_results)

    print()
    print("=" * 72)
    if best is None:
        print("No configuration passed the full benchmark.")
        return 2

    cfg = best["config"]
    print("Winning configuration")
    print(f"  mode          : {cfg['kind']}")
    print(f"  threads       : {cfg['threads']}")
    print(f"  batch_size    : {cfg['batch_size']}")
    print(f"  context_size  : {cfg['context_size']}")
    print(f"  n_gpu_layers  : {cfg['n_gpu_layers']}")
    print(f"  generation    : {best['avg_generation_tps']:.3f} tok/s")
    print(f"  prompt        : {best['avg_prompt_tps']:.3f} tok/s")

    if args.apply:
        apply_profile(best)
        print(f"Applied locally: {PROFILE_PATH}")
    else:
        print("Not applied. Re-run with --apply to save this device-local profile.")

    return 0


def print_result(
    result: dict[str, Any],
    *,
    search_phase: bool = False,
) -> None:
    cfg = result["config"]
    label = (
        f"{cfg['kind']} "
        f"threads={cfg['threads']} "
        f"batch={cfg['batch_size']} "
        f"ctx={cfg['context_size']} "
        f"ngl={cfg['n_gpu_layers']}"
    )
    if not result.get("success"):
        print(f"  FAIL  {label} :: {result.get('error')}")
        return

    suffix = ""
    if result.get("truncated"):
        suffix = (
            " :: SEARCH CEILING REACHED"
            if search_phase
            else " :: TRUNCATED"
        )

    print(
        f"  PASS  {label} :: "
        f"{result.get('avg_generation_tps', 0):.2f} tok/s generation, "
        f"{result.get('avg_prompt_tps', 0):.2f} tok/s prompt"
        f"{suffix}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
