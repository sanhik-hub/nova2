from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path


CASES = [
    "A conducting rod of length L rotates with angular velocity ω about one end in a uniform magnetic field B perpendicular to the plane of rotation. Derive the emf induced across the rod.",
    "A body is projected vertically upward with initial speed u. Derive the maximum height reached.",
    "A 2 kg object is acted on by a constant force of 10 N. Find its acceleration.",
]


def post(url: str, payload: dict, timeout: int = 180) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
        return {
            "ok": True,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "status": response.status,
            "raw": raw,
        }
    except Exception as exc:
        return {
            "ok": False,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "error": str(exc),
        }


def inspect_payload(raw: str) -> dict:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"valid_json": False, "error": "invalid_json"}

    steps = value.get("steps", [])
    errors = []
    placeholder_indices = []
    if not isinstance(steps, list):
        errors.append("steps_not_list")
        steps = []

    for index, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            errors.append(f"step_{index}_not_object")
            continue
        correct = step.get("correct_index")
        if not isinstance(correct, int) or isinstance(correct, bool) or correct not in range(4):
            errors.append(f"step_{index}_invalid_correct_index")
            if str(correct).strip().upper() == "CORRECT_INDEX":
                placeholder_indices.append(index)

        options = step.get("options")
        if not isinstance(options, list) or len(options) != 4:
            errors.append(f"step_{index}_option_count")
        if isinstance(options, list) and len(options) == 4:
            if len({str(x).strip() for x in options}) != 4:
                errors.append(f"step_{index}_duplicate_options")

    return {
        "valid_json": True,
        "success": bool(value.get("success")),
        "step_count": len(steps),
        "errors": errors,
        "placeholder_indices": placeholder_indices,
        "solution_generation_elapsed_ms": value.get("solution_generation_elapsed_ms"),
        "mcq_generation_elapsed_ms": value.get("mcq_generation_elapsed_ms"),
        "elapsed_ms": value.get("elapsed_ms"),
        "api_error": value.get("error"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/api/guided/start")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = []
    for case in CASES:
        for run in range(1, args.runs + 1):
            response = post(args.url, {
                "query": case,
                "include_hints": False,
                "include_full_solution": False,
            })
            row = {"case": case, "run": run, **response}
            if response["ok"]:
                row.update(inspect_payload(response["raw"]))
            results.append(row)
            print(
                f"{run} | {response['elapsed_ms']:>9.2f} ms | "
                f"{'HTTP OK' if response['ok'] else 'FAILED'} | {case}"
            )
            if response.get("ok"):
                print(f"    {row.get('errors', [])}")

    valid = [
        r for r in results
        if r.get("success") and not r.get("errors")
    ]
    times = [r["elapsed_ms"] for r in results if r.get("ok")]
    summary = {
        "runs": len(results),
        "valid": len(valid),
        "valid_rate": round(len(valid) / len(results), 3) if results else 0,
        "avg_elapsed_ms": round(statistics.mean(times), 2) if times else None,
        "median_elapsed_ms": round(statistics.median(times), 2) if times else None,
        "max_elapsed_ms": round(max(times), 2) if times else None,
        "placeholder_correct_index_count": sum(
            len(r.get("placeholder_indices", [])) for r in results
        ),
    }

    output = {"summary": summary, "results": results}
    destination = Path(args.output) if args.output else Path("scripts/results") / (
        f"guided_output_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
