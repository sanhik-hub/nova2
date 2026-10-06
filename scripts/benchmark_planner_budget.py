from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path


CASES = [
    "Differentiate x^2 + 3x + 5.",
    "Find the integral of 2x^2 - 4x.",
    "What is the formula for nitrobenzene?",
    "A charge of 2 μC and a charge of 3 μC are separated by 0.5 m. Calculate the electrostatic force.",
    "A rod of length L rotates with angular velocity ω in a uniform magnetic field B. Derive the induced emf.",
    "What is the pH of a 0.01 M strong acid solution?",
    "A gas occupies 2 L at 300 K. What volume does it occupy at 450 K at constant pressure?",
    "Explain why increasing temperature affects the rate of a chemical reaction.",
]


SYSTEM_PROMPT = """You are Nova's routing planner.

Classify the user's request and return one compact JSON object with exactly
these fields:
{"type":"plan","operation":"...","provider":"...","reasoning":[],"idea":"...","answer_tokens":0}

Use provider values only when genuinely needed. Use a computational provider
for exact symbolic/numerical computation when appropriate; use the local
reasoning model for semantic or explanatory STEM questions. Do not solve the
user's problem. Do not write a final answer. Keep reasoning empty unless a
short routing note is genuinely useful. answer_tokens is only an advisory
estimate for the final response, never a hard limit.

Return JSON only. Keep the response as short as possible.
"""


def post(url: str, payload: dict, timeout: int = 120) -> dict:
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
        elapsed = (time.perf_counter() - started) * 1000
        return {"ok": True, "elapsed_ms": round(elapsed, 2), "raw": raw}
    except Exception as exc:
        elapsed = (time.perf_counter() - started) * 1000
        return {"ok": False, "elapsed_ms": round(elapsed, 2), "error": str(exc)}


def parse_plan(raw: str) -> tuple[bool, str]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return False, "invalid_json"
    if not isinstance(value, dict):
        return False, "not_object"
    required = {"type", "operation", "provider", "reasoning", "idea", "answer_tokens"}
    if not required.issubset(value):
        return False, "missing_fields"
    if value.get("type") != "plan":
        return False, "wrong_type"
    if not isinstance(value.get("answer_tokens"), (int, float)):
        return False, "bad_answer_tokens"
    return True, "ok"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8081/v1/chat/completions")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--budgets", default="32,48,64,96")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    results = []

    for budget in budgets:
        for case in CASES:
            for run in range(1, args.runs + 1):
                payload = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps({
                            "explanation_preference": "off",
                            "user_request": case,
                        }, ensure_ascii=False)},
                    ],
                    "temperature": 0.0,
                    "max_tokens": budget,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                result = post(args.url, payload)
                row = {
                    "budget": budget,
                    "case": case,
                    "run": run,
                    **result,
                }
                if result["ok"]:
                    valid, status = parse_plan(result["raw"])
                    row["valid"] = valid
                    row["status"] = status
                    row["output_chars"] = len(result["raw"])
                else:
                    row["valid"] = False
                    row["status"] = "request_failed"
                results.append(row)
                print(
                    f"{budget:>3} tokens | {result['elapsed_ms']:>9.2f} ms | "
                    f"{'OK' if row['valid'] else row['status']:<16} | {case}"
                )

    summary = []
    for budget in budgets:
        rows = [r for r in results if r["budget"] == budget]
        valid = sum(bool(r["valid"]) for r in rows)
        times = [r["elapsed_ms"] for r in rows if r["ok"]]
        summary.append({
            "budget": budget,
            "runs": len(rows),
            "valid": valid,
            "valid_rate": round(valid / len(rows), 3) if rows else 0,
            "avg_ms": round(statistics.mean(times), 2) if times else None,
            "median_ms": round(statistics.median(times), 2) if times else None,
            "max_ms": round(max(times), 2) if times else None,
        })

    output = {"budgets": budgets, "cases": CASES, "summary": summary, "results": results}
    destination = Path(args.output) if args.output else Path("scripts/results") / (
        f"planner_budget_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
