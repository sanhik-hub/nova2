#!/usr/bin/env python3
"""
Exercise Nova's lightweight guided-attempt planner across multiple STEM domains.

Run from the repository root:
    python tools/test_planner.py

The script does not call the main guided-attempt generator. It talks directly
to the planner endpoint so planner behavior can be measured independently.

Outputs:
    tools/planner_test_results.json
    tools/planner_test_results.csv
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "http://127.0.0.1:8081/v1/chat/completions"
PLANNER_URL = os.getenv("NOVA_PLANNER_URL", DEFAULT_URL)
JSON_OUT = ROOT / "tools" / "planner_test_results.json"
CSV_OUT = ROOT / "tools" / "planner_test_results.csv"

SYSTEM_PROMPT = """
You are Nova's lightweight guided-attempt complexity estimator.

Your output is advisory metadata only. You do NOT solve the problem and you
do NOT determine its solution steps.

Return exactly one JSON object:
{"complexity":"simple|moderate|complex|very_complex","step_count_advisory":N}

Use your best estimate of how much reasoning a stronger model will need.
The advisory step count may be greater than 5 for complex problems.
It is not a hard limit and must not contain a solution path, equations,
formulas, results, options, hints, or explanations.
""".strip()

TESTS = [
    {"id": "math_calculus_01", "domain": "Mathematics", "name": "Basic derivative",
     "problem": "Differentiate x^3 + 2x^2 - 5x + 7 with respect to x."},
    {"id": "math_calculus_02", "domain": "Mathematics", "name": "Implicit differentiation",
     "problem": "For x^2 + xy + y^2 = 7, find dy/dx."},
    {"id": "math_calculus_03", "domain": "Mathematics", "name": "Definite integral",
     "problem": "Evaluate the definite integral of x^2 sin(x) from 0 to pi."},
    {"id": "math_algebra_01", "domain": "Mathematics", "name": "System of equations",
     "problem": "Solve the simultaneous equations 2x + 3y = 13 and 4x - y = 5."},
    {"id": "physics_mechanics_01", "domain": "Physics", "name": "Rotating rod",
     "problem": "A uniform rod of length L and mass M is rotating about one end with angular velocity omega. Determine the angular momentum of the rod about the axis of rotation."},
    {"id": "physics_mechanics_02", "domain": "Physics", "name": "Projectile motion",
     "problem": "A projectile is launched from ground with speed u at angle theta. Derive its maximum height and horizontal range."},
    {"id": "physics_electrostatics_01", "domain": "Physics", "name": "Electric field integration",
     "problem": "Find the electric field on the axis of a uniformly charged finite ring at a point a distance x from its center."},
    {"id": "physics_thermodynamics_01", "domain": "Physics", "name": "Thermodynamic cycle",
     "problem": "An ideal gas undergoes a multi-step cycle consisting of an isothermal expansion, an isochoric heating, and an adiabatic compression. Determine the net work done over the cycle."},
    {"id": "chemistry_stoichiometry_01", "domain": "Chemistry", "name": "Stoichiometry",
     "problem": "A sample contains 5.00 g of calcium carbonate. Calculate the amount of carbon dioxide produced when it reacts completely with excess hydrochloric acid."},
    {"id": "chemistry_equilibrium_01", "domain": "Chemistry", "name": "Chemical equilibrium",
     "problem": "For N2 + 3H2 reversible equilibrium with NH3, determine the equilibrium composition when initial concentrations and the equilibrium constant are given."},
    {"id": "chemistry_thermochemistry_01", "domain": "Chemistry", "name": "Hess's law",
     "problem": "Using three supplied thermochemical equations, determine the enthalpy change for the target reaction by combining and reversing the equations as necessary."},
    {"id": "mixed_complex_01", "domain": "Mixed STEM", "name": "Multi-stage mechanics derivation",
     "problem": "A solid cylinder rolls without slipping down an inclined plane from rest. Determine its final speed after descending a vertical height h, then compare the result with a hoop and explain the dependence on the distribution of mass."},
]


def parse_json_text(raw: Any) -> Any:
    if not isinstance(raw, str):
        raise ValueError("planner content is not text")

    text = raw.strip()
    fence = chr(96) * 3
    if text.startswith(fence):
        text = re.sub(r"^" + fence + r"(?:json)?\s*", "", text, count=1, flags=re.I)
        text = re.sub(r"\s*" + fence + r"$", "", text, count=1)

    return json.loads(text)


def request_planner(problem: str) -> dict[str, Any]:
    body = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                "Problem:\n" + problem + "\n\n"
                "Estimate only the problem complexity and approximate "
                "guided-step count. Return the required JSON object."
            )},
        ],
        "max_tokens": 160,
        "temperature": 0,
    }

    request = urllib.request.Request(
        PLANNER_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw_body = response.read().decode("utf-8")
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return {"http_success": True, "elapsed_ms": elapsed_ms, "response": json.loads(raw_body)}
    except urllib.error.HTTPError as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        detail = exc.read().decode("utf-8", errors="replace")
        return {"http_success": False, "elapsed_ms": elapsed_ms, "error": f"HTTP {exc.code}: {detail}"}
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return {"http_success": False, "elapsed_ms": elapsed_ms, "error": str(exc)}


def inspect_result(test: dict[str, Any], call: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": test["id"],
        "domain": test["domain"],
        "name": test["name"],
        "problem": test["problem"],
        "planner_url": PLANNER_URL,
        "http_success": call["http_success"],
        "elapsed_ms": call["elapsed_ms"],
        "finish_reason": "",
        "completion_tokens": "",
        "prompt_tokens": "",
        "raw_content": "",
        "valid_json": False,
        "complexity": "",
        "schema_valid": False,
        "contains_path": False,
        "contains_solution_like_text": False,
        "error": call.get("error", ""),
    }

    if not call["http_success"]:
        return row

    payload = call["response"]
    choices = payload.get("choices") or []
    usage = payload.get("usage") or {}
    row["finish_reason"] = choices[0].get("finish_reason", "") if choices else ""
    row["completion_tokens"] = usage.get("completion_tokens", "")
    row["prompt_tokens"] = usage.get("prompt_tokens", "")

    content = ""
    if choices:
        content = (choices[0].get("message") or {}).get("content") or ""
    row["raw_content"] = content

    try:
        parsed = parse_json_text(content)
        row["valid_json"] = True
    except Exception as exc:
        row["error"] = f"JSON parse error: {exc}"
        return row

    if not isinstance(parsed, dict):
        row["error"] = "Planner JSON is not an object."
        return row

    row["complexity"] = parsed.get("complexity", "")
    row["contains_path"] = "path" in parsed

    solution_terms = (
        "calculate ", "derive ", "solve ", "evaluate ", "find ",
        "determine ", "formula", "equation", "=",
    )
    row["contains_solution_like_text"] = any(
        term in str(parsed).lower() for term in solution_terms
    )

    row["schema_valid"] = (
        set(parsed.keys()) == {"complexity"}
        and parsed.get("complexity") in {"simple", "moderate", "complex", "very_complex"}
    )
    return row


def main() -> int:
    rows = []

    print(f"Planner endpoint: {PLANNER_URL}")
    print(f"Running {len(TESTS)} domain-diverse planner tests...")

    for number, test in enumerate(TESTS, start=1):
        print(f"[{number:02d}/{len(TESTS):02d}] {test['domain']}: {test['name']}")
        row = inspect_result(test, request_planner(test["problem"]))
        rows.append(row)

        status = "PASS" if row["schema_valid"] and row["advisory_range_valid"] else "CHECK"
        print(
            f"    {status} | {row['elapsed_ms']} ms | "
            f"complexity={row['complexity'] or '-'} | "
            f"finish={row['finish_reason'] or '-'}"
        )

    run = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "planner_url": PLANNER_URL,
        "test_count": len(rows),
        "summary": {
            "http_success": sum(bool(r["http_success"]) for r in rows),
            "valid_json": sum(bool(r["valid_json"]) for r in rows),
            "schema_valid": sum(bool(r["schema_valid"]) for r in rows),
        },
        "tests": rows,
    }

    JSON_OUT.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = list(rows[0].keys()) if rows else []
    with CSV_OUT.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Structured results: {JSON_OUT}")
    print(f"Tabular results:    {CSV_OUT}")
    print(json.dumps(run["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
