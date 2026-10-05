#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "scripts" / "stem_regression_cases.json"
DEFAULT_OUTPUT_DIR = ROOT / "scripts" / "results"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def persist_report(output: Path, report: dict) -> None:
    """Atomically persist the current report so it is inspectable mid-run."""
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(output)


def update_summary(report: dict, total: int, current: int, current_test: str | None = None) -> None:
    results = report.get("results", [])
    counts: dict[str, int] = {}
    timings: dict[str, list[float]] = {}
    for item in results:
        status = str(item.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
        kind = str(item.get("kind") or "unknown")
        timings.setdefault(kind, []).append(float(item.get("timing_ms") or 0.0))

    report["progress"] = {
        "completed_tests": len(results),
        "total_tests": total,
        "current_index": current,
        "current_test": current_test,
        "percent": round(len(results) * 100 / total, 2) if total else 100.0,
    }
    report["summary"] = {
        "total_tests": len(results),
        "status_counts": counts,
        "failed_tests": sum(item.get("status") != "pass" for item in results),
        "standard_tests": sum(item.get("kind") == "standard" for item in results),
        "guided_tests": sum(item.get("kind") == "guided" for item in results),
        "average_ms": {
            k: round(sum(v) / len(v), 2) for k, v in timings.items() if v
        },
        "max_ms": {
            k: round(max(v), 2) for k, v in timings.items() if v
        },
    }


def request_json(base_url: str, path: str, payload: dict | None = None, timeout: float = 180.0) -> dict:
    url = base_url.rstrip("/") + path
    data = None
    headers = {"Accept": "application/json"}
    method = "GET"
    if payload is not None:
        method = "POST"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    started = time.perf_counter()
    try:
        with urlopen(Request(url, data=data, headers=headers, method=method), timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
            return {
                "http_status": response.status,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                "json": parsed,
                "raw_response": raw,
                "exception": None,
                "traceback": None,
            }
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        return {
            "http_status": exc.code,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "json": parsed,
            "raw_response": raw,
            "exception": f"HTTPError: {exc}",
            "traceback": traceback.format_exc(),
        }
    except Exception as exc:
        return {
            "http_status": None,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "json": None,
            "raw_response": "",
            "exception": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def classify(response: dict) -> tuple[str, str | None]:
    if response["exception"]:
        return "transport_error", response["exception"]
    payload = response["json"]
    if not isinstance(payload, dict):
        return "invalid_response", "Response was not a JSON object."
    if payload.get("success") is True:
        return "pass", None
    return "backend_error", str(payload.get("error") or "success=false")


def standard_test(base_url: str, case: dict, explanation: str, timeout: float) -> dict:
    request = {"query": case["question"], "explanation": explanation}
    response = request_json(base_url, "/api/reason", request, timeout)
    status, error = classify(response)
    return {
        "kind": "standard",
        "case_id": case["id"],
        "domain": case["domain"],
        "explanation": explanation,
        "question": case["question"],
        "request": request,
        "status": status,
        "error": error,
        "timing_ms": response["elapsed_ms"],
        "http_status": response["http_status"],
        "response": response["json"],
        "raw_response": response["raw_response"],
        "exception": response["exception"],
        "traceback": response["traceback"],
    }


def guided_test(base_url: str, case: dict, include_hints: bool, timeout: float) -> dict:
    request = {
        "query": case["question"],
        "include_hints": include_hints,
        "include_full_solution": False,
    }
    response = request_json(base_url, "/api/guided/start", request, timeout)
    status, error = classify(response)
    payload = response["json"]
    structural_errors = []

    if status == "pass":
        steps = payload.get("steps")
        if not isinstance(steps, list) or len(steps) < 2:
            structural_errors.append("fewer than two guided steps")
        for index, step in enumerate(steps or [], 1):
            if not isinstance(step, dict):
                structural_errors.append(f"step {index} is not an object")
                continue
            options = step.get("options")
            correct_index = step.get("correct_index")
            if not isinstance(options, list) or len(options) != 4:
                structural_errors.append(f"step {index}: expected four options")
            if not isinstance(correct_index, int) or isinstance(correct_index, bool) or not 0 <= correct_index < 4:
                structural_errors.append(f"step {index}: invalid correct_index={correct_index!r}")
            if isinstance(options, list) and len(options) == 4:
                if any(not isinstance(o, dict) or not str(o.get("text") or "").strip() for o in options):
                    structural_errors.append(f"step {index}: empty/invalid option")

    if structural_errors:
        status = "contract_error"
        error = "; ".join(structural_errors)

    return {
        "kind": "guided",
        "case_id": case["id"],
        "domain": case["domain"],
        "question": case["question"],
        "request": request,
        "status": status,
        "error": error,
        "timing_ms": response["elapsed_ms"],
        "http_status": response["http_status"],
        "response": response["json"],
        "raw_response": response["raw_response"],
        "exception": response["exception"],
        "traceback": response["traceback"],
    }


def test_key(kind: str, case_id: str, explanation: str | None = None) -> str:
    """Return a stable identity for one regression test."""
    if kind == "standard":
        return f"standard:{case_id}:{explanation}"
    return f"guided:{case_id}"

def result_test_key(result: dict) -> str:
    return test_key(str(result.get("kind") or ""), str(result.get("case_id") or ""), result.get("explanation"))

def find_resume_report(
    output_dir: Path,
    *,
    cases_file: str,
    guided: bool,
    guided_hints: bool,
    base_url: str,
) -> Path | None:
    """Find the newest incomplete report matching the requested run configuration.

    A directory can contain interrupted 60-test standard-only runs alongside
    interrupted 80-test guided runs. Resume must select only a compatible run;
    otherwise a newer but incompatible report can mask the correct one.
    """
    if not output_dir.exists():
        return None

    candidates = []
    for path in output_dir.glob("regression_*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if data.get("run_status") == "completed":
            continue

        configuration = data.get("configuration") or {}
        if bool(configuration.get("guided", True)) != guided:
            continue
        if bool(configuration.get("guided_hints", False)) != guided_hints:
            continue
        if str(data.get("base_url") or "").rstrip("/") != base_url.rstrip("/"):
            continue
        if str(data.get("cases_file") or "") != cases_file:
            continue

        candidates.append((path.stat().st_mtime, path))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]

def expected_test_keys(cases: list[dict], skip_guided: bool) -> list[str]:
    keys = []
    for case in cases:
        for explanation in ("off", "concise", "detailed"):
            keys.append(test_key("standard", str(case["id"]), explanation))
        if not skip_guided:
            keys.append(test_key("guided", str(case["id"])))
    return keys
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--health-timeout", type=float, default=10.0)
    parser.add_argument("--guided-hints", action="store_true")
    parser.add_argument("--skip-guided", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume the newest incomplete report, or the report named by --output.")
    args = parser.parse_args()

    cases_path = args.cases if args.cases.is_absolute() else ROOT / args.cases
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise SystemExit("Cases file must contain a JSON array.")

    output = args.output
    if output is not None and not output.is_absolute():
        output = ROOT / output

    if args.resume:
        if output is None:
            cases_key = str(
                cases_path.relative_to(ROOT)
                if cases_path.is_relative_to(ROOT)
                else cases_path
            )
            output = find_resume_report(
                DEFAULT_OUTPUT_DIR,
                cases_file=cases_key,
                guided=not args.skip_guided,
                guided_hints=args.guided_hints,
                base_url=args.base_url,
            )
        if output is None or not output.exists():
            mode = "guided" if not args.skip_guided else "standard-only"
            hints = " with hints" if args.guided_hints else ""
            raise SystemExit(
                f"--resume requested, but no incomplete {mode}{hints} "
                f"report matching the current cases file and backend was found."
            )
        output = output.resolve()
        try:
            report = json.loads(output.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SystemExit(f"Could not read resume report {output}: {exc}") from exc
        if not isinstance(report, dict):
            raise SystemExit(f"Resume report {output} is not a JSON object.")
        if report.get("run_status") == "completed":
            raise SystemExit(f"Cannot resume completed report: {output}")
        existing_config = report.get("configuration") or {}
        if bool(existing_config.get("guided", True)) != (not args.skip_guided):
            raise SystemExit("Resume configuration mismatch: --skip-guided must match the interrupted run.")
        if bool(existing_config.get("guided_hints", False)) != bool(args.guided_hints):
            raise SystemExit("Resume configuration mismatch: --guided-hints must match the interrupted run.")
        if str(report.get("base_url") or "").rstrip("/") != args.base_url.rstrip("/"):
            raise SystemExit("Resume configuration mismatch: --base-url must match the interrupted run.")
        if not isinstance(report.get("results"), list):
            raise SystemExit("Resume report has no valid results list.")
        report.setdefault("summary", {})
        report["run_status"] = "resuming"
        report["resumed_at"] = now_iso()
        print(f"Resuming report: {output}", flush=True)
    else:
        if output is None:
            DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output = DEFAULT_OUTPUT_DIR / f"regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "schema_version": 1,
            "started_at": now_iso(),
            "base_url": args.base_url.rstrip("/"),
            "cases_file": str(cases_path.relative_to(ROOT) if cases_path.is_relative_to(ROOT) else cases_path),
            "configuration": {
                "standard_explanations": ["off", "concise", "detailed"],
                "guided": not args.skip_guided,
                "guided_hints": args.guided_hints,
                "request_timeout_seconds": args.request_timeout,
            },
            "health": None,
            "results": [],
            "summary": {},
        }
    print(f"Checking backend: {args.base_url}", flush=True)
    health = request_json(args.base_url, "/api/health", timeout=args.health_timeout)
    report["health"] = health
    total = len(cases) * 3 + (0 if args.skip_guided else len(cases))
    expected_keys = expected_test_keys(cases, args.skip_guided)
    completed_keys = {result_test_key(item) for item in report.get("results", []) if isinstance(item, dict)}
    unknown_keys = completed_keys - set(expected_keys)
    if unknown_keys:
        raise SystemExit("Resume report contains tests that are not part of the current cases/configuration.")
    if len(completed_keys) != len(report.get("results", [])):
        raise SystemExit("Resume report contains duplicate test records; refusing to guess which run state is authoritative.")
    if args.resume:
        print(f"Resuming after {len(completed_keys)} completed test(s); next test will be #{len(completed_keys) + 1}.", flush=True)
    update_summary(report, total, len(completed_keys), "health")
    persist_report(output, report)

    if health["exception"] or not isinstance(health["json"], dict):
        report["finished_at"] = now_iso()
        report["run_status"] = "backend_unavailable"
        report["summary"]["fatal"] = True
        report["summary"]["reason"] = health["exception"] or "invalid health response"
        persist_report(output, report)
        print(f"Backend unavailable. Report: {output}", flush=True)
        return 2

    current = len(completed_keys)

    def should_run(kind: str, case_id: str, explanation: str | None = None) -> bool:
        return test_key(kind, str(case_id), explanation) not in completed_keys

    def run_one(kind: str, label: str, factory) -> None:
        nonlocal current
        current += 1
        print(f"[{current}/{total}] {label}", flush=True)
        try:
            result = factory()
        except Exception as exc:
            result = {
                "kind": kind,
                "case_id": case["id"],
                "domain": case["domain"],
                "question": case["question"],
                "request": {},
                "status": "runner_error",
                "error": f"{type(exc).__name__}: {exc}",
                "timing_ms": 0.0,
                "http_status": None,
                "response": None,
                "raw_response": "",
                "exception": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        report["results"].append(result)
        update_summary(report, total, current, label)
        persist_report(output, report)
        if result["status"] != "pass":
            print(f"    ERROR: {result['error']}", flush=True)

    try:
        for case in cases:
            for explanation in ("off", "concise", "detailed"):
                if not should_run("standard", case["id"], explanation):
                    continue
                run_one(
                    "standard",
                    f"STANDARD {case['id']} [{explanation}]",
                    lambda case=case, explanation=explanation:
                        standard_test(args.base_url, case, explanation, args.request_timeout),
                )
            if not args.skip_guided and should_run("guided", case["id"]):
                run_one(
                    "guided",
                    f"GUIDED   {case['id']}",
                    lambda case=case:
                        guided_test(args.base_url, case, args.guided_hints, args.request_timeout),
                )
    except KeyboardInterrupt:
        report["finished_at"] = now_iso()
        report["run_status"] = "interrupted"
        update_summary(report, total, current, "interrupted")
        persist_report(output, report)
        print(f"Interrupted. Partial report: {output}", flush=True)
        return 130
    except Exception as exc:
        report["finished_at"] = now_iso()
        report["run_status"] = "runner_crash"
        report["runner_exception"] = f"{type(exc).__name__}: {exc}"
        report["runner_traceback"] = traceback.format_exc()
        update_summary(report, total, current, "runner_crash")
        persist_report(output, report)
        print(f"Runner crashed. Partial report: {output}", flush=True)
        return 1

    report["finished_at"] = now_iso()
    report["run_status"] = "completed"
    update_summary(report, total, total, None)
    persist_report(output, report)
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"Structured report: {output}", flush=True)
    return 0 if report["summary"]["failed_tests"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
