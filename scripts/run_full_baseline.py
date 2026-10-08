from __future__ import annotations

import argparse
import html.entities
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.shared import Inches, Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

try:
    import latex2mathml.converter
    import mathml2omml
    MATH_RENDERING_AVAILABLE = True
except Exception:
    latex2mathml = None
    mathml2omml = None
    MATH_RENDERING_AVAILABLE = False


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "scripts" / "baseline_cases.json"
OUT = ROOT / "scripts" / "results"
CHECKPOINT = OUT / "baseline_checkpoint.json"
MODES = ("off", "concise", "detailed")


def call(base, path, payload=None, timeout=1800):
    started = time.perf_counter()
    data = None
    headers = {"Accept": "application/json"}
    method = "GET"
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"

    try:
        with urlopen(
            Request(base.rstrip("/") + path, data=data, headers=headers, method=method),
            timeout=timeout,
        ) as response:
            raw = response.read().decode("utf-8", "replace")
            try:
                obj = json.loads(raw)
            except Exception:
                obj = None
            return {
                "status": response.status,
                "ms": (time.perf_counter() - started) * 1000,
                "json": obj,
                "raw": raw,
                "error": None,
            }
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            obj = json.loads(raw)
        except Exception:
            obj = None
        return {
            "status": exc.code,
            "ms": (time.perf_counter() - started) * 1000,
            "json": obj,
            "raw": raw,
            "error": str(exc),
        }
    except Exception as exc:
        return {
            "status": None,
            "ms": (time.perf_counter() - started) * 1000,
            "json": None,
            "raw": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def ok(result):
    return (
        not result["error"]
        and isinstance(result["json"], dict)
        and result["json"].get("success") is True
    )


def guided_check(payload):
    if not isinstance(payload, dict) or payload.get("success") is not True:
        return "backend failure"
    steps = payload.get("steps")
    if not isinstance(steps, list) or len(steps) < 2:
        return "fewer than two guided steps"
    for number, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            return f"step {number} invalid"
        options = step.get("options")
        if not isinstance(options, list) or len(options) != 4:
            return f"step {number}: expected four options"
        correct = step.get("correct_index")
        if not isinstance(correct, int) or isinstance(correct, bool) or not 0 <= correct < 4:
            return f"step {number}: invalid correct_index={correct!r}"
    return ""


def shorten(text, limit=18000):
    text = "" if text is None else str(text)
    return text if len(text) <= limit else text[:limit] + "\n… [truncated]"


def format_seconds(milliseconds):
    total = max(0, int(round(milliseconds / 1000)))
    minutes, seconds = divmod(total, 60)
    return f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"


def pretty_value(value, indent=0):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(
            f"{i + 1}. {pretty_value(item, indent + 2)}"
            for i, item in enumerate(value)
        )
    if isinstance(value, dict):
        preferred = (
            "answer",
            "result",
            "solution",
            "explanation",
            "steps",
            "hints",
            "options",
            "question",
            "correct_index",
        )
        lines = []
        used = set()
        for key in preferred:
            if key in value:
                used.add(key)
                rendered = pretty_value(value[key], indent + 2)
                if key == "correct_index":
                    rendered = str(value[key])
                if isinstance(value[key], (dict, list)):
                    lines.append(f"{key.replace('_', ' ').title()}:")
                    lines.append(rendered)
                else:
                    lines.append(f"{key.replace('_', ' ').title()}: {rendered}")
        for key, item in value.items():
            if key in used or key in {"success"}:
                continue
            rendered = pretty_value(item, indent + 2)
            if rendered:
                lines.append(f"{key.replace('_', ' ').title()}: {rendered}")
        return "\n".join(lines)
    return str(value)


def standard_output(result):
    payload = result.get("json")
    if isinstance(payload, dict):
        answer = payload.get("answer")
        if answer is not None:
            return shorten(str(answer).strip())
        for key in ("result", "solution", "explanation"):
            if payload.get(key) is not None:
                return shorten(pretty_value(payload[key]).strip())
    return shorten(result.get("raw", ""))


def guided_output(payload):
    if not isinstance(payload, dict):
        return shorten(payload or "")
    lines = []
    steps = payload.get("steps")
    if isinstance(steps, list):
        for number, step in enumerate(steps, 1):
            if not isinstance(step, dict):
                continue
            lines.append(f"Step {number}")
            question = step.get("question")
            if question:
                lines.append(str(question))
            hints = step.get("hints")
            if hints:
                lines.append("Hints:")
                if isinstance(hints, list):
                    lines.extend(f"• {x}" for x in hints)
                else:
                    lines.append(str(hints))
            options = step.get("options")
            if isinstance(options, list):
                labels = "ABCD"
                for i, option in enumerate(options):
                    label = labels[i] if i < len(labels) else str(i + 1)
                    lines.append(f"{label}. {option}")
            lines.append("")
    else:
        lines.append(pretty_value(payload))
    return shorten("\n".join(lines).strip())


def solution_output(payload):
    if not isinstance(payload, dict):
        return shorten(payload or "")
    for key in ("solution", "answer", "explanation", "result"):
        if payload.get(key):
            return shorten(pretty_value(payload[key]))
    return ""


def shade(cell, color="E8EDF3"):
    tc = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:fill"), color)
    tc.append(sh)


def set_cell_width(cell, width_inches):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:type"), "dxa")
    tc_w.set(qn("w:w"), str(int(width_inches * 1440)))


def set_row_no_split(row):
    tr_pr = row._tr.get_or_add_trPr()
    if tr_pr.find(qn("w:cantSplit")) is None:
        tr_pr.append(OxmlElement("w:cantSplit"))


def safe_text(value):
    text = shorten(value)
    text = html.unescape(text)
    # Remove Markdown/HTML transport artifacts before writing to Word.
    text = text.replace("\r\n", "\n").replace("\\r\\n", "\n").replace("\\n", "\n")
    text = re.sub(r"&#x20;|&nbsp;", " ", text)
    text = re.sub(r"(?m)^\s*\|\s*\|\s*$", "", text)
    text = re.sub(r"(?m)^\s*\|\s*(?:-+\s*\|\s*)+$", "", text)
    text = re.sub(r"(?m)^\s*\|\s*$", "", text)
    text = re.sub(r"(?m)^\s*\|\s*(.*?)\s*\|\s*$", r"\1", text)
    text = re.sub(r"(?m)^\s*#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*\*{3,}\s*$", "", text)
    text = re.sub(r"(?m)^\s*[-*_]{3,}\s*$", "", text)
    text = re.sub(r"(?m)^(\s*)(\d+)\\\.\s*", r"\1\2. ", text)

    chunks = []
    for line in text.split("\n"):
        line = line.rstrip()
        out = []
        for token in line.split(" "):
            if len(token) > 45:
                token = "\u200b".join(token[i:i + 45] for i in range(0, len(token), 45))
            out.append(token)
        chunks.append(" ".join(out))
    return "\n".join(chunks)


MATH_DELIMITER_RE = re.compile(
    r"(\\\[.*?\\\]|\\\(.*?\\\)|\$\$.*?\$\$|\$[^$\n]+\$)",
    re.DOTALL,
)

LATEX_COMMAND_RE = re.compile(
    r"\\(?:frac|dfrac|tfrac|sqrt|root|int|sum|prod|lim|ln|log|sin|cos|tan|"
    r"arcsin|arccos|arctan|exp|Gamma|gamma|Delta|delta|theta|omega|pi|infty|"
    r"zeta|cdot|times|to|rightarrow|leq|geq|approx|pm|mp|boxed|left|right|"
    r"begin|end)\b"
)


def _strip_math_delimiters(text):
    text = text.strip()
    if text.startswith("$$") and text.endswith("$$"):
        return text[2:-2].strip()
    if text.startswith("$") and text.endswith("$"):
        return text[1:-1].strip()
    if text.startswith(r"\[") and text.endswith(r"\]"):
        return text[2:-2].strip()
    if text.startswith(r"\(") and text.endswith(r"\)"):
        return text[2:-2].strip()
    return text


def _fallback_latex(text):
    """Readable fallback when the optional native Word-math dependencies are unavailable."""
    text = html.unescape(text)
    text = re.sub(r"\\boxed\s*\{(.+)\}", r"\1", text)
    replacements = {
        r"\infty": "∞",
        r"\pi": "π",
        r"\theta": "θ",
        r"\omega": "ω",
        r"\Gamma": "Γ",
        r"\gamma": "γ",
        r"\Delta": "Δ",
        r"\delta": "δ",
        r"\zeta": "ζ",
        r"\cdot": "·",
        r"\times": "×",
        r"\to": "→",
        r"\rightarrow": "→",
        r"\leq": "≤",
        r"\geq": "≥",
        r"\approx": "≈",
        r"\pm": "±",
        r"\mp": "∓",
        r"\int": "∫",
        r"\sum": "Σ",
        r"\prod": "Π",
        r"\ln": "ln",
        r"\log": "log",
        r"\sin": "sin",
        r"\cos": "cos",
        r"\tan": "tan",
        r"\left": "",
        r"\right": "",
        r"\,": " ",
        r"\!": "",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"\\(?:boxed|dfrac|tfrac|frac)\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\^\{([^{}]+)\}", r"^\1", text)
    text = re.sub(r"_\{([^{}]+)\}", r"_\1", text)
    text = text.replace("**", "")
    text = text.replace("\\", "")
    return text

def _latex_to_omml(latex):
    if not MATH_RENDERING_AVAILABLE:
        return None
    try:
        mathml = latex2mathml.converter.convert(latex, display="inline")
        omml = mathml2omml.convert(mathml, html.entities.name2codepoint)
        if isinstance(omml, str) and "xmlns:m=" not in omml:
            omml = omml.replace(
                "<m:oMath",
                '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"',
                1,
            )
        return ET.fromstring(omml)
    except Exception:
        return None


def _append_math(paragraph, latex, size=7):
    latex = _strip_math_delimiters(latex)
    omml = _latex_to_omml(latex)
    if omml is None:
        run = paragraph.add_run(_fallback_latex(latex))
        run.font.size = Pt(size)
        return
    paragraph._p.append(omml)


def _looks_like_bare_math(line):
    stripped = line.strip()
    if not stripped:
        return False
    if LATEX_COMMAND_RE.search(stripped):
        return True
    return bool(re.search(r"(?:\^\{[^}]+\}|\^\d|_\{[^}]+\}|\\[A-Za-z]+)", stripped)
                and len(stripped.split()) <= 18)


def _append_rich_line(paragraph, line, size=7, bold=False):
    line = html.unescape(line)
    # Markdown hard-break escapes should not appear in the finished DOCX.
    line = re.sub(r"\\\\\\s*$", "", line)
    line = re.sub(r"(?<!\\)\\\\\\.", ".", line)
    matches = list(MATH_DELIMITER_RE.finditer(line))
    if not matches and _looks_like_bare_math(line):
        _append_math(paragraph, line, size)
        return

    cursor = 0
    for match in matches:
        if match.start() > cursor:
            run = paragraph.add_run(safe_text(line[cursor:match.start()]))
            run.bold = bold
            run.font.size = Pt(size)
        _append_math(paragraph, match.group(0), size)
        cursor = match.end()
    if cursor < len(line):
        run = paragraph.add_run(safe_text(line[cursor:]))
        run.bold = bold
        run.font.size = Pt(size)


def put(cell, value, bold=False, size=7):
    cell.text = ""
    text = "" if value is None else str(value)
    lines = text.split("\n")
    in_display_math = False
    paragraph_index = 0

    for line in lines:
        stripped = line.strip()
        if stripped in ("\u0024\u0024", "\u0024", r"\\[", r"\\]"):
            if stripped == "\u0024\u0024":
                in_display_math = not in_display_math
            elif stripped in ("\u0024", r"\\["):
                in_display_math = True
            else:
                in_display_math = False
            continue

        paragraph = cell.paragraphs[0] if paragraph_index == 0 else cell.add_paragraph()
        paragraph_index += 1
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.paragraph_format.space_before = Pt(0)
        paragraph.paragraph_format.line_spacing = 1.0

        if in_display_math and stripped:
            _append_math(paragraph, stripped, size)
        else:
            _append_rich_line(paragraph, line, size=size, bold=bold)


def make_table(document, headers, widths):
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    tbl_pr = table._tbl.tblPr
    tbl_layout = tbl_pr.find(qn("w:tblLayout"))
    if tbl_layout is None:
        tbl_layout = OxmlElement("w:tblLayout")
        tbl_pr.append(tbl_layout)
    tbl_layout.set(qn("w:type"), "fixed")
    for index, header in enumerate(headers):
        set_cell_width(table.rows[0].cells[index], widths[index])
        put(table.rows[0].cells[index], header, True, 7)
        shade(table.rows[0].cells[index])
    set_row_no_split(table.rows[0])
    row_properties = table.rows[0]._tr.get_or_add_trPr()
    row_properties.append(OxmlElement("w:tblHeader"))
    return table


def set_table_row_widths(row, widths):
    set_row_no_split(row)
    for cell, width in zip(row.cells, widths):
        set_cell_width(cell, width)


def atomic_write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def add_case_header(document, case_number, case):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(8)
    paragraph.paragraph_format.space_after = Pt(3)
    run = paragraph.add_run(f"Test {case_number} : ")
    run.bold = True
    run.font.size = Pt(10)
    _append_rich_line(paragraph, case["question"], size=10, bold=True)

    meta = document.add_paragraph(
        f'{case["difficulty"].title()} • {case["domain"].upper()}'
    )
    meta.paragraph_format.space_after = Pt(4)
    meta.runs[0].font.size = Pt(7)


def make_case_table(document, rows, guided=False):
    if guided:
        headers = ["Test", "Actual MCQs + Hints", "MCQ Time", "Full Solution", "Solution Time", "Error", "Accuracy Remarks"]
        widths = [0.60, 3.70, 0.65, 2.20, 0.75, 1.00, 1.10]
    else:
        headers = ["Test", "Actual Result", "Elapsed", "Error", "Accuracy Remarks"]
        widths = [0.70, 5.10, 0.75, 1.35, 1.10]

    table = make_table(document, headers, widths)
    for item in rows:
        table_row = table.add_row()
        cells = table_row.cells
        set_table_row_widths(table_row, widths)
        values = item
        for index, value in enumerate(values):
            put(cells[index], value, size=7 if not guided else 6.8)
    return table


def save_docx(report, path, base_url):
    document = Document()
    section = document.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width = Inches(11.0)
    section.page_height = Inches(8.5)
    for margin in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
        setattr(section, margin, Inches(0.45))

    document.add_heading("Nova2 — Full Pre-Change STEM Baseline", 0)
    document.add_paragraph(
        f"Started: {report['started_at']} | Status: {report.get('status', 'running')} | "
        f"Backend: {base_url}"
    )
    document.add_paragraph(
        "Each question is shown once, followed by its four tests: Off, Concise, Detailed, "
        "and Guided. Output is rendered as readable text rather than raw JSON. "
        "Accuracy Remarks are intentionally blank for manual evaluation. Mathematical expressions are converted from LaTeX into native Word equations (OMML) when the optional rendering dependencies are installed; a readable Unicode fallback is used otherwise."
    )

    standard_by_id = {
        case_id: {}
        for case_id in (case["id"] for case in report["cases"])
    }
    guided_by_id = {}
    for result in report["results"]:
        if result["kind"] == "standard":
            standard_by_id.setdefault(result["id"], {})[result["mode"]] = result
        elif result["kind"] == "guided":
            guided_by_id[result["id"]] = result

    for case_number, case in enumerate(report["cases"], 1):
        add_case_header(document, case_number, case)
        rows = []
        for mode in MODES:
            result = standard_by_id.get(case["id"], {}).get(mode)
            if result is None:
                rows.append([mode.title(), "", "", "Not run", ""])
                continue
            rows.append([
                mode.title(),
                result.get("display_output", standard_output(result)),
                format_seconds(result["ms"]),
                result.get("error", "") or (
                    result.get("response", {}).get("error", "")
                    if isinstance(result.get("response"), dict)
                    else ""
                ),
                "",
            ])
        make_case_table(document, rows, guided=False)

        guided_result = guided_by_id.get(case["id"])
        if guided_result is None:
            guided_rows = [["Guided", "", "", "", "", "Not run", ""]]
        else:
            guided_rows = [[
                "Guided",
                guided_result.get("display_guided", guided_output(guided_result.get("response"))),
                format_seconds(guided_result["ms"]),
                guided_result.get("display_solution", solution_output(guided_result.get("solution"))),
                format_seconds(guided_result.get("solution_ms", 0)),
                guided_result.get("error", "") or (
                    guided_result.get("response", {}).get("error", "")
                    if isinstance(guided_result.get("response"), dict)
                    else ""
                ),
                "",
            ]]
        make_case_table(document, guided_rows, guided=True)

        if case_number != len(report["cases"]):
            document.add_paragraph()

    try:
        document.save(path)
    except PermissionError:
        fallback = path.with_name(path.stem + "_new.docx")
        document.save(fallback)
        return fallback
    return path

def save_report(report, stamp, base_url):
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = OUT / "baseline_checkpoint.json"
    atomic_write_json(checkpoint, report)

    json_path = OUT / f"baseline_{stamp}.json"
    atomic_write_json(json_path, report)

    docx_path = OUT / f"baseline_{stamp}.docx"
    actual_docx = save_docx(report, docx_path, base_url)
    return checkpoint, json_path, actual_docx


def result_key(kind, case_id, mode=None):
    return (kind, case_id, mode)


def load_or_create_report(cases, base_url, resume):
    OUT.mkdir(parents=True, exist_ok=True)
    if resume and CHECKPOINT.exists():
        try:
            report = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
            if (
                report.get("base_url") == base_url
                and [c.get("id") for c in report.get("cases", [])] == [c.get("id") for c in cases]
            ):
                return report, report.get("stamp") or datetime.now().strftime("%Y%m%d_%H%M%S")
        except Exception:
            pass

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "stamp": stamp,
        "cases": cases,
        "results": [],
        "status": "running",
    }, stamp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from scripts/results/baseline_checkpoint.json (default: true).",
    )
    parser.add_argument(
        "--max-new-tests",
        type=int,
        default=0,
        help="Stop cleanly after this many newly completed test calls; 0 means all.",
    )
    args = parser.parse_args()

    cases = json.loads(CASES.read_text(encoding="utf-8"))
    report, stamp = load_or_create_report(cases, args.base_url, args.resume)

    health = call(args.base_url, "/api/health", timeout=120)
    if health["error"] or not isinstance(health["json"], dict):
        raise SystemExit(f"Backend unavailable: {health['error'] or health['raw']}")

    existing = {
        result_key(r["kind"], r["id"], r.get("mode"))
        for r in report["results"]
    }
    completed_calls = len(report["results"])
    new_tests = 0

    def checkpoint():
        report["status"] = "paused"
        save_report(report, stamp, args.base_url)

    try:
        for case in cases:
            for mode in MODES:
                key = result_key("standard", case["id"], mode)
                if key in existing:
                    print(f"[resume] SKIP STANDARD {case['id']} {mode}", flush=True)
                    continue

                number = completed_calls + 1
                print(f"[{number}] STANDARD {case['id']} {mode}", flush=True)
                response = call(
                    args.base_url,
                    "/api/reason",
                    {"query": case["question"], "explanation": mode},
                    args.timeout,
                )
                error = response["error"] or (
                    None
                    if ok(response)
                    else str((response["json"] or {}).get("error", "success=false"))
                )
                report["results"].append(
                    {
                        "kind": "standard",
                        "id": case["id"],
                        "difficulty": case["difficulty"],
                        "domain": case["domain"],
                        "mode": mode,
                        "question": case["question"],
                        "ms": response["ms"],
                        "response": response["json"],
                        "raw": response["raw"],
                        "display_output": standard_output(response),
                "structured_output": (
                    response["json"].get("answer")
                    if isinstance(response.get("json"), dict)
                    else None
                ),
                        "error": error,
                    }
                )
                existing.add(key)
                completed_calls += 1
                new_tests += 1
                report["status"] = "running"
                save_report(report, stamp, args.base_url)

                if args.max_new_tests and new_tests >= args.max_new_tests:
                    checkpoint()
                    print("\nPaused cleanly. Re-run the same command to resume.", flush=True)
                    return

            key = result_key("guided", case["id"])
            if key in existing:
                print(f"[resume] SKIP GUIDED {case['id']}", flush=True)
                continue

            number = completed_calls + 1
            print(f"[{number}] GUIDED {case['id']}", flush=True)
            response = call(
                args.base_url,
                "/api/guided/start",
                {
                    "query": case["question"],
                    "include_hints": True,
                    "include_full_solution": False,
                },
                args.timeout,
            )
            error = response["error"] or guided_check(response["json"])
            solution = None

            if not error and isinstance(response["json"], dict):
                solution = call(
                    args.base_url,
                    "/api/guided/solution",
                    {
                        "query": case["question"],
                        "steps": response["json"].get("steps", []),
                    },
                    args.timeout,
                )
                if not ok(solution):
                    error = (
                        error
                        or solution["error"]
                        or str((solution["json"] or {}).get("error", "success=false"))
                    )

            report["results"].append(
                {
                    "kind": "guided",
                    "id": case["id"],
                    "difficulty": case["difficulty"],
                    "domain": case["domain"],
                    "question": case["question"],
                    "ms": response["ms"],
                    "response": response["json"],
                    "raw": response["raw"],
                    "display_guided": guided_output(response["json"]),
                    "solution": solution["json"] if solution else None,
                    "solution_raw": solution["raw"] if solution else "",
                    "display_solution": solution_output(solution["json"]) if solution else "",
                    "solution_ms": solution["ms"] if solution else 0,
                    "error": error,
                }
            )
            existing.add(key)
            completed_calls += 1
            new_tests += 1
            report["status"] = "running"
            save_report(report, stamp, args.base_url)

            if args.max_new_tests and new_tests >= args.max_new_tests:
                checkpoint()
                print("\nPaused cleanly. Re-run the same command to resume.", flush=True)
                return

        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["status"] = "completed"
        checkpoint_path, json_path, docx_path = save_report(report, stamp, args.base_url)
        print("\nBaseline completed.", flush=True)
        print(f"Checkpoint: {checkpoint_path}", flush=True)
        print(f"JSON backup: {json_path}", flush=True)
        print(f"DOCX report: {docx_path}", flush=True)

    except KeyboardInterrupt:
        checkpoint()
        print("\nBaseline paused safely.", flush=True)
        print("Completed tests are checkpointed. Re-run the same command to resume.", flush=True)
        return


if __name__ == "__main__":
    main()