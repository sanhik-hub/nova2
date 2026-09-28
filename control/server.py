from __future__ import annotations

import json
import urllib.request
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


NOVA_ROOT = Path(r"C:\Users\sanhi\Documents\nova2")

RUNNER = NOVA_ROOT / "run_nova.ps1"
PYTHON = Path(r"C:\Users\sanhi\Documents\code\python\.venv\Scripts\python.exe")

CONFIG_SCRIPT = (
    NOVA_ROOT
    / "backend"
    / "providers"
    / "local_llm"
    / "config.py"
)

LLAMA_SERVER = (
    NOVA_ROOT
    / "engines"
    / "llama.cpp"
    / "llama-server.exe"
)

FRONTEND_ROOT = NOVA_ROOT / "frontend"

HOST = "127.0.0.1"
PORT = 8100

BACKEND_PORT = 8000
FRONTEND_PORT = 5173

LOG_LOCK = threading.Lock()
LOGS: list[str] = []


def log(message: str):
    stamp = time.strftime("%H:%M:%S")
    line = f"{stamp}  {message}"

    with LOG_LOCK:
        LOGS.append(line)

        if len(LOGS) > 300:
            del LOGS[:-300]

    print(line, flush=True)


def ps(command: str, timeout: int = 15) -> str:
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        cwd=str(NOVA_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip()
            or completed.stdout.strip()
            or f"PowerShell exited with {completed.returncode}"
        )

    return completed.stdout.strip()


def load_llm_config():
    try:
        result = subprocess.run(
            [
                str(PYTHON),
                str(CONFIG_SCRIPT),
                "--json",
            ],
            cwd=str(NOVA_ROOT),
            capture_output=True,
            text=True,
            timeout=15,
        )

        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.strip()
                or "config.py failed"
            )

        data = json.loads(result.stdout)

        model_path = str(
            data.get("model_path", "")
        ).strip()

        server_url = str(
            data.get("server_url", "")
        ).strip()

        return {
            "model_path": model_path,
            "server_url": server_url,
            "model_name": (
                Path(model_path).stem
                if model_path
                else ""
            ),
        }

    except Exception as exc:
        return {
            "model_path": "",
            "server_url": "",
            "model_name": "",
            "error": str(exc),
        }


def get_processes():
    command = r"""
Get-CimInstance Win32_Process |
    Where-Object {
        $_.Name -in @(
            "python.exe",
            "pythonw.exe",
            "llama-server.exe",
            "WolframKernel.exe",
            "node.exe"
        )
    } |
    Select-Object ProcessId, ParentProcessId, Name, CommandLine |
    ConvertTo-Json -Compress -Depth 4
"""

    try:
        raw = ps(command)

        if not raw:
            return []

        data = json.loads(raw)

        if isinstance(data, dict):
            data = [data]

        return data

    except Exception as exc:
        log(f"Process inspection failed: {exc}")
        return []


def port_listener(port: int):
    command = f"""
Get-NetTCPConnection `
    -LocalPort {port} `
    -State Listen `
    -ErrorAction SilentlyContinue |
    Select-Object -First 1 `
        OwningProcess,
        LocalAddress,
        LocalPort,
        State |
    ConvertTo-Json -Compress
"""

    try:
        raw = ps(command)

        if not raw:
            return None

        return json.loads(raw)

    except Exception as exc:
        log(
            f"Port inspection failed for "
            f"{port}: {exc}"
        )
        return None


def process_snapshot():
    processes = get_processes()

    backend = []
    frontend = []
    llama = []
    wolfram = []

    backend_listener = port_listener(BACKEND_PORT)
    frontend_listener = port_listener(FRONTEND_PORT)

    backend_pid = None
    frontend_pid = None

    if isinstance(backend_listener, dict):
        backend_pid = backend_listener.get(
            "OwningProcess"
        )
    elif isinstance(backend_listener, list) and backend_listener:
        backend_pid = backend_listener[0].get(
            "OwningProcess"
        )

    if isinstance(frontend_listener, dict):
        frontend_pid = frontend_listener.get(
            "OwningProcess"
        )
    elif isinstance(frontend_listener, list) and frontend_listener:
        frontend_pid = frontend_listener[0].get(
            "OwningProcess"
        )

    for process in processes:
        name = str(
            process.get("Name") or ""
        )

        command = str(
            process.get("CommandLine") or ""
        )

        pid = process.get("ProcessId")
        parent = process.get("ParentProcessId")

        name_lower = name.lower()
        command_lower = command.lower()

        # Backend is identified by the process
        # actually owning port 8000.
        if (
            backend_pid is not None
            and str(pid) == str(backend_pid)
            and name_lower in (
                "python.exe",
                "pythonw.exe",
            )
        ):
            backend.append(
                {
                    "pid": pid,
                    "parent_pid": parent,
                    "name": name,
                    "command": command,
                }
            )

        # Frontend is identified primarily by
        # the process owning Vite's port.
        if (
            frontend_pid is not None
            and str(pid) == str(frontend_pid)
            and name_lower == "node.exe"
        ):
            frontend.append(
                {
                    "pid": pid,
                    "parent_pid": parent,
                    "name": name,
                    "command": command,
                }
            )

        if (
            name_lower == "llama-server.exe"
            and str(LLAMA_SERVER).lower()
            in command_lower
        ):
            llama.append(
                {
                    "pid": pid,
                    "parent_pid": parent,
                    "name": name,
                    "command": command,
                }
            )

        if name_lower == "wolframkernel.exe":
            wolfram.append(
                {
                    "pid": pid,
                    "parent_pid": parent,
                    "name": name,
                    "command": command,
                }
            )

    return {
        "backend": backend,
        "frontend": frontend,
        "llm": llama,
        "wolfram": wolfram,
    }


def http_check(url: str):
    try:
        request = urllib.request.Request(
            url,
            method="GET",
        )

        with urllib.request.urlopen(
            request,
            timeout=5,
        ) as response:
            return {
                "ok": True,
                "status": int(response.status),
            }

    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
        }


def http_json_check(url: str):
    try:
        request = urllib.request.Request(
            url,
            method="GET",
        )

        with urllib.request.urlopen(
            request,
            timeout=5,
        ) as response:
            raw = response.read().decode(
                "utf-8",
                errors="replace",
            )

        result = json.loads(raw)

        if isinstance(result, dict):
            result.setdefault("ok", True)

        return result

    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
        }

def http_json_post(url: str):
    request = urllib.request.Request(url, method="POST")
    with urllib.request.urlopen(request, timeout=10) as response:
        raw = response.read().decode("utf-8", errors="replace")
    result = json.loads(raw) if raw else {}
    return result if isinstance(result, dict) else {}


def health_status():
    config = load_llm_config()
    processes = process_snapshot()

    backend_listener = port_listener(
        BACKEND_PORT
    )

    frontend_listener = port_listener(
        FRONTEND_PORT
    )

    backend_http = http_check(
        f"http://{HOST}:{BACKEND_PORT}/docs"
    )

    frontend_http = http_check(
        f"http://{HOST}:{FRONTEND_PORT}/"
    )

    llm_url = (
        config.get("server_url", "")
        .rstrip("/")
    )

    if llm_url:
        llm_health = http_check(
            f"{llm_url}/health"
        )
    else:
        llm_health = {
            "ok": False,
            "error": "No server_url configured",
        }

    # The backend itself is the authoritative source
    # for the persistent Wolfram session.
    wolfram_health = {
        "ok": False,
        "wolfram": False,
    }

    if backend_http.get("ok"):
        wolfram_health = http_json_check(
            f"http://{HOST}:{BACKEND_PORT}/api/health"
        )

    session = wolfram_health.get("wolfram_session") or {}
    session_started = bool(session.get("session_started"))
    kernel_count = len(processes["wolfram"])

    # UI says KERNEL RUNNING only when both the Python session and
    # an actual WolframKernel.exe process are present.
    wolfram_active = session_started and kernel_count > 0

    # A service is considered running when its actual
    # endpoint is reachable. Process information is
    # supplemental diagnostic information.
    backend_running = bool(
        backend_listener and backend_http.get("ok")
    )

    frontend_running = bool(
        frontend_listener and frontend_http.get("ok")
    )

    llm_running = bool(
        llm_health.get("ok")
    )

    return {
        "backend": {
            "process": backend_running,
            "pid": (
                backend_listener.get("OwningProcess")
                if isinstance(backend_listener, dict)
                else (
                    processes["backend"][0]["pid"]
                    if processes["backend"]
                    else None
                )
            ),
            "port": BACKEND_PORT,
            "listener": backend_listener,
            "http": backend_http,
        },

        "frontend": {
            "process": frontend_running,
            "pid": (
                frontend_listener.get("OwningProcess")
                if isinstance(frontend_listener, dict)
                else (
                    processes["frontend"][0]["pid"]
                    if processes["frontend"]
                    else None
                )
            ),
            "port": FRONTEND_PORT,
            "listener": frontend_listener,
            "http": frontend_http,
            "root": str(FRONTEND_ROOT),
        },

        "llm": {
            "process": llm_running,
            "pid": (
                processes["llm"][0]["pid"]
                if processes["llm"]
                else (
                    port_listener(
                        8080
                    ).get("OwningProcess")
                    if isinstance(
                        port_listener(8080),
                        dict
                    )
                    else None
                )
            ),
            "server_url": config.get(
                "server_url",
                "",
            ),
            "model_name": config.get(
                "model_name",
                "",
            ),
            "model_path": config.get(
                "model_path",
                "",
            ),
            "health": llm_health,
        },

        "wolfram": {
            "process": wolfram_active,
            "pids": [
                process["pid"]
                for process in processes["wolfram"]
            ],
            "count": kernel_count,
            "embedded": True,
            "healthy": wolfram_active,
            "session_exists": bool(session.get("session_exists")),
            "session_started": session_started,
            "note": (
                "Wolfram runs as a persistent "
                "session inside the Nova backend."
            ),
            "health": wolfram_health,
        },

        "runner": {
            "exists": RUNNER.exists(),
            "path": str(RUNNER),
        },

        "timestamp": time.time(),
    }


def start_backend():
    if port_listener(BACKEND_PORT):
        return {"ok": True, "message": "Backend is already running."}

    log("Starting FastAPI backend...")
    subprocess.Popen(
        [
            str(PYTHON),
            "-m", "uvicorn", "backend.main:app",
            "--host", HOST,
            "--port", str(BACKEND_PORT),
        ],
        cwd=str(NOVA_ROOT),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    return {"ok": True, "message": "Backend start requested."}


def start_frontend():
    if port_listener(FRONTEND_PORT):
        return {"ok": True, "message": "Frontend is already running."}

    npm = subprocess.run(
        ["where.exe", "npm.exe"],
        capture_output=True, text=True,
    )
    if npm.returncode != 0 or not npm.stdout.strip():
        raise RuntimeError("npm.exe was not found in PATH.")

    log("Starting Vite frontend...")
    subprocess.Popen(
        [
            npm.stdout.strip().splitlines()[0],
            "run", "dev", "--",
            "--host", HOST,
            "--port", str(FRONTEND_PORT),
        ],
        cwd=str(FRONTEND_ROOT),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    return {"ok": True, "message": "Frontend start requested."}


def start_llm():
    if port_listener(8080):
        return {"ok": True, "message": "Local model server is already running."}

    config = load_llm_config()
    model_path = str(config.get("model_path", "")).strip()
    server_url = str(config.get("server_url", "")).strip()
    if not model_path or not server_url:
        raise RuntimeError("Local model configuration is incomplete.")

    if not Path(model_path).is_absolute():
        model_path = str(NOVA_ROOT / model_path)

    parsed = urlparse(server_url)
    port = parsed.port or 8080

    log(f"Starting local model server on port {port}...")
    subprocess.Popen(
        [
            str(LLAMA_SERVER), "-m", model_path,
            "--host", parsed.hostname or HOST,
            "--port", str(port),
        ],
        cwd=str(NOVA_ROOT),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    return {"ok": True, "message": "Local model server start requested."}


def start_all():
    if not RUNNER.exists():
        raise RuntimeError(
            f"Runner not found: {RUNNER}"
        )

    log(
        "Starting Nova through "
        "run_nova.ps1..."
    )

    subprocess.Popen(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(RUNNER),
        ],
        cwd=str(NOVA_ROOT),
        creationflags=(
            subprocess.CREATE_NEW_PROCESS_GROUP
        ),
    )

    return {
        "ok": True,
        "message": (
            "run_nova.ps1 started."
        ),
    }


def terminate_process_list(
    processes,
    label: str,
):
    if not processes:
        return False

    for process in processes:
        pid = int(process["pid"])

        log(
            f"Terminating {label} "
            f"PID {pid}..."
        )

        subprocess.run(
            [
                "taskkill",
                "/PID",
                str(pid),
                "/T",
                "/F",
            ],
            capture_output=True,
            text=True,
        )

    return True


def terminate_backend():
    processes = process_snapshot()["backend"]

    if not processes:
        return {
            "ok": True,
            "message": (
                "Backend is not running."
            ),
        }

    terminate_process_list(
        processes,
        "Nova backend",
    )

    return {
        "ok": True,
        "message": "Backend terminated.",
    }


def terminate_frontend():
    processes = process_snapshot()["frontend"]

    if not processes:
        return {
            "ok": True,
            "message": (
                "Frontend is not running."
            ),
        }

    terminate_process_list(
        processes,
        "frontend",
    )

    return {
        "ok": True,
        "message": "Frontend terminated.",
    }


def terminate_llm():
    processes = process_snapshot()["llm"]

    if not processes:
        return {
            "ok": True,
            "message": (
                "Local model server "
                "is not running."
            ),
        }

    terminate_process_list(
        processes,
        "llama-server",
    )

    return {
        "ok": True,
        "message": (
            "Local model server terminated."
        ),
    }


def terminate_wolfram():
    backend_http = http_check(f"http://{HOST}:{BACKEND_PORT}/docs")

    if backend_http.get("ok"):
        try:
            result = http_json_post(
                f"http://{HOST}:{BACKEND_PORT}/api/wolfram/restart"
            )
            if result.get("success"):
                log("Wolfram session terminated cleanly through backend.")
                return {"ok": True, "message": "Wolfram session terminated cleanly."}
        except Exception as exc:
            log(f"Wolfram session reset failed: {exc}")

    processes = process_snapshot()["wolfram"]
    terminate_process_list(processes, "WolframKernel")
    return {"ok": True, "message": "Wolfram kernel processes terminated."}

def terminate_all():
    terminate_frontend()
    terminate_llm()
    terminate_wolfram()
    terminate_backend()

    log("Nova processes terminated.")

    return {
        "ok": True,
        "message": (
            "Nova frontend, backend, "
            "local model and Wolfram "
            "processes terminated."
        ),
    }


def restart_all():
    terminate_all()

    time.sleep(1)

    return start_all()


def test_wolfram():
    status = health_status()

    if not status["backend"]["process"]:
        return {
            "ok": False,
            "message": (
                "Backend is not running."
            ),
        }

    log(
        "Testing Wolfram through "
        "/api/reason with 2 + 2..."
    )

    command = r"""
$body = @{
    query = "2 + 2"
    explanation = $false
} | ConvertTo-Json

try {
    $r = Invoke-RestMethod `
        -Uri "http://127.0.0.1:8000/api/reason" `
        -Method Post `
        -ContentType "application/json" `
        -Body $body `
        -TimeoutSec 15

    $r | ConvertTo-Json `
        -Depth 10 `
        -Compress
}
catch {
    [pscustomobject]@{
        success = $false
        error = $_.Exception.Message
    } | ConvertTo-Json -Compress
}
"""

    try:
        raw = ps(
            command,
            timeout=20,
        )

        result = (
            json.loads(raw)
            if raw
            else {}
        )

        if result.get("success"):
            log(
                "Wolfram/reasoning "
                "test succeeded."
            )

            return {
                "ok": True,
                "message": (
                    "Computation test succeeded."
                ),
                "result": result,
            }

        log(
            "Computation test failed: "
            f"{result.get('error', 'Unknown error')}"
        )

        return {
            "ok": False,
            "message": result.get(
                "error",
                "Computation test failed.",
            ),
            "result": result,
        }

    except Exception as exc:
        log(
            f"Wolfram test failed: {exc}"
        )

        return {
            "ok": False,
            "message": str(exc),
        }


def api_response(
    handler,
    payload,
    status=200,
):
    body = json.dumps(
        payload,
        indent=2,
    ).encode("utf-8")

    handler.send_response(status)

    handler.send_header(
        "Content-Type",
        "application/json; charset=utf-8",
    )

    handler.send_header(
        "Content-Length",
        str(len(body)),
    )

    handler.send_header(
        "Cache-Control",
        "no-store, no-cache, must-revalidate",
    )

    handler.send_header(
        "Pragma",
        "no-cache",
    )

    handler.end_headers()

    handler.wfile.write(body)


class Handler(BaseHTTPRequestHandler):

    def log_message(
        self,
        format,
        *args,
    ):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/":
            try:
                content = (
                    NOVA_ROOT
                    / "control"
                    / "index.html"
                ).read_bytes()

            except Exception as exc:
                api_response(
                    self,
                    {"error": str(exc)},
                    500,
                )
                return

            self.send_response(200)

            self.send_header(
                "Content-Type",
                "text/html; charset=utf-8",
            )

            self.send_header(
                "Content-Length",
                str(len(content)),
            )

            # Important: do not let the browser
            # cache the Control Center itself.
            self.send_header(
                "Cache-Control",
                "no-store, no-cache, must-revalidate",
            )

            self.send_header(
                "Pragma",
                "no-cache",
            )

            self.end_headers()

            self.wfile.write(content)

            return

        if parsed.path == "/api/status":
            api_response(
                self,
                health_status(),
            )
            return

        if parsed.path == "/api/logs":
            with LOG_LOCK:
                logs = list(LOGS)

            api_response(
                self,
                {"logs": logs},
            )
            return

        if parsed.path == "/api/processes":
            api_response(
                self,
                process_snapshot(),
            )
            return

        api_response(
            self,
            {"error": "Not found"},
            404,
        )

    def do_POST(self):
        parsed = urlparse(self.path)

        try:
            if parsed.path == "/api/start/backend":
                api_response(self, start_backend())
                return
            if parsed.path == "/api/start/frontend":
                api_response(self, start_frontend())
                return
            if parsed.path == "/api/start/llm":
                api_response(self, start_llm())
                return

            if parsed.path == "/api/restart/backend":
                terminate_backend()
                time.sleep(0.5)
                api_response(self, start_backend())
                return
            if parsed.path == "/api/restart/frontend":
                terminate_frontend()
                time.sleep(0.5)
                api_response(self, start_frontend())
                return
            if parsed.path == "/api/restart/llm":
                terminate_llm()
                time.sleep(0.5)
                api_response(self, start_llm())
                return
            if parsed.path == "/api/restart/wolfram":
                result = http_json_post(
                    f"http://{HOST}:{BACKEND_PORT}/api/wolfram/restart"
                )
                api_response(self, {
                    "ok": bool(result.get("success")),
                    "message": result.get("message", "Wolfram session reset."),
                })
                return

            if parsed.path == "/api/start":
                api_response(
                    self,
                    start_all(),
                )
                return

            if parsed.path == "/api/terminate/backend":
                api_response(
                    self,
                    terminate_backend(),
                )
                return

            if parsed.path == "/api/terminate/frontend":
                api_response(
                    self,
                    terminate_frontend(),
                )
                return

            if parsed.path == "/api/terminate/llm":
                api_response(
                    self,
                    terminate_llm(),
                )
                return

            if parsed.path == "/api/terminate/wolfram":
                api_response(
                    self,
                    terminate_wolfram(),
                )
                return

            if parsed.path == "/api/terminate/all":
                api_response(
                    self,
                    terminate_all(),
                )
                return

            if parsed.path == "/api/restart":
                api_response(
                    self,
                    restart_all(),
                )
                return

            if parsed.path == "/api/test/wolfram":
                api_response(
                    self,
                    test_wolfram(),
                )
                return

            api_response(
                self,
                {"error": "Not found"},
                404,
            )

        except Exception as exc:
            log(
                f"Control operation failed: {exc}"
            )

            api_response(
                self,
                {
                    "ok": False,
                    "error": str(exc),
                },
                500,
            )


def main():
    if not RUNNER.exists():
        log(
            "WARNING: runner does not exist: "
            f"{RUNNER}"
        )

    log(
        "PrismOffline Control Center starting."
    )

    log(
        f"Nova root: {NOVA_ROOT}"
    )

    log(
        f"Control URL: "
        f"http://{HOST}:{PORT}"
    )

    server = ThreadingHTTPServer(
        (HOST, PORT),
        Handler,
    )

    server.serve_forever()


if __name__ == "__main__":
    main()
