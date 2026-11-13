from __future__ import annotations

import json
import os
import urllib.request
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


NOVA_ROOT = Path(os.environ.get(
    "NOVA_ROOT", str(Path(__file__).resolve().parents[1])
)).resolve()
RUNNER = NOVA_ROOT / "run_nova.ps1"
PYTHON = Path(os.environ.get(
    "NOVA_PYTHON",
    r"C:\Users\sanhi\Documents\code\python\.venv\Scripts\python.exe",
))
CONFIG_SCRIPT = NOVA_ROOT / "backend" / "providers" / "local_llm" / "config.py"
PLANNER_CONFIG_SCRIPT = NOVA_ROOT / "backend" / "providers" / "planner_llm" / "config.py"
LLAMA_SERVER = Path(os.environ.get(
    "NOVA_LLAMA_SERVER",
    str(NOVA_ROOT / "engines" / "llama.cpp" / "llama-server.exe"),
))
FRONTEND_ROOT = NOVA_ROOT / "frontend"

HOST = os.environ.get("NOVA_HOST", "127.0.0.1")
PORT = int(os.environ.get("NOVA_CONTROL_PORT", "8100"))
BACKEND_PORT = int(os.environ.get("NOVA_BACKEND_PORT", "8000"))
FRONTEND_PORT = int(os.environ.get("NOVA_FRONTEND_PORT", "5173"))
LLM_PORT = int(os.environ.get("NOVA_LLM_PORT", "8080"))
PLANNER_PORT = int(os.environ.get("NOVA_PLANNER_PORT", "8081"))
MANAGE_SHARED_MODELS = os.environ.get("NOVA_MANAGE_SHARED_MODELS", "1").lower() not in {"0", "false", "no"}

LOG_LOCK = threading.Lock()
LOGS: list[str] = []
CONTROL_SERVER: ThreadingHTTPServer | None = None
BACKEND_PROCESS: subprocess.Popen | None = None
FRONTEND_PROCESS: subprocess.Popen | None = None
LLM_PROCESS: subprocess.Popen | None = None
PLANNER_PROCESS: subprocess.Popen | None = None
SHUTDOWN_EVENT = threading.Event()


def log(message: str):
    stamp = time.strftime("%H:%M:%S")
    line = f"{stamp}  {message}"

    with LOG_LOCK:
        LOGS.append(line)

        if len(LOGS) > 300:
            del LOGS[:-300]

    print(line, flush=True)


def run_command(
    args: list[str],
    timeout: int = 15,
) -> subprocess.CompletedProcess[str]:
    """Run a native Windows command without spawning PowerShell."""
    return subprocess.run(
        args,
        cwd=str(NOVA_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


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
            "context_size": int(data.get("context_size", 32768)),
            "threads": int(data.get("threads", 6)),
            "batch_size": int(data.get("batch_size", 128)),
            "n_gpu_layers": int(data.get("n_gpu_layers", 0)),
        }

    except Exception as exc:
        return {
            "model_path": "",
            "server_url": "",
            "model_name": "",
            "error": str(exc),
        }


def load_planner_config():
    try:
        result = subprocess.run(
            [str(PYTHON), str(PLANNER_CONFIG_SCRIPT), "--json"],
            cwd=str(NOVA_ROOT),
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "planner config failed")

        data = json.loads(result.stdout)
        model_path = str(data.get("model_path", "")).strip()
        server_url = str(data.get("server_url", "")).strip()

        return {
            "model_path": model_path,
            "server_url": server_url,
            "model_name": Path(model_path).stem if model_path else "",
            "context_size": int(data.get("context_size", 4096)),
            "threads": int(data.get("threads", 6)),
            "batch_size": int(data.get("batch_size", 128)),
            "n_gpu_layers": int(data.get("n_gpu_layers", 0)),
        }
    except Exception as exc:
        return {
            "model_path": "",
            "server_url": "",
            "model_name": "",
            "error": str(exc),
        }


def get_processes():
    """Inspect relevant processes using native tasklist, not PowerShell."""
    names = {
        "python.exe",
        "pythonw.exe",
        "llama-server.exe",
        "WolframKernel.exe",
        "node.exe",
    }

    try:
        result = run_command(
            ["tasklist", "/FO", "CSV", "/NH"],
            timeout=10,
        )

        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.strip()
                or f"tasklist exited with {result.returncode}"
            )

        import csv
        from io import StringIO

        processes = []
        for row in csv.reader(StringIO(result.stdout)):
            if len(row) < 2 or row[0] not in names:
                continue

            try:
                pid = int(row[1])
            except ValueError:
                continue

            processes.append({
                "ProcessId": pid,
                "ParentProcessId": None,
                "Name": row[0],
                "CommandLine": "",
            })

        return processes

    except Exception as exc:
        log(f"Process inspection failed: {exc}")
        return []


def port_listener(port: int):
    """Find a listening TCP port using native netstat.

    No-listener is a normal startup/restart state and is not logged.
    """
    try:
        result = run_command(
            ["netstat", "-ano", "-p", "tcp"],
            timeout=10,
        )

        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.strip()
                or f"netstat exited with {result.returncode}"
            )

        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) < 5 or parts[0].upper() != "TCP":
                continue

            local = parts[1]
            state = parts[3].upper()
            pid = parts[4]

            if state != "LISTENING":
                continue

            try:
                local_port = int(local.rsplit(":", 1)[1])
                owning_pid = int(pid)
            except (ValueError, IndexError):
                continue

            if local_port != port:
                continue

            return {
                "OwningProcess": owning_pid,
                "LocalAddress": local.rsplit(":", 1)[0],
                "LocalPort": local_port,
                "State": "Listen",
            }

        return None

    except Exception as exc:
        log(f"Port inspection failed for {port}: {exc}")
        return None


def process_snapshot():
    """Report only processes bound to this instance's configured ports.

    Wolfram kernels are managed through the backend session API; machine-wide
    process enumeration is not a safe ownership mechanism.
    """
    processes = get_processes()

    def listener_process(port: int, allowed_names: set[str]):
        listener = port_listener(port)
        if not isinstance(listener, dict):
            return []
        pid = str(listener.get("OwningProcess"))
        return [
            {
                "pid": process.get("ProcessId"),
                "parent_pid": process.get("ParentProcessId"),
                "name": process.get("Name"),
                "command": process.get("CommandLine", ""),
            }
            for process in processes
            if str(process.get("ProcessId")) == pid
            and str(process.get("Name", "")).lower() in allowed_names
        ]

    return {
        "backend": listener_process(BACKEND_PORT, {"python.exe", "pythonw.exe"}),
        "frontend": listener_process(FRONTEND_PORT, {"node.exe"}),
        "llm": listener_process(LLM_PORT, {"llama-server.exe"}),
        "planner": listener_process(PLANNER_PORT, {"llama-server.exe"}),
        "wolfram": [],
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

    planner_config = load_planner_config()
    planner_url = planner_config.get("server_url", "").rstrip("/")
    planner_health = (
        http_check(f"{planner_url}/health")
        if planner_url
        else {"ok": False, "error": "No planner server_url configured"}
    )
    planner_listener = port_listener(PLANNER_PORT)

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
    session_exists = bool(session.get("session_exists"))
    session_started = bool(session.get("session_started"))
    # The backend API is authoritative for this instance's Wolfram session.
    # Never use a machine-wide WolframKernel.exe count as an ownership signal.
    wolfram_active = bool(wolfram_health.get("wolfram")) or (
        session_exists and session_started
    )
    # This is an instance-owned backend session count, not a machine-wide
    # count of native WolframKernel.exe processes.
    kernel_count = int(wolfram_active)

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

        "planner": {
            "process": bool(planner_health.get("ok")),
            "pid": (
                planner_listener.get("OwningProcess")
                if isinstance(planner_listener, dict)
                else None
            ),
            "server_url": planner_config.get("server_url", ""),
            "model_name": planner_config.get("model_name", ""),
            "model_path": planner_config.get("model_path", ""),
            "health": planner_health,
        },

        "llm": {
            "process": llm_running,
            "pid": (
                processes["llm"][0]["pid"]
                if processes["llm"]
                else (port_listener(LLM_PORT) or {}).get("OwningProcess")
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
            "session_exists": session_exists,
            "session_started": session_started,
            "note": (
                "Wolfram runs as a persistent "
                "session inside the Nova backend."
            ),
            "health": wolfram_health,
        },

        "processes": processes,

        "runner": {
            "exists": RUNNER.exists(),
            "path": str(RUNNER),
        },

        "timestamp": time.time(),
    }


def service_log_path(filename: str) -> Path:
    """Return a persistent log path beneath Nova's logs directory."""
    log_dir = NOVA_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / filename


LOG_TARGETS = {
    "backend": ("logs/backend.log", "FastAPI Backend"),
    "frontend": ("logs/frontend.log", "Frontend"),
    "llm": ("logs/local-model.log", "Local Model"),
    "planner": ("logs/planner-model.log", "Planner Model"),
    # Wolfram runs in-process; its stage diagnostics live beside backend/main.py.
    "wolfram": ("backend/reasoning_timing.log", "Wolfram Engine diagnostics"),
}


def start_process(args: list[str], cwd: Path, *, log_name: str | None = None):
    """Start a child process, optionally redirecting stdout/stderr to a log.

    The Control Center does not need a new process group because termination
    uses taskkill /T. Avoiding CREATE_NEW_PROCESS_GROUP also removes a
    Windows-specific launch failure seen with some Python/console hosts.
    """
    options = {
        "cwd": str(cwd),
        "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
    }
    if log_name:
        path = service_log_path(log_name)
        with path.open("a", encoding="utf-8", buffering=1) as stream:
            stream.write(
                f"\n\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"Starting: {' '.join(str(part) for part in args)}\n"
            )
            stream.flush()
            return subprocess.Popen(
                args,
                stdout=stream,
                stderr=subprocess.STDOUT,
                **options,
            )
    return subprocess.Popen(args, **options)


def open_service_log(component: str):
    """Open a live tail of a fixed, allow-listed service log in a new terminal."""
    if component not in LOG_TARGETS:
        raise ValueError(f"Unknown service log: {component}")

    relative_path, label = LOG_TARGETS[component]
    path = NOVA_ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)

    escaped_path = str(path).replace("'", "''")
    escaped_label = label.replace("'", "''")
    command = (
        f"$Host.UI.RawUI.WindowTitle = 'Nova - {escaped_label} log'; "
        f"Get-Content -LiteralPath '{escaped_path}' -Tail 100 -Wait"
    )
    subprocess.Popen(
        ["powershell.exe", "-NoExit", "-Command", command],
        cwd=str(NOVA_ROOT),
        creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
    )
    return {"ok": True, "message": f"Opened live log for {label}.", "path": str(path)}


def start_planner():
    global PLANNER_PROCESS

    config = load_planner_config()
    model_path = str(config.get("model_path", "")).strip()
    server_url = str(config.get("server_url", "")).strip()

    if not model_path or not server_url:
        raise RuntimeError("Planner model configuration is incomplete.")

    if not Path(model_path).is_absolute():
        model_path = str(NOVA_ROOT / model_path)

    if not Path(model_path).exists():
        raise RuntimeError(f"Planner model not found: {model_path}")

    parsed = urlparse(server_url)
    port = parsed.port or PLANNER_PORT
    if port_listener(port):
        return {"ok": True, "message": "Planner model server is already running."}
    if not MANAGE_SHARED_MODELS:
        raise RuntimeError(
            "Shared planner server is not running. Start the main Nova Control Center first; "
            "the isolated test instance will not load a second planner model."
        )
    if not LLAMA_SERVER.exists():
        raise RuntimeError(f"llama-server executable not found: {LLAMA_SERVER}")

    log(f"Starting small planner model server on port {port}...")
    PLANNER_PROCESS = start_process(
        [
            str(LLAMA_SERVER), "-m", model_path,
            "--host", parsed.hostname or HOST,
            "--port", str(port),
            "--ctx-size", str(config.get("context_size", 4096)),
            "--threads", str(config.get("threads", 6)),
            "--batch-size", str(config.get("batch_size", 128)),
            "--n-gpu-layers", str(config.get("n_gpu_layers", 0)),
        ],
        NOVA_ROOT,
        log_name="planner-model.log",
    )
    return {"ok": True, "message": "Planner model server start requested."}


def start_backend():
    global BACKEND_PROCESS

    if port_listener(BACKEND_PORT):
        return {"ok": True, "message": "Backend is already running."}

    if not PYTHON.exists():
        raise RuntimeError(f"Python executable not found: {PYTHON}")

    if not (NOVA_ROOT / "backend" / "main.py").exists():
        raise RuntimeError(f"Backend entry point not found: {NOVA_ROOT / 'backend' / 'main.py'}")

    log("Starting FastAPI backend...")
    process = start_process(
        [
            str(PYTHON),
            "-u",
            "-m", "uvicorn", "backend.main:app",
            "--host", HOST,
            "--port", str(BACKEND_PORT),
            "--reload",
            "--reload-dir", str(NOVA_ROOT / "backend"),
        ],
        NOVA_ROOT,
        log_name="backend.log",
    )

    BACKEND_PROCESS = process

    # Uvicorn can spend time in FastAPI lifespan startup while the persistent
    # Wolfram session is being initialized. Do not report success merely
    # because Popen succeeded: verify the actual HTTP endpoint.
    deadline = time.time() + 45
    while time.time() < deadline:
        if http_check(f"http://{HOST}:{BACKEND_PORT}/docs").get("ok"):
            log("Backend is ready.")
            return {
                "ok": True,
                "message": "Backend is running.",
                "pid": process.pid,
            }
        if process.poll() is not None:
            raise RuntimeError(
                f"Backend process exited during startup with code {process.returncode}."
            )
        time.sleep(0.5)

    raise RuntimeError(
        "Backend process was launched but did not become ready within 45 seconds. "
        "Check the Control Center event log and backend dependencies."
    )


def start_wolfram():
    if BACKEND_PROCESS is None or BACKEND_PROCESS.poll() is not None:
        raise RuntimeError(
            "Wolfram can only be started through a backend launched by this Control Center."
        )

    backend_http = http_check(f"http://{HOST}:{BACKEND_PORT}/docs")

    if not backend_http.get("ok"):
        raise RuntimeError("Backend must be running before starting Wolfram.")

    log("Starting and verifying persistent Wolfram session...")

    result = http_json_post(
        f"http://{HOST}:{BACKEND_PORT}/api/wolfram/start"
    )

    if not result.get("success"):
        raise RuntimeError(
            result.get("message", "Wolfram session failed to start.")
        )

    log("Wolfram persistent session is ready.")
    return {
        "ok": True,
        "message": result.get(
            "message",
            "Wolfram session started and verified.",
        ),
    }


def start_frontend():
    global FRONTEND_PROCESS

    if port_listener(FRONTEND_PORT):
        return {"ok": True, "message": "Frontend is already running."}

    npm = subprocess.run(
        ["where.exe", "npm.exe"],
        capture_output=True, text=True,
    )
    if npm.returncode != 0 or not npm.stdout.strip():
        raise RuntimeError("npm.exe was not found in PATH.")

    log("Starting Vite frontend...")
    FRONTEND_PROCESS = start_process(
        [
            npm.stdout.strip().splitlines()[0],
            "run", "dev", "--",
            "--host", HOST,
            "--port", str(FRONTEND_PORT),
        ],
        FRONTEND_ROOT,
        log_name="frontend.log",
    )
    return {"ok": True, "message": "Frontend start requested."}


def start_llm():
    global LLM_PROCESS

    config = load_llm_config()
    model_path = str(config.get("model_path", "")).strip()
    server_url = str(config.get("server_url", "")).strip()

    if not model_path or not server_url:
        raise RuntimeError("Local model configuration is incomplete.")

    if not Path(model_path).is_absolute():
        model_path = str(NOVA_ROOT / model_path)

    parsed = urlparse(server_url)
    port = parsed.port or LLM_PORT
    if port_listener(port):
        return {"ok": True, "message": "Local model server is already running."}
    if not MANAGE_SHARED_MODELS:
        raise RuntimeError(
            "Shared local model server is not running. Start the main Nova Control Center first; "
            "the isolated test instance will not load a second model."
        )
    if not LLAMA_SERVER.exists():
        raise RuntimeError(f"llama-server executable not found: {LLAMA_SERVER}")

    log(f"Starting local model server on port {port}...")
    LLM_PROCESS = start_process(
        [
            str(LLAMA_SERVER), "-m", model_path,
            "--host", parsed.hostname or HOST,
            "--port", str(port),
            "--ctx-size", str(
                load_llm_config().get("context_size", 32768)
            ),
            "--threads", str(
                load_llm_config().get("threads", 6)
            ),
            "--batch-size", str(
                load_llm_config().get("batch_size", 128)
            ),
            "--n-gpu-layers", str(
                load_llm_config().get("n_gpu_layers", 0)
            ),
        ],
        NOVA_ROOT,
        log_name="local-model.log",
    )
    return {"ok": True, "message": "Local model server start requested."}


def wait_for_http(url: str, timeout: int, name: str):
    deadline = time.time() + timeout

    while time.time() < deadline and not SHUTDOWN_EVENT.is_set():
        if http_check(url).get("ok"):
            log(f"{name} is ready.")
            return True
        time.sleep(0.5)

    if SHUTDOWN_EVENT.is_set():
        log(f"Stopping wait for {name} because Control Center is closing.")
        return False

    log(f"{name} did not become ready within {timeout} seconds.")
    return False


def start_all(restart_existing_backend: bool = False):
    """Start Nova services in sequence.

    This function may take several minutes because model servers can need
    time to load large local models. It is intentionally run in a worker
    thread so the Control Center HTTP request never has to stay open for
    the entire startup sequence.
    """
    log("Starting Nova services natively...")

    try:
        start_llm()
        if SHUTDOWN_EVENT.is_set():
            raise RuntimeError("Control Center is closing.")

        start_planner()
        if SHUTDOWN_EVENT.is_set():
            raise RuntimeError("Control Center is closing.")
        planner_config = load_planner_config()
        planner_url = str(
            planner_config.get("server_url", "")
        ).rstrip("/")

        if not planner_url or not wait_for_http(
            f"{planner_url}/health",
            30,
            "Planner model server",
        ):
            raise RuntimeError(
                "The planner model server did not become ready."
            )

        config = load_llm_config()
        llm_url = str(
            config.get("server_url", "")
        ).rstrip("/")

        if not llm_url or not wait_for_http(
            f"{llm_url}/health",
            60,
            "Local model server",
        ):
            raise RuntimeError(
                "The local model server did not become ready."
            )

        if restart_existing_backend and port_listener(BACKEND_PORT):
            if BACKEND_PROCESS is not None and BACKEND_PROCESS.poll() is None:
                log("Restarting the backend process owned by this Control Center...")
                terminate_backend()
                release_deadline = time.time() + 10
                while port_listener(BACKEND_PORT) and time.time() < release_deadline:
                    time.sleep(0.25)
                if port_listener(BACKEND_PORT):
                    raise RuntimeError("The owned backend did not stop; refusing to launch a second backend.")
            else:
                log("An external backend already owns this port; reusing it without terminating it.")

        start_backend()
        if SHUTDOWN_EVENT.is_set():
            raise RuntimeError("Control Center is closing.")

        if not wait_for_http(
            f"http://{HOST}:{BACKEND_PORT}/docs",
            30,
            "Backend",
        ):
            raise RuntimeError(
                "Backend did not become ready."
            )

        if BACKEND_PROCESS is not None and BACKEND_PROCESS.poll() is None:
            start_wolfram()
        else:
            log("Skipping Wolfram session management because the backend is externally owned.")

        if SHUTDOWN_EVENT.is_set():
            raise RuntimeError("Control Center is closing.")

        start_frontend()
        if SHUTDOWN_EVENT.is_set():
            raise RuntimeError("Control Center is closing.")

        if not wait_for_http(
            f"http://{HOST}:{FRONTEND_PORT}/",
            30,
            "Frontend",
        ):
            raise RuntimeError(
                "Frontend did not become ready."
            )

        log("Nova services are running.")

    except Exception as exc:
        log(f"Start All failed: {exc}")
        log("Rolling back partially started Nova services...")
        try:
            terminate_all()
        except Exception as cleanup_exc:
            log(f"Startup rollback failed: {cleanup_exc}")

    finally:
        global START_ALL_RUNNING
        START_ALL_RUNNING = False


START_ALL_RUNNING = False
START_ALL_LOCK = threading.Lock()


def start_all_background(restart_existing_backend: bool = False):
    """Queue Start All and return immediately to the browser."""
    global START_ALL_RUNNING

    with START_ALL_LOCK:
        if START_ALL_RUNNING:
            return {
                "ok": True,
                "running": True,
                "message": "Start All is already running.",
            }

        START_ALL_RUNNING = True

    thread = threading.Thread(
        target=start_all,
        args=(restart_existing_backend,),
        name="nova-start-all",
        daemon=True,
    )
    thread.start()

    log("Start All queued; startup is continuing in the background.")

    return {
        "ok": True,
        "running": True,
        "message": "Nova startup started in the background.",
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


def terminate_owned_process(process, label: str):
    """Terminate only a Popen handle created by this Control Center."""
    if process is None or process.poll() is not None:
        return False

    pid = int(process.pid)
    log(f"Terminating owned {label} process tree PID {pid}...")
    result = subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        capture_output=True,
        text=True,
    )
    try:
        process.wait(timeout=10)
    except (AttributeError, subprocess.TimeoutExpired):
        pass
    return result.returncode == 0


def terminate_backend():
    global BACKEND_PROCESS
    process = BACKEND_PROCESS
    if process is None:
        return {"ok": True, "message": "Backend was not started by this Control Center; leaving it running."}
    BACKEND_PROCESS = None
    if terminate_owned_process(process, "Nova backend/reloader"):
        return {"ok": True, "message": "Owned backend and reload worker terminated."}
    return {"ok": True, "message": "Owned backend process is already stopped."}


def terminate_frontend():
    global FRONTEND_PROCESS
    process = FRONTEND_PROCESS
    if process is None:
        return {"ok": True, "message": "Frontend was not started by this Control Center; leaving it running."}
    FRONTEND_PROCESS = None
    terminate_owned_process(process, "Nova frontend")
    return {"ok": True, "message": "Owned frontend process terminated."}


def terminate_planner():
    global PLANNER_PROCESS
    process = PLANNER_PROCESS
    if process is None:
        return {"ok": True, "message": "Planner server is shared or was not started by this Control Center; leaving it running."}
    PLANNER_PROCESS = None
    terminate_owned_process(process, "planner model server")
    return {"ok": True, "message": "Owned planner server terminated."}


def terminate_llm():
    global LLM_PROCESS
    process = LLM_PROCESS
    if process is None:
        return {"ok": True, "message": "Local model server is shared or was not started by this Control Center; leaving it running."}
    LLM_PROCESS = None
    terminate_owned_process(process, "local model server")
    return {"ok": True, "message": "Owned local model server terminated."}


def terminate_wolfram():
    """Reset only the Wolfram session owned by this Control Center's backend."""
    if BACKEND_PROCESS is None or BACKEND_PROCESS.poll() is not None:
        return {
            "ok": True,
            "message": "No backend-owned Wolfram session to terminate; external kernels are untouched.",
        }
    try:
        result = http_json_post(
            f"http://{HOST}:{BACKEND_PORT}/api/wolfram/restart"
        )
    except Exception as exc:
        return {"ok": False, "message": f"Could not reset owned Wolfram session: {exc}"}
    if not result.get("success"):
        return {
            "ok": False,
            "message": result.get("message", "Owned Wolfram session reset failed."),
        }
    log("Owned Wolfram session reset through its backend API.")
    return {"ok": True, "message": "Owned Wolfram session reset successfully."}


def terminate_all():
    # Only process handles created by this Control Center are terminated.
    terminate_frontend()
    terminate_wolfram()
    terminate_backend()
    terminate_planner()
    terminate_llm()
    log("Owned Nova processes terminated; shared/external services were left running.")
    return {
        "ok": True,
        "message": "Owned frontend, backend and service processes terminated; shared model servers were preserved.",
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
            "message": "Backend is not running.",
        }

    log("Testing Wolfram through /api/reason with 2 + 2...")

    try:
        body = json.dumps({
            "query": "2 + 2",
            "explanation": False,
        }).encode("utf-8")

        request = urllib.request.Request(
            f"http://{HOST}:{BACKEND_PORT}/api/reason",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read().decode("utf-8", errors="replace")

        result = json.loads(raw) if raw else {}

        if result.get("success"):
            log("Wolfram/reasoning test succeeded.")
            return {
                "ok": True,
                "message": "Computation test succeeded.",
                "result": result,
            }

        error = result.get("error", "Computation test failed.")
        log(f"Computation test failed: {error}")
        return {
            "ok": False,
            "message": error,
            "result": result,
        }

    except Exception as exc:
        log(f"Wolfram test failed: {exc}")
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
            try:
                api_response(
                    self,
                    health_status(),
                )
            except Exception as exc:
                log(f"Status request failed: {exc}")
                api_response(
                    self,
                    {
                        "ok": False,
                        "error": str(exc),
                    },
                    500,
                )
            return

        if parsed.path == "/api/logs":
            try:
                with LOG_LOCK:
                    logs = list(LOGS)

                api_response(
                    self,
                    {"logs": logs},
                )
            except Exception as exc:
                log(f"Log request failed: {exc}")
                api_response(
                    self,
                    {
                        "ok": False,
                        "error": str(exc),
                    },
                    500,
                )
            return

        if parsed.path == "/api/processes":
            try:
                api_response(
                    self,
                    process_snapshot(),
                )
            except Exception as exc:
                log(f"Process request failed: {exc}")
                api_response(
                    self,
                    {
                        "ok": False,
                        "error": str(exc),
                    },
                    500,
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
            if parsed.path.startswith("/api/open-log/"):
                component = parsed.path.rsplit("/", 1)[-1]
                api_response(self, open_service_log(component))
                return

            if parsed.path == "/api/start/backend":
                api_response(self, start_backend())
                return
            if parsed.path == "/api/start/frontend":
                api_response(self, start_frontend())
                return
            if parsed.path == "/api/start/wolfram":
                api_response(self, start_wolfram())
                return
            if parsed.path == "/api/start/llm":
                api_response(self, start_llm())
                return
            if parsed.path == "/api/start/planner":
                api_response(self, start_planner())
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
            if parsed.path == "/api/restart/planner":
                terminate_planner()
                time.sleep(0.5)
                api_response(self, start_planner())
                return
            if parsed.path == "/api/restart/wolfram":
                result = terminate_wolfram()
                api_response(self, result, 200 if result.get("ok") else 409)
                return

            if parsed.path == "/api/shutdown":
                api_response(
                    self,
                    {"ok": True, "message": "Control Center is shutting down; Nova services will stop."},
                    202,
                )
                server = CONTROL_SERVER
                if server is not None:
                    threading.Thread(
                        target=shutdown_control_center,
                        args=(server,),
                        name="nova-control-shutdown",
                        daemon=True,
                    ).start()
                return

            if parsed.path == "/api/start":
                api_response(
                    self,
                    start_all_background(),
                    202,
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

            if parsed.path == "/api/terminate/planner":
                api_response(
                    self,
                    terminate_planner(),
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


def shutdown_control_center(server: ThreadingHTTPServer):
    """Stop Nova services, then stop the Control Center HTTP server."""
    SHUTDOWN_EVENT.set()
    log("Control Center closing; stopping all Nova services...")
    try:
        terminate_all()
    except Exception as exc:
        log(f"Service shutdown encountered an error: {exc}")
    finally:
        log("Stopping Control Center HTTP server.")
        server.shutdown()


def main():
    global CONTROL_SERVER

    log("PrismOffline Control Center starting.")
    log(f"Nova root: {NOVA_ROOT}")
    log(f"Control URL: http://{HOST}:{PORT}")

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    CONTROL_SERVER = server
    SHUTDOWN_EVENT.clear()

    try:
        # Start the complete local stack whenever Control Center starts.
        # The startup worker is asynchronous so the dashboard remains responsive
        # while local model files load.
        start_all_background(restart_existing_backend=True)
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        log("Control Center interrupted; stopping Nova services...")
    finally:
        SHUTDOWN_EVENT.set()
        server.server_close()
        try:
            terminate_all()
        except Exception as exc:
            log(f"Final service cleanup failed: {exc}")
        CONTROL_SERVER = None
        log("Control Center stopped.")



if __name__ == "__main__":
    main()
