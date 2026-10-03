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

PLANNER_CONFIG_SCRIPT = (
    NOVA_ROOT
    / "backend"
    / "providers"
    / "planner_llm"
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
PLANNER_PORT = 8081

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

        if name_lower == "llama-server.exe":
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
    kernel_count = len(processes["wolfram"])

    # WolframLanguageSession.started is not used as the sole runtime
    # signal here. After a successful evaluation the native kernel can
    # still be present even when that property is momentarily false.
    # The backend session object plus an actual WolframKernel.exe give
    # us the reliable Control Center state.
    wolfram_active = session_exists and kernel_count > 0

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
                else (
                    (
                        port_listener(8080) or {}
                    ).get("OwningProcess")
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


def start_process(args: list[str], cwd: Path):
    """Start a Windows child process with a safe fallback.

    Some Windows/Python launch contexts reject CREATE_NEW_PROCESS_GROUP
    with OSError 22. The Control Center must still be able to start Nova,
    so retry without that optional process-group flag.
    """
    try:
        return subprocess.Popen(
            args,
            cwd=str(cwd),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    except OSError as exc:
        if getattr(exc, "winerror", None) != 87 and getattr(exc, "errno", None) != 22:
            raise

        log(
            "Process-group launch was rejected by Windows; "
            "retrying without CREATE_NEW_PROCESS_GROUP."
        )

        return subprocess.Popen(
            args,
            cwd=str(cwd),
        )


def start_planner():
    listener = port_listener(PLANNER_PORT)
    if listener:
        return {"ok": True, "message": "Planner model server is already running."}

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

    log(f"Starting small planner model server on port {port}...")
    start_process(
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
    )
    return {"ok": True, "message": "Planner model server start requested."}


def start_backend():
    if port_listener(BACKEND_PORT):
        return {"ok": True, "message": "Backend is already running."}

    log("Starting FastAPI backend...")
    start_process(
        [
            str(PYTHON),
            "-m", "uvicorn", "backend.main:app",
            "--host", HOST,
            "--port", str(BACKEND_PORT),
        ],
        NOVA_ROOT,
    )
    return {"ok": True, "message": "Backend start requested."}


def start_wolfram():
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
    if port_listener(FRONTEND_PORT):
        return {"ok": True, "message": "Frontend is already running."}

    npm = subprocess.run(
        ["where.exe", "npm.exe"],
        capture_output=True, text=True,
    )
    if npm.returncode != 0 or not npm.stdout.strip():
        raise RuntimeError("npm.exe was not found in PATH.")

    log("Starting Vite frontend...")
    start_process(
        [
            npm.stdout.strip().splitlines()[0],
            "run", "dev", "--",
            "--host", HOST,
            "--port", str(FRONTEND_PORT),
        ],
        FRONTEND_ROOT,
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
    start_process(
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
    )
    return {"ok": True, "message": "Local model server start requested."}


def wait_for_http(url: str, timeout: int, name: str):
    deadline = time.time() + timeout

    while time.time() < deadline:
        if http_check(url).get("ok"):
            log(f"{name} is ready.")
            return True
        time.sleep(0.5)

    log(f"{name} did not become ready within {timeout} seconds.")
    return False


def start_all():
    """Start Nova services in sequence.

    This function may take several minutes because model servers can need
    time to load large local models. It is intentionally run in a worker
    thread so the Control Center HTTP request never has to stay open for
    the entire startup sequence.
    """
    log("Starting Nova services natively...")

    try:
        start_llm()

        start_planner()
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

        start_backend()

        if not wait_for_http(
            f"http://{HOST}:{BACKEND_PORT}/docs",
            30,
            "Backend",
        ):
            raise RuntimeError(
                "Backend did not become ready."
            )

        start_wolfram()

        start_frontend()

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

    finally:
        global START_ALL_RUNNING
        START_ALL_RUNNING = False


START_ALL_RUNNING = False
START_ALL_LOCK = threading.Lock()


def start_all_background():
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


def terminate_planner():
    listener = port_listener(PLANNER_PORT)
    if not listener:
        return {"ok": True, "message": "Planner model server is not running."}

    pid = int(listener["OwningProcess"])
    log(f"Terminating planner model server PID {pid}...")
    subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        capture_output=True,
        text=True,
    )
    return {"ok": True, "message": "Planner model server terminated."}


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
    """Reset the backend session and remove any kernel left behind.

    WolframLanguageSession.terminate() can occasionally release the
    Python-side session without immediately removing the native
    WolframKernel.exe process. Never report a clean termination until
    the native process is actually gone.
    """
    backend_http = http_check(f"http://{HOST}:{BACKEND_PORT}/docs")

    if backend_http.get("ok"):
        try:
            result = http_json_post(
                f"http://{HOST}:{BACKEND_PORT}/api/wolfram/restart"
            )

            if result.get("success"):
                deadline = time.time() + 5

                while time.time() < deadline:
                    if not process_snapshot()["wolfram"]:
                        log("Wolfram session terminated cleanly through backend.")
                        return {
                            "ok": True,
                            "message": "Wolfram session terminated cleanly.",
                        }
                    time.sleep(0.25)

                log(
                    "Wolfram session reset succeeded, but a native "
                    "WolframKernel.exe process remained. Forcing termination."
                )
        except Exception as exc:
            log(f"Wolfram session reset failed: {exc}")

    processes = process_snapshot()["wolfram"]

    if processes:
        terminate_process_list(processes, "WolframKernel")

        deadline = time.time() + 5
        while time.time() < deadline:
            if not process_snapshot()["wolfram"]:
                break
            time.sleep(0.25)

    if process_snapshot()["wolfram"]:
        return {
            "ok": False,
            "message": "WolframKernel.exe could not be terminated.",
        }

    return {
        "ok": True,
        "message": "Wolfram kernel processes terminated.",
    }


def terminate_all():
    terminate_frontend()
    terminate_llm()
    terminate_planner()
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


def main():
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
