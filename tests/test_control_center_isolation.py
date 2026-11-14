from types import SimpleNamespace

from control import server as control_server


def _fake_process(pid=4321):
    return SimpleNamespace(pid=pid, poll=lambda: None, wait=lambda timeout=None: None)


def test_terminate_llm_controls_existing_fixed_port_server(monkeypatch):
    killed = []
    monkeypatch.setattr(control_server, "LLM_PROCESS", None)
    monkeypatch.setattr(control_server, "process_snapshot", lambda: {
        "llm": [{"pid": 100, "name": "llama-server.exe"}]
    })
    monkeypatch.setattr(control_server.subprocess, "run", lambda args, **k: (killed.append(args) or SimpleNamespace(returncode=0)))
    result = control_server.terminate_llm()
    assert result["ok"] is True
    assert killed == [["taskkill", "/PID", "100", "/T", "/F"]]


def test_terminate_planner_controls_existing_fixed_port_server(monkeypatch):
    killed = []
    monkeypatch.setattr(control_server, "PLANNER_PROCESS", None)
    monkeypatch.setattr(control_server, "process_snapshot", lambda: {
        "planner": [{"pid": 101, "name": "llama-server.exe"}]
    })
    monkeypatch.setattr(control_server.subprocess, "run", lambda args, **k: (killed.append(args) or SimpleNamespace(returncode=0)))
    result = control_server.terminate_planner()
    assert result["ok"] is True
    assert killed == [["taskkill", "/PID", "101", "/T", "/F"]]


def test_terminate_backend_controls_existing_fixed_port_server(monkeypatch):
    killed = []
    monkeypatch.setattr(control_server, "BACKEND_PROCESS", None)
    monkeypatch.setattr(control_server, "process_snapshot", lambda: {
        "backend": [{"pid": 200, "name": "python.exe"}]
    })
    monkeypatch.setattr(control_server.subprocess, "run", lambda args, **k: (killed.append(args) or SimpleNamespace(returncode=0)))
    result = control_server.terminate_backend()
    assert result["ok"] is True
    assert killed == [["taskkill", "/PID", "200", "/T", "/F"]]


def test_terminate_frontend_controls_existing_fixed_port_server(monkeypatch):
    killed = []
    monkeypatch.setattr(control_server, "FRONTEND_PROCESS", None)
    monkeypatch.setattr(control_server, "process_snapshot", lambda: {
        "frontend": [{"pid": 300, "name": "node.exe"}]
    })
    monkeypatch.setattr(control_server.subprocess, "run", lambda *a, **k: (killed.append(a) or SimpleNamespace(returncode=0)))
    result = control_server.terminate_frontend()
    assert result["ok"] is True
    assert killed == [(["taskkill", "/PID", "300", "/T", "/F"],)]


def test_terminate_wolfram_never_scans_or_kills_global_kernels(monkeypatch):
    killed = []
    monkeypatch.setattr(control_server, "BACKEND_PROCESS", None)
    monkeypatch.setattr(control_server, "http_check", lambda url: {"ok": False})
    monkeypatch.setattr(control_server, "process_snapshot", lambda: {
        "wolfram": [{"pid": 400, "name": "WolframKernel.exe"}]
    })
    monkeypatch.setattr(control_server.subprocess, "run", lambda *a, **k: killed.append(a))
    result = control_server.terminate_wolfram()
    assert result["ok"] is True
    assert killed == []


def test_owned_llm_process_can_be_terminated(monkeypatch):
    killed = []
    process = _fake_process()
    monkeypatch.setattr(control_server, "LLM_PROCESS", process)
    monkeypatch.setattr(
        control_server.subprocess,
        "run",
        lambda args, **kwargs: (killed.append(args) or SimpleNamespace(returncode=0)),
    )
    result = control_server.terminate_llm()
    assert result["ok"] is True
    assert killed == [["taskkill", "/PID", "4321", "/T", "/F"]]
    assert control_server.LLM_PROCESS is None



def test_terminate_wolfram_uses_fixed_port_backend_even_if_not_spawned_here(monkeypatch):
    calls = []
    monkeypatch.setattr(control_server, "BACKEND_PROCESS", None)
    monkeypatch.setattr(control_server, "http_check", lambda url: {"ok": True})
    monkeypatch.setattr(
        control_server,
        "http_json_post",
        lambda url: (calls.append(url) or {"success": True}),
    )
    result = control_server.terminate_wolfram()
    assert result["ok"] is True
    assert calls == [
        f"http://{control_server.HOST}:{control_server.BACKEND_PORT}/api/wolfram/stop"
    ]


def test_fixed_port_planner_reuses_healthy_server_without_listener_pid(monkeypatch, tmp_path):
    monkeypatch.setattr(control_server, "PLANNER_PROCESS", None)
    monkeypatch.setattr(control_server, "load_planner_config", lambda: {
        "model_path": str(tmp_path / "missing-planner.gguf"),
        "server_url": "http://127.0.0.1:8081",
    })
    monkeypatch.setattr(control_server, "port_listener", lambda port: None)
    monkeypatch.setattr(control_server, "http_check", lambda url: {"ok": url.endswith("/health")})
    result = control_server.start_planner()
    assert result["ok"] is True
    assert "already running" in result["message"]


def test_fixed_port_llm_reuses_healthy_server_without_listener_pid(monkeypatch, tmp_path):
    monkeypatch.setattr(control_server, "LLM_PROCESS", None)
    monkeypatch.setattr(control_server, "load_llm_config", lambda: {
        "model_path": str(tmp_path / "missing-local.gguf"),
        "server_url": "http://127.0.0.1:8080",
    })
    monkeypatch.setattr(control_server, "port_listener", lambda port: None)
    monkeypatch.setattr(control_server, "http_check", lambda url: {"ok": url.endswith("/health")})
    result = control_server.start_llm()
    assert result["ok"] is True
    assert "already running" in result["message"]



def test_all_services_use_one_fixed_port_set():
    assert control_server.PORT == 8100
    assert control_server.BACKEND_PORT == 8000
    assert control_server.FRONTEND_PORT == 5173
    assert control_server.LLM_PORT == 8080
    assert control_server.PLANNER_PORT == 8081
