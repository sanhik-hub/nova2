from types import SimpleNamespace

from control import server as control_server


def _fake_process(pid=4321):
    return SimpleNamespace(pid=pid, poll=lambda: None, wait=lambda timeout=None: None)


def test_terminate_llm_preserves_shared_server(monkeypatch):
    killed = []
    monkeypatch.setattr(control_server, "LLM_PROCESS", None)
    monkeypatch.setattr(control_server, "process_snapshot", lambda: {
        "llm": [{"pid": 100, "name": "llama-server.exe"}]
    })
    monkeypatch.setattr(control_server.subprocess, "run", lambda *a, **k: killed.append(a))
    result = control_server.terminate_llm()
    assert result["ok"] is True
    assert killed == []


def test_terminate_planner_preserves_shared_server(monkeypatch):
    killed = []
    monkeypatch.setattr(control_server, "PLANNER_PROCESS", None)
    monkeypatch.setattr(control_server, "port_listener", lambda port: {
        "OwningProcess": 100
    } if port == control_server.PLANNER_PORT else None)
    monkeypatch.setattr(control_server.subprocess, "run", lambda *a, **k: killed.append(a))
    result = control_server.terminate_planner()
    assert result["ok"] is True
    assert killed == []


def test_terminate_backend_does_not_kill_unowned_listener(monkeypatch):
    killed = []
    monkeypatch.setattr(control_server, "BACKEND_PROCESS", None)
    monkeypatch.setattr(control_server, "process_snapshot", lambda: {
        "backend": [{"pid": 200, "name": "python.exe"}]
    })
    monkeypatch.setattr(control_server.subprocess, "run", lambda *a, **k: killed.append(a))
    result = control_server.terminate_backend()
    assert result["ok"] is True
    assert killed == []


def test_terminate_frontend_does_not_kill_unowned_listener(monkeypatch):
    killed = []
    monkeypatch.setattr(control_server, "FRONTEND_PROCESS", None)
    monkeypatch.setattr(control_server, "process_snapshot", lambda: {
        "frontend": [{"pid": 300, "name": "node.exe"}]
    })
    monkeypatch.setattr(control_server.subprocess, "run", lambda *a, **k: killed.append(a))
    result = control_server.terminate_frontend()
    assert result["ok"] is True
    assert killed == []


def test_terminate_wolfram_never_scans_or_kills_global_kernels(monkeypatch):
    killed = []
    monkeypatch.setattr(control_server, "BACKEND_PROCESS", None)
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



def test_isolated_instance_refuses_to_load_duplicate_llm(monkeypatch, tmp_path):
    monkeypatch.setattr(control_server, "MANAGE_SHARED_MODELS", False)
    monkeypatch.setattr(control_server, "LLM_PROCESS", None)
    monkeypatch.setattr(control_server, "load_llm_config", lambda: {
        "model_path": str(tmp_path / "model.gguf"),
        "server_url": "http://127.0.0.1:8080",
    })
    monkeypatch.setattr(control_server, "port_listener", lambda port: None)
    monkeypatch.setattr(control_server, "LLAMA_SERVER", tmp_path / "llama-server.exe")
    try:
        control_server.start_llm()
    except RuntimeError as exc:
        assert "will not load a second model" in str(exc)
    else:
        raise AssertionError("Isolated instance must not start a duplicate model server")


def test_isolated_instance_refuses_to_load_duplicate_planner(monkeypatch, tmp_path):
    model = tmp_path / "planner.gguf"
    model.write_bytes(b"test")
    monkeypatch.setattr(control_server, "MANAGE_SHARED_MODELS", False)
    monkeypatch.setattr(control_server, "PLANNER_PROCESS", None)
    monkeypatch.setattr(control_server, "load_planner_config", lambda: {
        "model_path": str(model),
        "server_url": "http://127.0.0.1:8081",
    })
    monkeypatch.setattr(control_server, "port_listener", lambda port: None)
    monkeypatch.setattr(control_server, "LLAMA_SERVER", tmp_path / "llama-server.exe")
    try:
        control_server.start_planner()
    except RuntimeError as exc:
        assert "will not load a second planner model" in str(exc)
    else:
        raise AssertionError("Isolated instance must not start a duplicate planner")



def test_start_planner_reuses_shared_server_without_local_model_file(monkeypatch, tmp_path):
    monkeypatch.setattr(control_server, "MANAGE_SHARED_MODELS", False)
    monkeypatch.setattr(control_server, "PLANNER_PROCESS", None)
    monkeypatch.setattr(control_server, "load_planner_config", lambda: {
        "model_path": str(tmp_path / "missing-planner.gguf"),
        "server_url": "http://127.0.0.1:8081",
    })
    monkeypatch.setattr(
        control_server,
        "port_listener",
        lambda port: {"OwningProcess": 123} if port == 8081 else None,
    )
    result = control_server.start_planner()
    assert result["ok"] is True
    assert "already running" in result["message"]



def test_terminate_wolfram_stops_only_the_owned_backend_session(monkeypatch):
    calls = []
    monkeypatch.setattr(control_server, "BACKEND_PROCESS", _fake_process(pid=8765))
    monkeypatch.setattr(
        control_server,
        "http_json_post",
        lambda url: (calls.append(url) or {"success": True}),
    )
    result = control_server.terminate_wolfram()
    assert result["ok"] is True
    assert calls == ["http://127.0.0.1:8000/api/wolfram/stop"]
