from control import server as control_server


def _process(pid, parent, name, command):
    return {
        "ProcessId": pid,
        "ParentProcessId": parent,
        "Name": name,
        "CommandLine": command,
    }


def test_process_snapshot_uses_command_lines_when_listener_pid_is_zero(monkeypatch):
    monkeypatch.setattr(control_server, "get_processes", lambda: [
        _process(
            1144, 16156, "python.exe",
            "python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload",
        ),
        _process(
            604, 1144, "python.exe",
            "python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload",
        ),
        _process(
            2372, 25248, "node.exe",
            "node vite.js --host 127.0.0.1 --port 5173",
        ),
        _process(
            13664, 1100, "llama-server.exe",
            "llama-server.exe -m local.gguf --host 127.0.0.1 --port 8080",
        ),
        _process(
            24128, 1100, "llama-server.exe",
            "llama-server.exe -m planner.gguf --host 127.0.0.1 --port 8081",
        ),
    ])
    monkeypatch.setattr(
        control_server,
        "port_listener",
        lambda port: {"OwningProcess": 0, "LocalPort": port},
    )

    snapshot = control_server.process_snapshot()

    assert [item["pid"] for item in snapshot["backend"]] == [1144, 604]
    assert [item["pid"] for item in snapshot["frontend"]] == [2372]
    assert [item["pid"] for item in snapshot["llm"]] == [13664]
    assert [item["pid"] for item in snapshot["planner"]] == [24128]
    assert (control_server.primary_process(snapshot["backend"]) or {})["pid"] == 1144


def test_health_status_uses_http_health_and_real_process_ids(monkeypatch):
    processes = {
        "backend": [{"pid": 1144, "parent_pid": 16156, "name": "python.exe", "command": ""}],
        "frontend": [{"pid": 2372, "parent_pid": 25248, "name": "node.exe", "command": ""}],
        "llm": [{"pid": 13664, "parent_pid": 1100, "name": "llama-server.exe", "command": ""}],
        "planner": [{"pid": 24128, "parent_pid": 1100, "name": "llama-server.exe", "command": ""}],
        "wolfram": [],
    }
    monkeypatch.setattr(control_server, "load_llm_config", lambda: {
        "server_url": "http://127.0.0.1:8080",
        "model_name": "local-model",
        "model_path": "local.gguf",
    })
    monkeypatch.setattr(control_server, "load_planner_config", lambda: {
        "server_url": "http://127.0.0.1:8081",
        "model_name": "planner-model",
        "model_path": "planner.gguf",
    })
    monkeypatch.setattr(control_server, "process_snapshot", lambda: processes)
    monkeypatch.setattr(control_server, "port_listener", lambda port: None)
    monkeypatch.setattr(control_server, "http_check", lambda url: {
        "ok": True,
        "status": 200,
    })
    monkeypatch.setattr(control_server, "http_json_check", lambda url: {
        "ok": True,
        "wolfram": False,
        "wolfram_session": {},
    })

    status = control_server.health_status()

    assert status["backend"]["process"] is True
    assert status["backend"]["pid"] == 1144
    assert status["frontend"]["process"] is True
    assert status["frontend"]["pid"] == 2372
    assert status["llm"]["process"] is True
    assert status["llm"]["pid"] == 13664
    assert status["planner"]["process"] is True
    assert status["planner"]["pid"] == 24128
