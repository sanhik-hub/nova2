from pathlib import Path
from types import SimpleNamespace
import sys

from control import server as control_server


def test_backend_starts_with_source_reload(monkeypatch, tmp_path):
    backend_dir = tmp_path / "backend"
    backend_dir.mkdir()
    (backend_dir / "main.py").write_text("app = None\n", encoding="utf-8")

    launched = {}

    monkeypatch.setattr(control_server, "NOVA_ROOT", tmp_path)
    monkeypatch.setattr(control_server, "BACKEND_PROCESS", None)
    monkeypatch.setattr(control_server, "PYTHON", Path(sys.executable))
    monkeypatch.setattr(control_server, "port_listener", lambda port: None)

    def fake_start_process(args, cwd, **kwargs):
        launched["args"] = args
        launched["cwd"] = cwd
        launched["kwargs"] = kwargs
        return SimpleNamespace(pid=12345, poll=lambda: None)

    monkeypatch.setattr(control_server, "start_process", fake_start_process)
    monkeypatch.setattr(control_server, "http_check", lambda url: {"ok": True})

    result = control_server.start_backend()

    assert result["ok"] is True
    assert "--reload" in launched["args"]
    reload_dir_index = launched["args"].index("--reload-dir")
    assert Path(launched["args"][reload_dir_index + 1]) == backend_dir
    assert launched["cwd"] == tmp_path
    assert launched["kwargs"]["log_name"] == "backend.log"


def test_service_process_output_is_redirected_to_persistent_log(monkeypatch, tmp_path):
    captured = {}

    def fake_popen(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return SimpleNamespace(pid=4321)

    monkeypatch.setattr(control_server, "NOVA_ROOT", tmp_path)
    monkeypatch.setattr(control_server.subprocess, "Popen", fake_popen)

    control_server.start_process(["python", "-m", "uvicorn"], tmp_path, log_name="backend.log")

    log_path = tmp_path / "logs" / "backend.log"
    assert log_path.exists()
    assert "Starting: python -m uvicorn" in log_path.read_text(encoding="utf-8")
    assert captured["kwargs"]["stdout"].name == str(log_path)
    assert captured["kwargs"]["stderr"] == control_server.subprocess.STDOUT


def test_open_service_log_uses_allowlisted_component_and_new_console(monkeypatch, tmp_path):
    captured = {}

    def fake_popen(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return SimpleNamespace(pid=9876)

    monkeypatch.setattr(control_server, "NOVA_ROOT", tmp_path)
    monkeypatch.setattr(control_server.subprocess, "Popen", fake_popen)

    result = control_server.open_service_log("wolfram")

    assert result["ok"] is True
    assert (tmp_path / "backend" / "reasoning_timing.log").exists()
    assert captured["args"][0] == "powershell.exe"
    assert "-Wait" in captured["args"][-1]
    assert captured["kwargs"]["creationflags"] == getattr(
        control_server.subprocess, "CREATE_NEW_CONSOLE", 0
    )

    try:
        control_server.open_service_log("../unexpected")
    except ValueError:
        pass
    else:
        raise AssertionError("Unlisted log components must be rejected")


def test_all_five_dashboard_cards_have_log_buttons():
    dashboard_path = Path(__file__).resolve().parents[1] / "control" / "index.html"
    dashboard = dashboard_path.read_text(encoding="utf-8-sig")
    for component in ("backend", "frontend", "llm", "planner", "wolfram"):
        assert f"openServiceLog('{component}')" in dashboard
    server_source = Path(control_server.__file__).read_text(encoding="utf-8")
    assert 'parsed.path.startswith("/api/open-log/")' in server_source


def test_control_center_starts_services_and_cleans_up_on_exit(monkeypatch):
    calls = []

    class FakeServer:
        def __init__(self, address, handler):
            calls.append(("bind", address))
            self.daemon_threads = False

        def serve_forever(self, poll_interval=0.5):
            calls.append(("serve", poll_interval))
            raise KeyboardInterrupt

        def server_close(self):
            calls.append(("close",))

    monkeypatch.setattr(control_server, "ThreadingHTTPServer", FakeServer)
    monkeypatch.setattr(control_server, "start_all_background", lambda **kwargs: calls.append(("start_all", kwargs)))
    monkeypatch.setattr(control_server, "terminate_all", lambda: calls.append(("terminate_all",)))

    control_server.main()

    assert any(call[0] == "start_all" and call[1].get("restart_existing_backend") is True for call in calls)
    assert ("close",) in calls
    assert ("terminate_all",) in calls
    start_call_index = next(i for i, call in enumerate(calls) if call[0] == "start_all")
    assert start_call_index < calls.index(("serve", 0.5))
    assert calls.index(("close",)) < calls.index(("terminate_all",))


def test_control_center_exposes_explicit_shutdown_route_and_button():
    server_source = Path(control_server.__file__).read_text(encoding="utf-8")
    dashboard_path = Path(control_server.NOVA_ROOT) / "control" / "index.html"

    # Read the repository dashboard directly when running from a checkout.
    if not dashboard_path.exists():
        dashboard_path = Path(__file__).resolve().parents[1] / "control" / "index.html"
    dashboard_source = dashboard_path.read_text(encoding="utf-8")

    assert 'parsed.path == "/api/shutdown"' in server_source
    assert "def shutdown_control_center(" in server_source
    assert 'onclick="closeControlCenter()"' in dashboard_source
    assert 'fetch("/api/shutdown"' in dashboard_source
