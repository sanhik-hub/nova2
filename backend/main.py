from contextlib import asynccontextmanager
import logging
from pathlib import Path
from threading import get_ident
from time import perf_counter
import time

from fastapi import FastAPI, Header, HTTPException

from backend.services.reasoning_service import ReasoningService
from backend.services.guided_attempt_service import GuidedAttemptService

from backend.schemas.computation import (
    ComputationRequest,
    ComputationResponse,
    PlotRequest,
)

from backend.services.computation_service import ComputationService
from backend.schemas.guided_attempt import GuidedAttemptRequest
from backend.services.auth_storage import (
    authenticate,
    create_session,
    create_user,
    delete_item,
    delete_session,
    get_settings,
    get_user_for_token,
    initialize_database,
    list_items,
    save_item,
    update_settings,
    update_item,
)


initialize_database()

computation = ComputationService()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Start one persistent Wolfram kernel after application import has
    completed, and terminate it cleanly when FastAPI shuts down.
    """

    engine = computation.router.providers["wolfram"].engine

    # WolframLanguageSession is lazy. start() performs the harmless
    # warm-up evaluation needed to launch the native kernel.
    engine.start()

    try:
        yield
    finally:
        engine.close()


app = FastAPI(
    title="Offline Computational Engine",
    description="Computational backend powered by multiple providers",
    version="0.3.0",
    lifespan=lifespan,
)


# Reuse the same provider router so Wolfram has one persistent session.
reasoning = ReasoningService(router=computation.router)
guided_attempts = GuidedAttemptService(router=computation.router)

# Temporary request-stage diagnostics for investigating slow /api/reason calls.
# The log records timings and metadata only; it does not write prompt contents.
_timing_logger = logging.getLogger("nova.reasoning_timing")
_timing_logger.setLevel(logging.INFO)
_timing_logger.propagate = False
if not _timing_logger.handlers:
    _timing_handler = logging.FileHandler(
        Path(__file__).resolve().parent / "reasoning_timing.log",
        encoding="utf-8",
    )
    _timing_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    _timing_logger.addHandler(_timing_handler)


def _install_reasoning_timing_diagnostics() -> None:
    """Log stage entry/exit so a stalled synchronous call can be located."""
    original_generate = reasoning.router.generate

    def timed_generate(*args, **kwargs):
        started = perf_counter()
        call_id = f"{get_ident()}-{time.time_ns()}"
        prompt = kwargs.get("prompt", "")
        system_prompt = kwargs.get("system_prompt", "")
        _timing_logger.info(
            "llm_generate START call=%s prompt_chars=%s system_prompt_chars=%s max_tokens=%s timeout=%s tool_call=%s json_mode=%s",
            call_id,
            len(prompt) if isinstance(prompt, str) else "unknown",
            len(system_prompt) if isinstance(system_prompt, str) else "unknown",
            kwargs.get("max_tokens", "default"),
            kwargs.get("request_timeout", "default"),
            bool(kwargs.get("tools")),
            bool(kwargs.get("response_format")),
        )
        try:
            result = original_generate(*args, **kwargs)
        except Exception:
            _timing_logger.exception(
                "llm_generate ERROR call=%s elapsed_ms=%.2f",
                call_id,
                (perf_counter() - started) * 1000,
            )
            raise
        _timing_logger.info(
            "llm_generate END call=%s elapsed_ms=%.2f success=%s result_chars=%s result_preview=%r error=%r",
            call_id,
            (perf_counter() - started) * 1000,
            result.get("success") if isinstance(result, dict) else "unknown",
            len(str(result.get("result") or "")) if isinstance(result, dict) else "unknown",
            str(result.get("result") or "")[:1400] if isinstance(result, dict) else repr(result)[:1400],
            result.get("error") if isinstance(result, dict) else None,
        )
        return result

    reasoning.router.generate = timed_generate

    original_execute_tool = reasoning._execute_tool

    def timed_execute_tool(operation, arguments):
        started = perf_counter()
        _timing_logger.info(
            "wolfram_execute START operation=%s arguments=%r thread=%s",
            operation,
            arguments,
            get_ident(),
        )
        try:
            result = original_execute_tool(operation, arguments)
        except Exception:
            _timing_logger.exception(
                "wolfram_execute ERROR operation=%s elapsed_ms=%.2f",
                operation,
                (perf_counter() - started) * 1000,
            )
            raise
        _timing_logger.info(
            "wolfram_execute END operation=%s elapsed_ms=%.2f success=%s error=%r result=%r",
            operation,
            (perf_counter() - started) * 1000,
            result.get("success") if isinstance(result, dict) else "unknown",
            result.get("error") if isinstance(result, dict) else None,
            str(result.get("result") or "")[:1000] if isinstance(result, dict) else repr(result)[:1000],
        )
        return result

    reasoning._execute_tool = timed_execute_tool

    original_pipeline = reasoning._solve_llm_wolfram

    def timed_pipeline(problem, explanation="detailed"):
        started = perf_counter()
        _timing_logger.info(
            "llm_wolfram_pipeline START explanation=%s question_chars=%s thread=%s",
            explanation,
            len(problem),
            get_ident(),
        )
        try:
            result = original_pipeline(problem, explanation=explanation)
        except Exception:
            _timing_logger.exception(
                "llm_wolfram_pipeline ERROR elapsed_ms=%.2f",
                (perf_counter() - started) * 1000,
            )
            raise
        _timing_logger.info(
            "llm_wolfram_pipeline END elapsed_ms=%.2f success=%s error=%r answer_preview=%r",
            (perf_counter() - started) * 1000,
            result.get("success") if isinstance(result, dict) else "unknown",
            result.get("error") if isinstance(result, dict) else None,
            str(result.get("answer") or "")[:500] if isinstance(result, dict) else repr(result)[:500],
        )
        return result

    reasoning._solve_llm_wolfram = timed_pipeline


_install_reasoning_timing_diagnostics()


def _current_user(authorization: str | None) -> dict:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Login required.")
    user = get_user_for_token(authorization[7:].strip())
    if user is None:
        raise HTTPException(status_code=401, detail="Your local session has expired. Please log in again.")
    return user


@app.post("/api/auth/register")
def register(payload: dict):
    try:
        user = create_user(str(payload.get("username") or ""), str(payload.get("password") or ""))
        token = create_session(user["id"])
        return {"success": True, "user": user, "token": token}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/auth/login")
def login(payload: dict):
    try:
        user = authenticate(str(payload.get("username") or ""), str(payload.get("password") or ""))
        token = create_session(user["id"])
        return {"success": True, "user": user, "token": token}
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@app.post("/api/auth/logout")
def logout(authorization: str | None = Header(default=None)):
    token = authorization[7:].strip() if authorization and authorization.lower().startswith("bearer ") else None
    delete_session(token)
    return {"success": True}


@app.get("/api/auth/me")
def me(authorization: str | None = Header(default=None)):
    return {"success": True, "user": _current_user(authorization)}


@app.get("/api/settings")
def settings(authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    return {"success": True, "settings": get_settings(user["id"])}


@app.put("/api/settings")
def update_user_settings(payload: dict, authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    return {"success": True, "settings": update_settings(
        user["id"],
        save_by_default=bool(payload.get("save_by_default", False)),
        save_followups_by_default=bool(payload.get("save_followups_by_default", False)),
    )}


@app.get("/api/storage")
def saved_items(authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    return {"success": True, "items": list_items(user["id"])}


@app.post("/api/storage")
def create_saved_item(payload: dict, authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    try:
        item = save_item(
            user["id"],
            kind=str(payload.get("kind") or ""),
            title=str(payload.get("title") or ""),
            question=str(payload.get("question") or ""),
            payload=payload.get("payload") if isinstance(payload.get("payload"), dict) else {},
        )
        return {"success": True, "item": item}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.put("/api/storage/{item_id}")
def update_saved_item(item_id: int, payload: dict, authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    try:
        item = update_item(
            user["id"],
            item_id,
            title=str(payload.get("title") or ""),
            question=str(payload.get("question") or ""),
            payload=payload.get("payload") if isinstance(payload.get("payload"), dict) else {},
        )
        return {"success": True, "item": item}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/follow-up")
def follow_up(payload: dict):
    original_question = str(payload.get("original_question") or "")
    original_answer = str(payload.get("original_answer") or "")
    follow_up = str(payload.get("follow_up") or "")
    history = payload.get("history") if isinstance(payload.get("history"), list) else []

    return reasoning.solve_follow_up(
        original_question=original_question,
        original_answer=original_answer,
        history=history,
        follow_up=follow_up,
    )


@app.delete("/api/storage/{item_id}")
def remove_saved_item(item_id: int, authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    return {"success": delete_item(user["id"], item_id)}


@app.post("/api/reason")
def reason(request: ComputationRequest):
    started = perf_counter()
    _timing_logger.info(
        "reason_request START explanation=%s question_chars=%s thread=%s",
        request.explanation,
        len(request.query or ""),
        get_ident(),
    )
    try:
        result = reasoning.solve(
            request.query,
            explanation=request.explanation,
        )
        _timing_logger.info(
            "reason_request END elapsed_ms=%.2f success=%s engine=%s error=%r answer_preview=%r",
            (perf_counter() - started) * 1000,
            result.get("success") if isinstance(result, dict) else "unknown",
            result.get("engine") if isinstance(result, dict) else "unknown",
            result.get("error") if isinstance(result, dict) else None,
            str(result.get("answer") or "")[:500] if isinstance(result, dict) else repr(result)[:500],
        )
        return result
    except Exception as exc:
        _timing_logger.exception(
            "reason_request ERROR elapsed_ms=%.2f",
            (perf_counter() - started) * 1000,
        )
        return {
            "success": False,
            "answer": None,
            "error": f"Reasoning request failed: {type(exc).__name__}: {exc}",
            "engine": None,
            "tool_calls": [],
            "cached": False,
        }


@app.post("/api/guided/start")
def start_guided_attempt(request: GuidedAttemptRequest):
    started = perf_counter()
    try:
        result = guided_attempts.generate(
            query=request.query,
            include_hints=request.include_hints,
            step_count=request.step_count,
        )
    except Exception as exc:
        result = {
            "success": False,
            "query": request.query,
            "steps": [],
            "include_hints": request.include_hints,
            "include_full_solution": request.include_full_solution,
            "full_solution": None,
            "error": f"Guided attempt failed: {type(exc).__name__}: {exc}",
        }

    elapsed_ms = round((perf_counter() - started) * 1000, 2)
    result["elapsed_ms"] = elapsed_ms
    result["mcq_elapsed_ms"] = result.get("mcq_generation_elapsed_ms", elapsed_ms)
    result["hint_elapsed_ms"] = (
        result.get("hint_elapsed_ms", result["mcq_elapsed_ms"])
        if request.include_hints
        else 0.0
    )
    result["include_full_solution"] = False
    result["full_solution"] = None
    return result


@app.post("/api/guided/solution")
def generate_guided_solution(payload: dict):
    started = perf_counter()
    query = str(payload.get("query") or "").strip()
    if not query:
        return {"success": False, "error": "Question is empty."}

    try:
        result = guided_attempts.generate_solution(
            query=query,
            steps=payload.get("steps") or [],
            verified_solution=str(payload.get("verified_solution") or "").strip() or None,
        )
    except Exception as exc:
        result = {
            "success": False,
            "error": f"Full solution generation failed: {type(exc).__name__}: {exc}",
        }
    result["elapsed_ms"] = round((perf_counter() - started) * 1000, 2)
    result["final_solution_elapsed_ms"] = result["elapsed_ms"]
    return result


@app.get("/")
def root():
    return {
        "name": "Offline Computational Engine",
        "status": "online",
    }


@app.get("/api/health")
def health():
    engine = computation.router.providers["wolfram"].engine
    session = engine.status()

    return {
        "status": "healthy",
        "wolfram": session["session_started"],
        "wolfram_session": session,
        "executable": engine.executable,
    }


@app.post("/api/wolfram/start")
def start_wolfram():
    engine = computation.router.providers["wolfram"].engine

    result = engine.start()

    if not result:
        return {
            "success": False,
            "message": (
                engine.status().get("startup_error")
                or "Wolfram session could not be started."
            ),
            "wolfram_session": engine.status(),
        }

    verification = engine.calculate("2 + 2")

    if not verification.success:
        return {
            "success": False,
            "message": verification.error or "Wolfram session verification failed.",
            "wolfram_session": engine.status(),
        }

    return {
        "success": True,
        "message": "Wolfram session started and verified.",
        "wolfram_session": engine.status(),
        "test_result": verification.result,
    }


@app.post("/api/wolfram/restart")
def restart_wolfram():
    engine = computation.router.providers["wolfram"].engine
    success = engine.restart()

    return {
        "success": success,
        "message": (
            "Wolfram persistent session restarted."
            if success
            else "Wolfram persistent session could not be restarted."
        ),
        "wolfram_session": engine.status(),
    }


@app.post("/api/wolfram/stop")
def stop_wolfram():
    """Stop only this backend's persistent Wolfram session."""
    engine = computation.router.providers["wolfram"].engine
    engine.close()
    session = engine.status()
    return {
        "success": not session.get("session_started", False),
        "message": "Wolfram persistent session stopped.",
        "wolfram_session": session,
    }


@app.post(
    "/api/plot",
    response_model=ComputationResponse,
)
def plot(request: PlotRequest):
    return computation.plot(
        expression=request.expression,
        variable=request.variable,
        xmin=request.xmin,
        xmax=request.xmax,
        provider=request.provider,
    )
