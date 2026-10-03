from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.services.reasoning_service import ReasoningService
from backend.services.guided_attempt_service import GuidedAttemptService

from backend.schemas.computation import (
    ComputationRequest,
    ComputationResponse,
    PlotRequest,
)

from backend.services.computation_service import ComputationService
from backend.schemas.guided_attempt import GuidedAttemptRequest


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


@app.post("/api/reason")
def reason(request: ComputationRequest):
    return reasoning.solve(
        request.query,
        explanation=request.explanation,
    )


@app.post("/api/guided/start")
def start_guided_attempt(request: GuidedAttemptRequest):
    return guided_attempts.generate(
        query=request.query,
        include_hints=request.include_hints,
        step_count=request.step_count,
    )


@app.post("/api/guided/solution")
def generate_guided_solution(payload: dict):
    return guided_attempts.generate_solution(
        query=str(payload.get("query") or ""),
        steps=payload.get("steps") or [],
    )


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
