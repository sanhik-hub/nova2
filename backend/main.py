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


app = FastAPI(
    title="Offline Computational Engine",
    description="Computational backend powered by multiple providers",
    version="0.3.0",
)

computation = ComputationService()
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

    # WolframLanguageSession starts its native kernel lazily on the
    # first evaluation. Warm it explicitly so Control Center can report
    # the persistent kernel as running immediately after startup.
    result = engine.calculate("2 + 2")

    if not result.success:
        return {
            "success": False,
            "message": result.error or "Wolfram session could not be started.",
            "wolfram_session": engine.status(),
        }

    return {
        "success": True,
        "message": "Wolfram session started and verified.",
        "wolfram_session": engine.status(),
        "test_result": result.result,
    }


@app.post("/api/wolfram/restart")
def restart_wolfram():
    engine = computation.router.providers["wolfram"].engine
    engine.restart()

    return {
        "success": True,
        "message": "Wolfram session reset. The kernel will start on the next evaluation.",
        "wolfram_session": engine.status(),
    }


@app.post(
    "/api/compute",
    response_model=ComputationResponse,
)
def compute(request: ComputationRequest):

    return computation.compute(
        query=request.query,
        operation=request.operation,
        provider=request.provider,
    )


@app.post(
    "/api/solve",
    response_model=ComputationResponse,
)
def solve(request: ExpressionRequest):

    return computation.solve(
        expression=request.expression,
        variable=request.variable,
        provider=request.provider,
    )


@app.post(
    "/api/simplify",
    response_model=ComputationResponse,
)
def simplify(request: ExpressionRequest):

    return computation.simplify(
        expression=request.expression,
        provider=request.provider,
    )


@app.post(
    "/api/factor",
    response_model=ComputationResponse,
)
def factor(request: ExpressionRequest):

    return computation.factor(
        expression=request.expression,
        provider=request.provider,
    )


@app.post(
    "/api/expand",
    response_model=ComputationResponse,
)
def expand(request: ExpressionRequest):

    return computation.expand(
        expression=request.expression,
        provider=request.provider,
    )


@app.post(
    "/api/differentiate",
    response_model=ComputationResponse,
)
def differentiate(request: ExpressionRequest):

    return computation.differentiate(
        expression=request.expression,
        variable=request.variable,
        provider=request.provider,
    )


@app.post(
    "/api/integrate",
    response_model=ComputationResponse,
)
def integrate(request: ExpressionRequest):

    return computation.integrate(
        expression=request.expression,
        variable=request.variable,
        provider=request.provider,
    )


@app.post(
    "/api/limit",
    response_model=ComputationResponse,
)
def limit(request: LimitRequest):

    return computation.limit(
        expression=request.expression,
        variable=request.variable,
        value=request.value,
        provider=request.provider,
    )
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
