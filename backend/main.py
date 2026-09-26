from fastapi import FastAPI
from backend.services.reasoning_service import ReasoningService

from backend.schemas.computation import (
    ComputationRequest,
    ComputationResponse,
    ExpressionRequest,
    LimitRequest,
    PlotRequest,
)

from backend.services.computation_service import ComputationService


app = FastAPI(
    title="Offline Computational Engine",
    description="Computational backend powered by multiple providers",
    version="0.3.0",
)

computation = ComputationService()
reasoning = ReasoningService()

@app.post("/api/reason")
def reason(request: ComputationRequest):
    return reasoning.solve(request.query)

@app.get("/")
def root():
    return {
        "name": "Offline Computational Engine",
        "status": "online",
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "wolfram": True,
        "executable": computation.router.providers[
            "wolfram"
        ].engine.executable,
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