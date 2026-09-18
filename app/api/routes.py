from fastapi import APIRouter, Depends, Request

from app.dependencies import get_interpreter
from app.interpreter.client import DirectiveInterpreter
from app.models.request import EnergyRequest
from app.models.response import EnergyResponse
from app.services.optimization_service import optimize_energy as run_optimization

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/optimize-energy", response_model=EnergyResponse)
async def optimize_energy(
    request: Request,
    payload: EnergyRequest,
    interpreter: DirectiveInterpreter = Depends(get_interpreter),
) -> EnergyResponse:
    return await run_optimization(
        payload,
        interpreter,
        request_id=getattr(request.state, "request_id", "-"),
    )
