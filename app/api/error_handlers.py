import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.errors import (
    DirectiveValidationError,
    InternalPlanValidationError,
    LLMInterpretationError,
    LLMProviderError,
    OptimizationInfeasibleError,
    SemanticInputError,
)

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        logger.info("request_validation_error path=%s", request.url.path)
        return JSONResponse(status_code=400, content={"detail": "Invalid request body."})

    @app.exception_handler(SemanticInputError)
    async def semantic_input_error_handler(request: Request, exc: SemanticInputError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(OptimizationInfeasibleError)
    async def optimization_infeasible_handler(request: Request, exc: OptimizationInfeasibleError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"detail": "The supplied energy constraints are infeasible."},
        )

    @app.exception_handler(LLMProviderError)
    async def llm_provider_error_handler(request: Request, exc: LLMProviderError) -> JSONResponse:
        logger.warning("llm_provider_error request_id=%s type=%s", getattr(request.state, "request_id", "-"), type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={"detail": "Directive interpretation is temporarily unavailable."},
        )

    @app.exception_handler(LLMInterpretationError)
    async def llm_interpretation_error_handler(request: Request, exc: LLMInterpretationError) -> JSONResponse:
        logger.warning("llm_interpretation_error request_id=%s type=%s", getattr(request.state, "request_id", "-"), type(exc).__name__)
        return JSONResponse(status_code=500, content={"detail": "Unable to interpret operator directives."})

    @app.exception_handler(DirectiveValidationError)
    async def directive_validation_error_handler(request: Request, exc: DirectiveValidationError) -> JSONResponse:
        logger.error("unrecovered_directive_validation_error request_id=%s", getattr(request.state, "request_id", "-"))
        return JSONResponse(status_code=500, content={"detail": "Unable to interpret operator directives."})

    @app.exception_handler(InternalPlanValidationError)
    async def internal_plan_validation_error_handler(request: Request, exc: InternalPlanValidationError) -> JSONResponse:
        logger.error("internal_plan_validation_error request_id=%s", getattr(request.state, "request_id", "-"))
        return JSONResponse(status_code=500, content={"detail": "Internal optimization validation failed."})

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("unexpected_error request_id=%s type=%s", getattr(request.state, "request_id", "-"), type(exc).__name__)
        return JSONResponse(status_code=500, content={"detail": "Internal service error."})
