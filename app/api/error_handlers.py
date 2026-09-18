import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import settings
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
        logger.info(
            "request_id=%s exception_category=%s detail=validation_errors:%d path=%s",
            getattr(request.state, "request_id", "-"),
            type(exc).__name__,
            len(exc.errors()),
            request.url.path,
        )
        return JSONResponse(status_code=400, content={"detail": "Invalid request body."})

    @app.exception_handler(SemanticInputError)
    async def semantic_input_error_handler(request: Request, exc: SemanticInputError) -> JSONResponse:
        logger.info(
            "request_id=%s exception_category=%s detail=%s",
            getattr(request.state, "request_id", "-"),
            type(exc).__name__,
            str(exc),
        )
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(OptimizationInfeasibleError)
    async def optimization_infeasible_handler(request: Request, exc: OptimizationInfeasibleError) -> JSONResponse:
        logger.info(
            "request_id=%s exception_category=%s detail=%s",
            getattr(request.state, "request_id", "-"),
            type(exc).__name__,
            str(exc),
        )
        return JSONResponse(
            status_code=422,
            content={"detail": "The supplied energy constraints are infeasible."},
        )

    @app.exception_handler(LLMProviderError)
    async def llm_provider_error_handler(request: Request, exc: LLMProviderError) -> JSONResponse:
        logger.warning(
            "request_id=%s exception_category=%s detail=%s",
            getattr(request.state, "request_id", "-"),
            type(exc).__name__,
            str(exc),
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Directive interpretation is temporarily unavailable."},
        )

    @app.exception_handler(LLMInterpretationError)
    async def llm_interpretation_error_handler(request: Request, exc: LLMInterpretationError) -> JSONResponse:
        logger.warning(
            "request_id=%s exception_category=%s detail=%s",
            getattr(request.state, "request_id", "-"),
            type(exc).__name__,
            str(exc),
        )
        return JSONResponse(status_code=500, content={"detail": "Unable to interpret operator directives."})

    @app.exception_handler(DirectiveValidationError)
    async def directive_validation_error_handler(request: Request, exc: DirectiveValidationError) -> JSONResponse:
        logger.error(
            "request_id=%s exception_category=%s detail=%s",
            getattr(request.state, "request_id", "-"),
            type(exc).__name__,
            str(exc),
        )
        return JSONResponse(status_code=500, content={"detail": "Unable to interpret operator directives."})

    @app.exception_handler(InternalPlanValidationError)
    async def internal_plan_validation_error_handler(request: Request, exc: InternalPlanValidationError) -> JSONResponse:
        logger.error(
            "request_id=%s exception_category=%s detail=%s",
            getattr(request.state, "request_id", "-"),
            type(exc).__name__,
            str(exc),
        )
        return JSONResponse(status_code=500, content={"detail": "Internal schedule validation failed."})

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "-")
        if settings.debug_tracebacks:
            logger.exception(
                "request_id=%s exception_category=%s",
                request_id,
                type(exc).__name__,
            )
        else:
            logger.error(
                "request_id=%s method=%s path=%s exception_category=%s detail=unhandled_error",
                request_id,
                request.method,
                request.url.path,
                type(exc).__name__,
            )
        return JSONResponse(status_code=500, content={"detail": "Internal service error."})
