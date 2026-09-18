import logging
import time
from uuid import uuid4

from fastapi import FastAPI, Request

from app.config import settings


logger = logging.getLogger("gridwise.requests")


def configure_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def register_request_logging(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        request_id = uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            status_code = 500
            raise
        finally:
            duration_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
                request_id,
                request.method,
                request.url.path,
                status_code,
                duration_ms,
            )
        response.headers["X-Request-ID"] = request_id
        return response
