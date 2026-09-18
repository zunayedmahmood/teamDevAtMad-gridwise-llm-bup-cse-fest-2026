from fastapi import FastAPI

from app.api.error_handlers import register_exception_handlers
from app.api.routes import router
from app.observability.logging import configure_logging, register_request_logging


def create_app() -> FastAPI:
    configure_logging()
    application = FastAPI(title="GridWise LLM", version="1.0.0")
    register_exception_handlers(application)
    register_request_logging(application)
    application.include_router(router)
    return application


app = create_app()
