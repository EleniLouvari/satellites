"""FastAPI application factory without startup database side effects."""

from importlib import import_module

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from parcel_app.auth import routes as auth
from parcel_app.jobs.registry import router_modules


def create_app() -> FastAPI:
    """Assemble the private-network API and expose a secret-free health check."""
    application = FastAPI(title="Parcel Studio")
    application.include_router(auth.router, prefix="/api")
    # Data sources and pipelines are listed once, in parcel_app.jobs.registry.
    for name in router_modules():
        application.include_router(import_module(name).router, prefix="/api")

    @application.get("/api/health")
    def health():
        """Report process health without database details or secrets."""
        return {"status": "ok"}

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        """Avoid echoing input data, which may contain passwords."""
        return JSONResponse(status_code=422, content={"detail": "Error: Invalid request fields"})

    @application.exception_handler(Exception)
    async def internal_error(request: Request, error: Exception):
        """Keep database and third-party exception text out of browser responses."""
        return JSONResponse(status_code=500, content={"detail": "Error: Request could not be completed"})

    return application


app = create_app()
