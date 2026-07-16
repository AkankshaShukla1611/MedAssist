from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.exceptions import MedAssistError
from app.core.logging import get_logger

log = get_logger(__name__)


async def medassist_error_handler(request: Request, exc: MedAssistError) -> JSONResponse:
    log.warning(
        "handled_application_error",
        error_type=type(exc).__name__,
        message=exc.message,
        path=request.url.path,
        **exc.context,
    )
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Never leak internals (stack traces, exception text) to the client —
    # log the real detail server-side, return a generic message to the caller.
    log.error(
        "unhandled_exception",
        error_type=type(exc).__name__,
        path=request.url.path,
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected internal error occurred. It has been logged."},
    )
