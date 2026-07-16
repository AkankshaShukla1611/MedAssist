"""
Domain-specific exceptions. Routes/services raise these instead of generic
HTTPException so the actual error handling policy (status code, log level,
whether details are exposed to the client) lives in ONE place:
app/core/exception_handlers.py — not scattered across every route.
"""


class MedAssistError(Exception):
    """Base class for all application errors."""

    status_code: int = 500
    public_message: str = "An unexpected error occurred."

    def __init__(self, message: str | None = None, **context):
        self.message = message or self.public_message
        self.context = context
        super().__init__(self.message)


class DocumentProcessingError(MedAssistError):
    status_code = 422
    public_message = "The document could not be processed."


class RetrievalError(MedAssistError):
    status_code = 503
    public_message = "The retrieval system is temporarily unavailable."


class LLMUnavailableError(MedAssistError):
    status_code = 503
    public_message = "The language model backend is temporarily unavailable."


class InvalidFilterError(MedAssistError):
    status_code = 400
    public_message = "Invalid search filter provided."


class SessionNotFoundError(MedAssistError):
    status_code = 404
    public_message = "Conversation session not found."
