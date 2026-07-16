"""
Structured logging, configured once at startup.
Every log line is JSON in production (machine-parseable for log aggregation:
ELK/Datadog/CloudWatch), and pretty-printed in development.

Usage anywhere in the codebase:
    from app.core.logging import get_logger
    log = get_logger(__name__)
    log.info("chunk_retrieved", chunk_id=123, score=0.87)
"""
import logging
import sys

import structlog

from app.core.config import settings


def configure_logging() -> None:
    log_level = logging.DEBUG if settings.ENV != "production" else logging.INFO

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.ENV == "production":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = __name__):
    return structlog.get_logger(name)
