"""
Celery application instance. Broker and result backend are both Redis
(see config.py) — this means Redis serves three roles in this system:
cache (app.core.cache), Celery broker, and Celery result backend, each on a
separate logical DB index (0/1/2) so they don't collide.

CELERY_TASK_ALWAYS_EAGER (settings): when True, tasks run synchronously
in-process instead of being sent to a broker — this is what lets the test
suite exercise task logic without a running Redis/worker (see
tests/conftest.py). It must be False in any real deployment.

Retry policy: tasks that talk to external-ish resources (LLM/Ollama HTTP
calls, disk I/O under contention) use `autoretry_for` with exponential
backoff, since transient failures (Ollama still warming up, a momentary
FAISS lock) shouldn't fail a whole document ingestion permanently.
"""
from celery import Celery
from celery.signals import task_failure, task_success

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

celery_app = Celery(
    "medassist",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "app.tasks.ingestion_tasks",
        "app.tasks.evaluation_tasks",
    ],
)

celery_app.conf.update(
    task_always_eager=settings.CELERY_TASK_ALWAYS_EAGER,
    task_eager_propagates=True,  # in eager/test mode, exceptions surface immediately instead of being swallowed

    # Reliability: a task is only removed from the queue after it completes
    # (not merely after it starts) — if a worker crashes mid-task, another
    # worker picks it up instead of the work silently disappearing.
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # don't let one worker hoard multiple long-running ingestion tasks

    task_track_started=True,  # lets the status endpoint report "started" distinctly from "queued"
    result_expires=86400,     # task results (for the status endpoint) kept 24h then cleaned up

    task_routes={
        "app.tasks.ingestion_tasks.*": {"queue": "ingestion"},
        "app.tasks.evaluation_tasks.*": {"queue": "evaluation"},
    },

    task_default_retry_delay=10,  # seconds
    task_time_limit=1800,   # hard kill after 30 min (an evaluation run over a large benchmark)
    task_soft_time_limit=1500,  # allow cleanup before the hard kill
)


@task_failure.connect
def _on_task_failure(sender=None, task_id=None, exception=None, **kwargs):
    log.error("celery_task_failed", task_name=sender.name if sender else None, task_id=task_id, error=str(exception))


@task_success.connect
def _on_task_success(sender=None, result=None, **kwargs):
    log.info("celery_task_succeeded", task_name=sender.name if sender else None)
