"""
Append-only audit trail for security-relevant and admin actions: uploads,
deletes, authentication events, admin dashboard access, and evaluation runs.

Design choices:
- Never raises: an audit-logging failure must never break the request it's
  logging (same "fail open" philosophy as app.core.cache). If the DB write
  fails, we log the failure to structlog and move on.
- Takes the raw Request so IP address can be captured consistently in one
  place rather than re-deriving it in every route.
- `action` uses a "resource.verb" convention (e.g. "document.upload",
  "auth.login_failed") so it's greppable/filterable in the audit API.
"""
import json

from fastapi import Request
from sqlalchemy.orm import Session

from app.database import models
from app.core.logging import get_logger
from app.core.ip_utils import get_client_ip

log = get_logger(__name__)


def record_audit_event(
    db: Session,
    action: str,
    success: bool,
    request: Request | None = None,
    user_id: int | None = None,
    resource_type: str | None = None,
    resource_id: str | int | None = None,
    details: dict | None = None,
) -> None:
    try:
        entry = models.AuditLog(
            user_id=user_id,
            ip_address=get_client_ip(request) if request is not None else None,
            endpoint=request.url.path if request else "internal",
            action=action,
            resource_type=resource_type,
            resource_id=str(resource_id) if resource_id is not None else None,
            success=success,
            details=json.dumps(details) if details else None,
        )
        db.add(entry)
        db.commit()
    except Exception as e:
        log.error("audit_log_write_failed", action=action, error=str(e))
        db.rollback()


def query_audit_logs(
    db: Session,
    user_id: int | None = None,
    action: str | None = None,
    success: bool | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[models.AuditLog]:
    query = db.query(models.AuditLog)
    if user_id is not None:
        query = query.filter(models.AuditLog.user_id == user_id)
    if action is not None:
        query = query.filter(models.AuditLog.action == action)
    if success is not None:
        query = query.filter(models.AuditLog.success == success)
    return query.order_by(models.AuditLog.created_at.desc()).offset(offset).limit(min(limit, 500)).all()
