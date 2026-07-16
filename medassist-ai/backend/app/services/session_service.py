"""
Session-aware conversation memory: groups conversation turns by `session_id`
so follow-up questions ("what about in pregnant patients?") can be answered
using the context of prior turns in the same session, without requiring the
client to resend the whole conversation each time.

`session_id` is client-generated (e.g. a UUID created once per chat session
in the frontend) and passed on each /chat request; if omitted, the request
is treated as a standalone question (fully backward compatible with the
original API).
"""
from sqlalchemy.orm import Session

from app.database import models
from app.core.config import settings


def get_recent_turns(db: Session, user_id: int, session_id: str | None) -> list[models.Conversation]:
    if not session_id:
        return []
    return (
        db.query(models.Conversation)
        .filter(models.Conversation.user_id == user_id, models.Conversation.session_id == session_id)
        .order_by(models.Conversation.created_at.desc())
        .limit(settings.MAX_SESSION_HISTORY_TURNS)
        .all()[::-1]  # chronological order for the prompt
    )


def format_history_for_prompt(turns: list[models.Conversation]) -> str:
    if not turns:
        return ""
    lines = []
    for t in turns:
        lines.append(f"Previous question: {t.question}\nPrevious answer summary: {t.answer[:300]}")
    return "\n\n".join(lines)
