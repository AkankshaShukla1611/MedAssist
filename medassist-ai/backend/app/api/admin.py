from sqlalchemy import func
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database import models
from app.core.security import require_roles

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get("/dashboard")
async def dashboard(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_roles("admin")),
):
    total_users = db.query(func.count(models.User.id)).scalar()
    total_documents = db.query(func.count(models.Document.id)).scalar()
    total_questions = db.query(func.count(models.Conversation.id)).scalar()

    embedding_status_counts = dict(
        db.query(models.Document.embedding_status, func.count(models.Document.id))
        .group_by(models.Document.embedding_status)
        .all()
    )

    return {
        "total_users": total_users,
        "total_documents": total_documents,
        "total_questions": total_questions,
        "embedding_status": embedding_status_counts,
    }


@router.get("/users")
async def list_users(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_roles("admin")),
):
    users = db.query(models.User).all()
    return [
        {"id": u.id, "name": u.name, "email": u.email, "role": u.role.value, "is_active": u.is_active}
        for u in users
    ]


@router.get("/analytics")
async def analytics(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_roles("admin")),
):
    # Most-asked questions is naive text grouping for the skeleton — swap for
    # semantic clustering later if you want real topic analysis.
    top_questions = (
        db.query(models.Conversation.question, func.count(models.Conversation.id).label("count"))
        .group_by(models.Conversation.question)
        .order_by(func.count(models.Conversation.id).desc())
        .limit(10)
        .all()
    )
    return {"most_searched_topics": [{"question": q, "count": c} for q, c in top_questions]}
