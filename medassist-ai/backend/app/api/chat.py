from fastapi import APIRouter, Depends, Request, HTTPException
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database import models
from app.core.security import get_current_user
from app.core.limiter import limiter
from app.schemas.chat import ChatRequest, ChatResponse, HistoryItem
from app.services.rag_service import answer_question

router = APIRouter(tags=["Chat"])


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
async def chat(
    request: Request,
    payload: ChatRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    result = await answer_question(
        db, current_user, payload.question,
        session_id=payload.session_id, category=payload.category,
    )
    return result


@router.get("/history", response_model=list[HistoryItem])
async def history(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    conversations = (
        db.query(models.Conversation)
        .filter(models.Conversation.user_id == current_user.id)
        .order_by(models.Conversation.created_at.desc())
        .limit(100)
        .all()
    )
    return [
        HistoryItem(
            id=c.id, session_id=c.session_id, question=c.question, answer=c.answer,
            confidence=c.confidence, created_at=c.created_at.isoformat(),
        )
        for c in conversations
    ]


@router.delete("/history", status_code=204)
async def clear_history(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    # Users can only ever delete their OWN history — never scoped by anything
    # client-supplied, to avoid IDOR-style deletion of someone else's data.
    db.query(models.Conversation).filter(models.Conversation.user_id == current_user.id).delete()
    db.commit()


@router.delete("/history/{conversation_id}", status_code=204)
async def delete_one_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    conversation = (
        db.query(models.Conversation)
        .filter(models.Conversation.id == conversation_id, models.Conversation.user_id == current_user.id)
        .first()
    )
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    db.delete(conversation)
    db.commit()
