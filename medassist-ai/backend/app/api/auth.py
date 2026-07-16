from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import HTTPException
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database import models
from app.schemas.auth import RegisterRequest, LoginRequest, TokenResponse, ProfileResponse
from app.services.auth_service import register_user, authenticate_user
from app.services.audit_service import record_audit_event
from app.core.security import get_current_user
from app.core.limiter import limiter
from app.core.config import settings

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=ProfileResponse, status_code=201)
@limiter.limit(settings.RATE_LIMIT_AUTH)
async def register(request: Request, payload: RegisterRequest, db: Session = Depends(get_db)):
    try:
        user = register_user(db, payload)
    except HTTPException as e:
        record_audit_event(
            db, action="auth.register", success=False, request=request,
            details={"email": payload.email, "reason": e.detail},
        )
        raise
    record_audit_event(
        db, action="auth.register", success=True, request=request,
        user_id=user.id, resource_type="User", resource_id=user.id,
    )
    return user


@router.post("/login", response_model=TokenResponse)
@limiter.limit(settings.RATE_LIMIT_AUTH)
async def login(request: Request, payload: LoginRequest, db: Session = Depends(get_db)):
    try:
        tokens = authenticate_user(db, payload)
    except HTTPException as e:
        # Deliberately don't log the attempted email in plaintext here for
        # failed logins — combined with IP address, that's enough to audit
        # brute-force patterns without persisting a list of "emails someone
        # tried logging in as" (a mild PII/enumeration-adjacent concern).
        record_audit_event(db, action="auth.login", success=False, request=request, details={"reason": e.detail})
        raise

    user = db.query(models.User).filter(models.User.email == payload.email).first()
    record_audit_event(
        db, action="auth.login", success=True, request=request,
        user_id=user.id if user else None,
    )
    return tokens


@router.get("/profile", response_model=ProfileResponse)
async def profile(current_user: models.User = Depends(get_current_user)):
    return current_user
