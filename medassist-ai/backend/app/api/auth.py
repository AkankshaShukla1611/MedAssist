from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database import models
from app.schemas.auth import RegisterRequest, LoginRequest, TokenResponse, ProfileResponse
from app.services.auth_service import register_user, authenticate_user
from app.core.security import get_current_user
from app.core.limiter import limiter
from app.core.config import settings

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=ProfileResponse, status_code=201)
@limiter.limit(settings.RATE_LIMIT_AUTH)
async def register(request: Request, payload: RegisterRequest, db: Session = Depends(get_db)):
    user = register_user(db, payload)
    return user


@router.post("/login", response_model=TokenResponse)
@limiter.limit(settings.RATE_LIMIT_AUTH)
async def login(request: Request, payload: LoginRequest, db: Session = Depends(get_db)):
    tokens = authenticate_user(db, payload)
    return tokens


@router.get("/profile", response_model=ProfileResponse)
async def profile(current_user: models.User = Depends(get_current_user)):
    return current_user
