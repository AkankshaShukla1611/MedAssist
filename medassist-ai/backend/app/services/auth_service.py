from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from app.database import models
from app.core.security import hash_password, verify_password, create_access_token, create_refresh_token
from app.schemas.auth import RegisterRequest, LoginRequest


def register_user(db: Session, payload: RegisterRequest) -> models.User:
    existing = db.query(models.User).filter(models.User.email == payload.email).first()
    if existing:
        # Same generic message whether the email exists or not would be ideal for
        # zero user-enumeration, but registration UX commonly needs this distinction.
        raise HTTPException(status_code=409, detail="An account with this email already exists")

    user = models.User(
        name=payload.name,
        email=payload.email,
        password_hash=hash_password(payload.password),
        role=payload.role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, payload: LoginRequest) -> dict:
    user = db.query(models.User).filter(models.User.email == payload.email).first()

    # Deliberately generic error + constant-time-ish flow: don't reveal whether
    # the email exists. verify_password still runs against a dummy hash if no
    # user is found, to reduce timing-based user enumeration.
    dummy_hash = "$2b$12$8pgoApEvOkLFo/VWemKQQu6mux96bacC0dmRtMsSk43CSvn5xEkiK"
    hash_to_check = user.password_hash if user else dummy_hash
    password_ok = verify_password(payload.password, hash_to_check)

    if not user or not password_ok:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")

    access_token = create_access_token(subject=str(user.id), role=user.role.value)
    refresh_token = create_refresh_token(subject=str(user.id))
    return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}
