import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Must be set before app.core.config is imported anywhere (JWT_SECRET has no default).
os.environ.setdefault("JWT_SECRET", "test-secret-key-for-pytest-only-do-not-use-in-prod")
os.environ.setdefault("ENV", "test")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "True")
os.environ.setdefault("CACHE_ENABLED", "False")  # no Redis in the test environment; cache degrades to no-op by design

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from app.database.database import Base, get_db
from app.main import app


@pytest.fixture(scope="function")
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db_session: Session):
    def _override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = _override_get_db
    app.state.limiter.reset()  # each test gets a fresh rate-limit budget
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def registered_user(client):
    payload = {
        "name": "Dr. Jane Test",
        "email": "jane.test@example.com",
        "password": "StrongPass123",
        "role": "doctor",
    }
    response = client.post("/auth/register", json=payload)
    assert response.status_code == 201
    return payload


@pytest.fixture
def auth_headers(client, registered_user):
    response = client.post("/auth/login", json={
        "email": registered_user["email"],
        "password": registered_user["password"],
    })
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_headers(client, db_session):
    """
    Admin accounts can't self-register (see schemas/auth.py — deliberate
    security control), so tests create one directly in the DB, mirroring
    what app.utils.create_admin does for real deployments.
    """
    from app.database import models
    from app.core.security import hash_password

    admin = models.User(
        name="Admin Test", email="admin.test@example.com",
        password_hash=hash_password("AdminPass123"), role=models.UserRole.ADMIN,
    )
    db_session.add(admin)
    db_session.commit()

    response = client.post("/auth/login", json={"email": "admin.test@example.com", "password": "AdminPass123"})
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
