"""
One-off script to create the first admin account.
Public /auth/register can never create an admin (see schemas/auth.py) —
this is the only way to mint one, and it must be run from inside the
backend container/host, not exposed as an HTTP endpoint.

Usage:
    docker compose exec backend python -m app.utils.create_admin
"""
import getpass

from app.database.database import SessionLocal, Base, engine
from app.database import models
from app.core.security import hash_password


def main():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    try:
        name = input("Admin name: ").strip()
        email = input("Admin email: ").strip().lower()
        password = getpass.getpass("Admin password (min 10 chars, mixed case + digit): ")

        existing = db.query(models.User).filter(models.User.email == email).first()
        if existing:
            print(f"A user with email {email} already exists.")
            return

        admin = models.User(
            name=name,
            email=email,
            password_hash=hash_password(password),
            role=models.UserRole.ADMIN,
        )
        db.add(admin)
        db.commit()
        print(f"Admin account created for {email}.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
