def test_register_and_login_flow(client):
    register_payload = {
        "name": "Dr. Alice",
        "email": "alice@example.com",
        "password": "SecurePass123",
        "role": "doctor",
    }
    r = client.post("/auth/register", json=register_payload)
    assert r.status_code == 201
    assert r.json()["email"] == "alice@example.com"
    assert r.json()["role"] == "doctor"

    r = client.post("/auth/login", json={"email": "alice@example.com", "password": "SecurePass123"})
    assert r.status_code == 200
    assert "access_token" in r.json()
    assert "refresh_token" in r.json()


def test_cannot_self_register_as_admin(client):
    r = client.post("/auth/register", json={
        "name": "Sneaky",
        "email": "sneaky@example.com",
        "password": "SecurePass123",
        "role": "admin",
    })
    assert r.status_code == 422  # pydantic validator rejects it


def test_weak_password_rejected(client):
    r = client.post("/auth/register", json={
        "name": "Bob",
        "email": "bob@example.com",
        "password": "weak",
        "role": "medical_student",
    })
    assert r.status_code == 422


def test_duplicate_email_rejected(client, registered_user):
    r = client.post("/auth/register", json=registered_user)
    assert r.status_code == 409


def test_login_wrong_password_generic_error(client, registered_user):
    r = client.post("/auth/login", json={"email": registered_user["email"], "password": "WrongPassword1"})
    assert r.status_code == 401
    assert r.json()["detail"] == "Invalid email or password"


def test_login_nonexistent_email_same_generic_error(client):
    r = client.post("/auth/login", json={"email": "ghost@example.com", "password": "WhateverPass1"})
    assert r.status_code == 401
    assert r.json()["detail"] == "Invalid email or password"


def test_profile_requires_auth(client):
    r = client.get("/auth/profile")
    assert r.status_code == 401


def test_profile_with_valid_token(client, auth_headers):
    r = client.get("/auth/profile", headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["email"] == "jane.test@example.com"


def test_non_admin_cannot_upload_document(client, auth_headers):
    files = {"file": ("test.pdf", b"%PDF-1.4 fake content", "application/pdf")}
    r = client.post(
        "/documents/upload",
        data={"title": "Test Doc"},
        files=files,
        headers=auth_headers,
    )
    assert r.status_code == 403


def test_non_admin_cannot_access_admin_dashboard(client, auth_headers):
    r = client.get("/admin/dashboard", headers=auth_headers)
    assert r.status_code == 403
