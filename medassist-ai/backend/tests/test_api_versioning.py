def test_v1_and_legacy_routes_both_serve_the_same_endpoint(client):
    v1_response = client.post("/api/v1/auth/register", json={
        "name": "Dr V1", "email": "v1@example.com", "password": "StrongPass123", "role": "doctor",
    })
    assert v1_response.status_code == 201
    assert "Deprecation" not in v1_response.headers

    legacy_response = client.post("/auth/register", json={
        "name": "Dr Legacy", "email": "legacy@example.com", "password": "StrongPass123", "role": "doctor",
    })
    assert legacy_response.status_code == 201
    assert legacy_response.headers.get("Deprecation") == "true"
    assert "/api/v1/auth/register" in legacy_response.headers.get("Link", "")


def test_health_and_metrics_paths_are_not_marked_deprecated(client):
    response = client.get("/health")
    assert "Deprecation" not in response.headers

    response = client.get("/metrics")
    assert "Deprecation" not in response.headers
