import json
from unittest.mock import patch, MagicMock


def test_non_admin_cannot_trigger_evaluation(client, auth_headers):
    r = client.post("/admin/evaluate", json={}, headers=auth_headers)
    assert r.status_code == 403


def test_trigger_evaluation_creates_queued_run(client, admin_headers):
    with patch("app.tasks.evaluation_tasks.run_evaluation_task.delay") as mock_delay:
        mock_delay.return_value = MagicMock(id="fake-task-id-123")
        r = client.post("/admin/evaluate", json={"max_questions": 5}, headers=admin_headers)

    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "queued"
    assert body["celery_task_id"] == "fake-task-id-123"
    assert "id" in body


def test_trigger_evaluation_defaults_work_with_empty_body(client, admin_headers):
    with patch("app.tasks.evaluation_tasks.run_evaluation_task.delay") as mock_delay:
        mock_delay.return_value = MagicMock(id="task-id")
        r = client.post("/admin/evaluate", json={}, headers=admin_headers)
    assert r.status_code == 202


def test_list_evaluations_returns_created_runs(client, admin_headers):
    with patch("app.tasks.evaluation_tasks.run_evaluation_task.delay") as mock_delay:
        mock_delay.return_value = MagicMock(id="task-id")
        client.post("/admin/evaluate", json={}, headers=admin_headers)

    r = client.get("/admin/evaluations", headers=admin_headers)
    assert r.status_code == 200
    runs = r.json()
    assert len(runs) == 1
    assert runs[0]["status"] == "queued"


def test_get_evaluation_detail_json(client, admin_headers):
    with patch("app.tasks.evaluation_tasks.run_evaluation_task.delay") as mock_delay:
        mock_delay.return_value = MagicMock(id="task-id")
        created = client.post("/admin/evaluate", json={}, headers=admin_headers).json()

    r = client.get(f"/admin/evaluations/{created['id']}", headers=admin_headers)
    assert r.status_code == 200
    assert r.json()["id"] == created["id"]
    assert r.json()["status"] == "queued"


def test_get_evaluation_detail_markdown(client, admin_headers):
    with patch("app.tasks.evaluation_tasks.run_evaluation_task.delay") as mock_delay:
        mock_delay.return_value = MagicMock(id="task-id")
        created = client.post("/admin/evaluate", json={}, headers=admin_headers).json()

    r = client.get(f"/admin/evaluations/{created['id']}?format=markdown", headers=admin_headers)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/markdown")
    assert f"Evaluation Run #{created['id']}" in r.text


def test_get_nonexistent_evaluation_returns_404(client, admin_headers):
    r = client.get("/admin/evaluations/999999", headers=admin_headers)
    assert r.status_code == 404


def test_task_status_endpoint_returns_shape(client, admin_headers):
    r = client.get("/admin/tasks/some-task-id", headers=admin_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["task_id"] == "some-task-id"
    assert "status" in body


def test_non_admin_cannot_query_audit_logs(client, auth_headers):
    r = client.get("/admin/audit-logs", headers=auth_headers)
    assert r.status_code == 403


def test_audit_logs_record_login_events(client, registered_user, admin_headers):
    client.post("/auth/login", json={"email": registered_user["email"], "password": registered_user["password"]})
    client.post("/auth/login", json={"email": registered_user["email"], "password": "WrongPassword123"})

    r = client.get("/admin/audit-logs?action=auth.login", headers=admin_headers)
    assert r.status_code == 200
    logs = r.json()
    successes = [l for l in logs if l["success"]]
    failures = [l for l in logs if not l["success"]]
    assert len(successes) >= 1
    assert len(failures) >= 1


def test_audit_logs_record_evaluation_trigger(client, admin_headers):
    with patch("app.tasks.evaluation_tasks.run_evaluation_task.delay") as mock_delay:
        mock_delay.return_value = MagicMock(id="task-id")
        client.post("/admin/evaluate", json={}, headers=admin_headers)

    r = client.get("/admin/audit-logs?action=evaluation.trigger", headers=admin_headers)
    assert r.status_code == 200
    assert len(r.json()) == 1
    assert r.json()[0]["success"] is True
