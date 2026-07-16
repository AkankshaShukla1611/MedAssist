import io
from unittest.mock import patch, MagicMock

MINIMAL_PDF_BYTES = b"%PDF-1.4\n%fake minimal pdf content for testing\n%%EOF"


def _upload(client, admin_headers, filename="test.pdf", content=MINIMAL_PDF_BYTES, title="Test Doc"):
    with patch("app.api.upload.ingest_document_task") as mock_task:
        mock_task.delay.return_value = MagicMock(id="fake-task-id")
        return client.post(
            "/documents/upload",
            data={"title": title, "category": "Cardiology"},
            files={"file": (filename, io.BytesIO(content), "application/pdf")},
            headers=admin_headers,
        )


def test_upload_computes_and_stores_checksum(client, admin_headers, db_session):
    r = _upload(client, admin_headers)
    assert r.status_code == 201
    body = r.json()
    assert body["checksum_sha256"] is not None
    assert len(body["checksum_sha256"]) == 64  # sha256 hex digest length


def test_upload_enqueues_celery_task_not_background_task(client, admin_headers):
    with patch("app.api.upload.ingest_document_task") as mock_task:
        mock_task.delay.return_value = MagicMock(id="fake-task-id-456")
        r = client.post(
            "/documents/upload",
            data={"title": "Test Doc"},
            files={"file": ("test.pdf", io.BytesIO(MINIMAL_PDF_BYTES), "application/pdf")},
            headers=admin_headers,
        )
    assert r.status_code == 201
    mock_task.delay.assert_called_once()
    assert r.json()["celery_task_id"] == "fake-task-id-456"


def test_duplicate_upload_is_rejected(client, admin_headers):
    first = _upload(client, admin_headers, title="First Upload")
    assert first.status_code == 201

    second = _upload(client, admin_headers, filename="different_name.pdf", title="Second Upload With Different Title")
    assert second.status_code == 409
    assert "already" in second.json()["detail"].lower()


def test_different_content_is_not_flagged_as_duplicate(client, admin_headers):
    first = _upload(client, admin_headers, content=MINIMAL_PDF_BYTES, title="Doc A")
    second = _upload(client, admin_headers, content=MINIMAL_PDF_BYTES + b"extra unique bytes", title="Doc B")
    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["checksum_sha256"] != second.json()["checksum_sha256"]


def test_non_pdf_upload_rejected_by_magic_bytes(client, admin_headers):
    r = client.post(
        "/documents/upload",
        data={"title": "Fake PDF"},
        files={"file": ("test.pdf", io.BytesIO(b"not a real pdf at all"), "application/pdf")},
        headers=admin_headers,
    )
    assert r.status_code == 400


def test_upload_records_audit_event(client, admin_headers):
    _upload(client, admin_headers)
    r = client.get("/admin/audit-logs?action=document.upload", headers=admin_headers)
    assert r.status_code == 200
    logs = r.json()
    assert len(logs) == 1
    assert logs[0]["success"] is True


def test_duplicate_upload_records_failed_audit_event(client, admin_headers):
    _upload(client, admin_headers, title="Original")
    _upload(client, admin_headers, filename="other.pdf", title="Duplicate Attempt")

    r = client.get("/admin/audit-logs?action=document.upload&success=false", headers=admin_headers)
    assert r.status_code == 200
    assert len(r.json()) == 1
