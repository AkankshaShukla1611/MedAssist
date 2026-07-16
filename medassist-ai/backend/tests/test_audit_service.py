from unittest.mock import MagicMock

from app.services.audit_service import record_audit_event, query_audit_logs
from app.core.ip_utils import get_client_ip


def test_record_audit_event_persists_row(db_session):
    record_audit_event(db_session, action="test.action", success=True, user_id=1, resource_type="Thing", resource_id=42)
    logs = query_audit_logs(db_session)
    assert len(logs) == 1
    assert logs[0].action == "test.action"
    assert logs[0].success is True
    assert logs[0].resource_id == "42"


def test_record_audit_event_with_no_request_uses_internal_endpoint(db_session):
    record_audit_event(db_session, action="test.internal", success=True, request=None)
    logs = query_audit_logs(db_session)
    assert logs[0].endpoint == "internal"


def test_record_audit_event_never_raises_on_db_failure(db_session):
    broken_db = MagicMock()
    broken_db.add.side_effect = Exception("simulated DB failure")
    # Should not raise, even though the underlying DB write fails.
    record_audit_event(broken_db, action="test.action", success=True)


def test_query_audit_logs_filters_by_action(db_session):
    record_audit_event(db_session, action="auth.login", success=True)
    record_audit_event(db_session, action="document.upload", success=True)
    results = query_audit_logs(db_session, action="auth.login")
    assert len(results) == 1
    assert results[0].action == "auth.login"


def test_query_audit_logs_filters_by_success(db_session):
    record_audit_event(db_session, action="auth.login", success=True)
    record_audit_event(db_session, action="auth.login", success=False)
    results = query_audit_logs(db_session, success=False)
    assert len(results) == 1
    assert results[0].success is False


def test_query_audit_logs_respects_limit_cap(db_session):
    for i in range(10):
        record_audit_event(db_session, action="test.action", success=True)
    results = query_audit_logs(db_session, limit=1000)  # should be capped at 500 internally, but only 10 exist
    assert len(results) == 10


def test_client_ip_prefers_x_forwarded_for():
    request = MagicMock()
    request.headers = {"X-Forwarded-For": "1.2.3.4, 5.6.7.8"}
    assert get_client_ip(request) == "1.2.3.4"


def test_client_ip_falls_back_to_client_host():
    request = MagicMock()
    request.headers = {}
    request.client.host = "10.0.0.1"
    assert get_client_ip(request) == "10.0.0.1"


def test_audit_event_with_no_request_stores_null_ip(db_session):
    # Internal/non-HTTP-triggered audit events (request=None) should still
    # record cleanly, with ip_address left null rather than raising.
    record_audit_event(db_session, action="system.internal_event", success=True, request=None)
    results = query_audit_logs(db_session, action="system.internal_event")
    assert len(results) == 1
    assert results[0].ip_address is None
