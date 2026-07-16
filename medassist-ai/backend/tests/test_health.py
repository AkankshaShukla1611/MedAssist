import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from app.core import health


def test_check_database_ok(db_session):
    result = health.check_database(db_session)
    assert result["status"] == "ok"


def test_check_database_failed_on_broken_session():
    broken_session = MagicMock()
    broken_session.execute.side_effect = Exception("connection refused")
    result = health.check_database(broken_session)
    assert result["status"] == "failed"
    assert "connection refused" in result["detail"]


def test_check_faiss_ok_when_directory_writable(tmp_path, monkeypatch):
    monkeypatch.setattr(health.settings, "FAISS_PATH", str(tmp_path / "index.faiss"))
    result = health.check_faiss()
    assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_check_llm_ok_on_200():
    mock_response = MagicMock(status_code=200)
    with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_response)):
        result = await health.check_llm()
    assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_check_llm_degraded_on_non_200():
    mock_response = MagicMock(status_code=503)
    with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_response)):
        result = await health.check_llm()
    assert result["status"] == "degraded"


@pytest.mark.asyncio
async def test_check_llm_failed_when_unreachable():
    with patch("httpx.AsyncClient.get", new=AsyncMock(side_effect=ConnectionError("refused"))):
        result = await health.check_llm()
    assert result["status"] == "failed"


def test_check_redis_failed_when_client_unavailable():
    with patch("app.core.cache.get_redis_client", return_value=None):
        result = health.check_redis()
    assert result["status"] == "failed"


def test_check_redis_ok_when_ping_succeeds():
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    with patch("app.core.cache.get_redis_client", return_value=mock_client):
        result = health.check_redis()
    assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_readiness_report_excludes_redis_from_pass_fail_gate(db_session):
    """
    Core design decision under test: Redis being down must NOT flip overall
    readiness to not_ready — cache failures degrade gracefully (see
    cache.py), so gating readiness on Redis would take healthy instances
    out of the load balancer over a non-fatal dependency.
    """
    with patch.object(health, "check_llm", new=AsyncMock(return_value={"status": "ok"})), \
         patch.object(health, "check_redis", return_value={"status": "failed", "detail": "down"}):
        report = await health.readiness_report(db_session)

    assert report["status"] == "ready"
    assert report["checks"]["redis"]["status"] == "failed"


@pytest.mark.asyncio
async def test_readiness_report_not_ready_when_database_fails():
    broken_session = MagicMock()
    broken_session.execute.side_effect = Exception("db down")

    with patch.object(health, "check_llm", new=AsyncMock(return_value={"status": "ok"})), \
         patch.object(health, "check_redis", return_value={"status": "ok"}):
        report = await health.readiness_report(broken_session)

    assert report["status"] == "not_ready"
