"""
Verifies the Alembic migration itself, not just the ORM models — these can
drift apart (e.g. someone adds a column to models.py and forgets to
generate a migration). Runs `alembic upgrade head` / `downgrade base`
against a throwaway SQLite file via subprocess (the real CLI path, not the
Python API) so this test catches the same thing a deploy would.
"""
import os
import subprocess
import sqlite3
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).parent.parent


def _run_alembic(command: str, db_path: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["JWT_SECRET"] = "test-secret-for-migration-check"
    env["DATABASE_URL"] = f"sqlite:///{db_path}"
    env["CACHE_ENABLED"] = "False"
    return subprocess.run(
        ["python3", "-m", "alembic"] + command.split(),
        cwd=str(BACKEND_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.fixture
def temp_db_path(tmp_path):
    path = tmp_path / "migration_test.db"
    yield path
    if path.exists():
        path.unlink()


def test_alembic_upgrade_head_creates_all_expected_tables(temp_db_path):
    result = _run_alembic("upgrade head", temp_db_path)
    assert result.returncode == 0, f"alembic upgrade failed:\n{result.stdout}\n{result.stderr}"

    conn = sqlite3.connect(str(temp_db_path))
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()

    expected_tables = {
        "users", "documents", "chunks", "conversations",
        "retrieved_sources", "retrieval_logs", "evaluation_runs", "audit_logs",
        "alembic_version",
    }
    missing = expected_tables - tables
    assert not missing, f"Migration did not create expected tables: {missing}"


def test_alembic_downgrade_base_removes_all_tables(temp_db_path):
    upgrade_result = _run_alembic("upgrade head", temp_db_path)
    assert upgrade_result.returncode == 0

    downgrade_result = _run_alembic("downgrade base", temp_db_path)
    assert downgrade_result.returncode == 0, f"alembic downgrade failed:\n{downgrade_result.stdout}\n{downgrade_result.stderr}"

    conn = sqlite3.connect(str(temp_db_path))
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()

    # alembic_version itself remains (it tracks "no migrations applied"), but
    # every application table must be gone.
    app_tables = tables - {"alembic_version"}
    assert not app_tables, f"Downgrade left tables behind: {app_tables}"
