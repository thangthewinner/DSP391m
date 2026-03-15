"""Tests for session index querying and retention cleanup."""

from datetime import datetime, timedelta

import pytest

from src.storage.transcript_store import TranscriptStore


@pytest.mark.asyncio
async def test_query_sessions_with_filters(tmp_path):
    store = TranscriptStore(storage_dir=tmp_path / "sessions")
    now = datetime.now()

    await store.init_session(
        session_id="s1",
        student_id="stu_a",
        exam_id="exam_1",
        exam_question="q1",
        started_at=now - timedelta(minutes=10),
    )
    await store.finalize_session(
        session_id="s1",
        status="completed",
        ended_at=now - timedelta(minutes=5),
        cheating_detected=False,
    )
    await store.init_session(
        session_id="s2",
        student_id="stu_b",
        exam_id="exam_2",
        exam_question="q2",
        started_at=now - timedelta(minutes=2),
    )

    result = await store.query_sessions(student_id="stu_a", limit=20, offset=0)
    assert result["total"] == 1
    assert result["items"][0]["session_id"] == "s1"
    assert result["items"][0]["status"] == "completed"

    ranged = await store.query_sessions(
        started_from=now - timedelta(minutes=3),
        started_to=now,
        limit=20,
        offset=0,
    )
    assert ranged["total"] == 1
    assert ranged["items"][0]["session_id"] == "s2"


@pytest.mark.asyncio
async def test_apply_retention_removes_old_buckets_and_prunes_index(tmp_path):
    store = TranscriptStore(storage_dir=tmp_path / "sessions")
    now = datetime.now()
    old_started = now - timedelta(days=45)
    new_started = now - timedelta(days=1)

    await store.init_session(
        session_id="old_session",
        student_id="stu_old",
        exam_id="exam_old",
        exam_question="",
        started_at=old_started,
    )
    old_dir = store._resolve_session_dir("old_session", create_if_missing=False)
    assert old_dir is not None
    forced_old_bucket = store.storage_dir / old_started.strftime("%Y-%m-%d") / "old_session"
    forced_old_bucket.parent.mkdir(parents=True, exist_ok=True)
    old_dir.rename(forced_old_bucket)

    await store.init_session(
        session_id="new_session",
        student_id="stu_new",
        exam_id="exam_new",
        exam_question="",
        started_at=new_started,
    )

    result = await store.apply_retention(retention_days=30)
    assert result["enabled"] is True
    assert result["deleted_session_dirs"] >= 1

    sessions = await store.query_sessions(limit=20, offset=0)
    ids = {item["session_id"] for item in sessions["items"]}
    assert "new_session" in ids
    assert "old_session" not in ids
