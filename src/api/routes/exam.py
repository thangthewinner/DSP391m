"""Exam session management endpoints."""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, status

from src.core.config import settings
from src.core.models import (
    ExamSessionCreate,
    ExamSessionResponse,
    ExamSessionStopResponse,
    ExamStatusResponse,
)
from src.core.session import session_manager
from src.processing import pipeline as pipeline_module
from src.storage.transcript_store import TranscriptStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/exam", tags=["exam"])


@router.post(
    "/start", response_model=ExamSessionResponse, status_code=status.HTTP_200_OK
)
async def start_exam(request: ExamSessionCreate):
    """Start a new exam monitoring session."""
    try:
        p = pipeline_module.pipeline
        if p is None or p.verifier is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Speaker verification is required but verifier is unavailable",
            )
        if not p.verifier.is_enrolled(request.student_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Student {request.student_id} is not enrolled. "
                    "Voice enrollment is required before starting exam."
                ),
            )

        session = session_manager.create_session(
            student_id=request.student_id,
            exam_id=request.exam_id,
            exam_question=request.exam_question,
        )
        await TranscriptStore().init_session(
            session_id=session.session_id,
            student_id=session.student_id,
            exam_id=session.exam_id,
            exam_question=session.exam_question,
            started_at=session.started_at,
        )

        # Pre-compute question embedding for similarity detection
        if p and request.exam_question.strip():
            import asyncio

            loop = asyncio.get_event_loop()
            session.question_embedding = await loop.run_in_executor(
                None, p.embedding.embed, request.exam_question
            )
            logger.info(f"Question embedding computed for session {session.session_id}")

        websocket_url = f"ws://{settings.api_host}:{settings.api_port}/ws/audio/{session.session_id}"

        logger.info(
            f"Started session {session.session_id} for student {request.student_id}"
        )

        return ExamSessionResponse(
            session_id=session.session_id,
            student_id=session.student_id,
            exam_id=session.exam_id,
            status=session.status,
            started_at=session.started_at,
            websocket_url=websocket_url,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting exam session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop/{session_id}", response_model=ExamSessionStopResponse)
async def stop_exam(session_id: str):
    """Stop an active exam session."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    # Signal pipeline stop first to keep producer/consumer lifecycle consistent.
    p = pipeline_module.pipeline
    if p is not None:
        p.stop_processing(session_id)

    # Mark session completed.
    session = session_manager.stop_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    # Best-effort cleanup of pending audio chunks.
    audio_queue = session_manager.get_audio_queue(session_id)
    if audio_queue is not None:
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except Exception:
                break

    logger.info(f"Stopped session {session_id}")
    await TranscriptStore().finalize_session(
        session_id=session.session_id,
        status=session.status.value,
        ended_at=session.ended_at or datetime.now(),
        cheating_detected=session.cheating_flag,
    )

    return ExamSessionStopResponse(
        session_id=session.session_id,
        status=session.status,
        ended_at=session.ended_at or datetime.now(),
        report_url=f"/api/exam/report/{session.session_id}",
    )


@router.get("/status/{session_id}", response_model=ExamStatusResponse)
async def get_exam_status(session_id: str):
    """Get current status of an exam session."""
    session = session_manager.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    if session.ended_at:
        elapsed = (session.ended_at - session.started_at).total_seconds()
    else:
        elapsed = (datetime.now() - session.started_at).total_seconds()

    return ExamStatusResponse(
        session_id=session.session_id,
        status=session.status,
        cheating_flag=session.cheating_flag,
        elapsed_time_seconds=elapsed,
        last_verification_time=session.last_verification_time,
        verification_status="failed" if session.last_verification_failed else "passed",
    )


@router.get("/report/{session_id}")
async def get_exam_report(session_id: str):
    """Get full exam report including transcripts."""
    session = session_manager.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    store = TranscriptStore()
    segments = await store.load_segments(session_id)

    elapsed = None
    if session.ended_at:
        elapsed = (session.ended_at - session.started_at).total_seconds()
    elif session.started_at:
        elapsed = (datetime.now() - session.started_at).total_seconds()

    report = {
        "schema_version": settings.report_schema_version,
        "session_id": session_id,
        "student_id": session.student_id,
        "exam_id": session.exam_id,
        "exam_question": session.exam_question,
        "status": session.status.value,
        "started_at": session.started_at.isoformat(),
        "ended_at": session.ended_at.isoformat() if session.ended_at else None,
        "elapsed_seconds": elapsed,
        "cheating_detected": session.cheating_flag,
        "verification_failures": session.verification_failures_count,
        "overlap_count": session.overlap_count,
        # Full transcript
        "transcript": [
            {
                "start": seg.timestamp_start,
                "end": seg.timestamp_end,
                "text": seg.text,
                "confidence": seg.confidence,
                "speaker_id": seg.speaker_id,
                "speaker_role": seg.speaker_role,
                "source": seg.source,
                "similarity": seg.similarity,
                "slm_verdict": seg.slm_verdict,
                "is_exam_related": seg.is_exam_related,
                "is_candidate_speech": seg.is_candidate_speech,
            }
            for seg in segments
        ],
        "total_segments": len(segments),
    }
    await store.save_report(session_id, report)
    return report


@router.get("/events/{session_id}")
async def poll_events(session_id: str):
    """
    Return and clear all pending events for a session.
    Frontend polls this every ~1.5s instead of maintaining a second WebSocket.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    import datetime as _dt

    now_ts = _dt.datetime.now().timestamp()
    events = []

    while session.events:
        events.append(session.events.popleft())

    return {
        "schema_version": settings.event_schema_version,
        "events": events,
        "timestamp": now_ts,
    }


@router.get("/history")
async def query_session_history(
    student_id: Optional[str] = None,
    exam_id: Optional[str] = None,
    status_filter: Optional[str] = None,
    started_from: Optional[str] = None,
    started_to: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """Query persisted session metadata with filters and pagination."""
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")

    def _parse_iso(name: str, value: Optional[str]) -> Optional[datetime]:
        if value is None:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid {name} datetime format. Expected ISO-8601.",
            ) from exc

    from_dt = _parse_iso("started_from", started_from)
    to_dt = _parse_iso("started_to", started_to)
    if from_dt and to_dt and from_dt > to_dt:
        raise HTTPException(
            status_code=400, detail="started_from must be <= started_to"
        )

    store = TranscriptStore()
    result = await store.query_sessions(
        student_id=student_id,
        exam_id=exam_id,
        status=status_filter,
        started_from=from_dt,
        started_to=to_dt,
        limit=limit,
        offset=offset,
    )
    result["schema_version"] = settings.report_schema_version
    return result


@router.get("/sessions")
async def list_sessions():
    """List all active exam sessions."""
    active_sessions = session_manager.list_active_sessions()

    return {
        "sessions": [
            {
                "session_id": s.session_id,
                "student_id": s.student_id,
                "exam_id": s.exam_id,
                "status": s.status.value,
                "started_at": s.started_at.isoformat(),
            }
            for s in active_sessions
        ],
        "total": len(active_sessions),
    }


@router.delete("/session/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str):
    """Delete a session and its transcript data."""
    session = session_manager.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    store = TranscriptStore()
    await store.delete_transcript(session_id)
    session_manager.cleanup_session(session_id)

    logger.info(f"Deleted session {session_id}")
