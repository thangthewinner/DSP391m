"""Exam session management endpoints."""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, status

from src.core.config import settings
from src.core.models import (
    ExamSessionCreate,
    ExamSessionResponse,
    ExamSessionStopResponse,
    ExamStatusResponse,
    SessionStatus,
)
from src.core.session import session_manager
from src.processing import pipeline as pipeline_module
from src.processing.decision_engine import decision_engine
from src.storage.transcript_store import TranscriptStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/exam", tags=["exam"])


@router.post("/start", response_model=ExamSessionResponse, status_code=status.HTTP_200_OK)
async def start_exam(request: ExamSessionCreate):
    """Start a new exam monitoring session."""
    try:
        session = session_manager.create_session(
            student_id=request.student_id,
            exam_id=request.exam_id,
            exam_question=request.exam_question,
        )

        # Pre-compute question embedding for similarity detection
        p = pipeline_module.pipeline
        if p and request.exam_question.strip():
            import asyncio
            loop = asyncio.get_event_loop()
            session.question_embedding = await loop.run_in_executor(
                None, p.embedding.embed, request.exam_question
            )
            logger.info(f"Question embedding computed for session {session.session_id}")

        websocket_url = (
            f"ws://{settings.api_host}:{settings.api_port}/ws/audio/{session.session_id}"
        )

        logger.info(f"Started session {session.session_id} for student {request.student_id}")

        return ExamSessionResponse(
            session_id=session.session_id,
            student_id=session.student_id,
            exam_id=session.exam_id,
            status=session.status,
            started_at=session.started_at,
            websocket_url=websocket_url,
        )

    except Exception as e:
        logger.error(f"Error starting exam session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop/{session_id}", response_model=ExamSessionStopResponse)
async def stop_exam(session_id: str):
    """Stop an active exam session."""
    session = session_manager.stop_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    logger.info(f"Stopped session {session_id}")

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
        current_suspicion_score=session.suspicion_score,
        cheating_flag=session.cheating_flag,
        flagged_segments_count=session.flagged_segments_count,
        elapsed_time_seconds=elapsed,
        last_verification_time=None,
        verification_status="passed",
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

    decision_report = decision_engine.get_report(session_id)

    return {
        "session_id": session_id,
        "student_id": session.student_id,
        "exam_id": session.exam_id,
        "exam_question": session.exam_question,
        "status": session.status.value,
        "started_at": session.started_at.isoformat(),
        "ended_at": session.ended_at.isoformat() if session.ended_at else None,
        "elapsed_seconds": elapsed,
        # Decision engine results
        "cheating_detected": decision_report["cheating_detected"],
        "suspicion_score": decision_report["suspicion_score"],
        "max_suspicion_score": decision_report["max_suspicion_score"],
        "confidence": decision_report["confidence"],
        "rationale": decision_report["rationale"],
        "flagged_segments": decision_report["flagged_segments"],
        # Full transcript
        "transcript": [
            {
                "start": seg.timestamp_start,
                "end": seg.timestamp_end,
                "text": seg.text,
                "confidence": seg.confidence,
            }
            for seg in segments
        ],
        "total_segments": len(segments),
    }


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
