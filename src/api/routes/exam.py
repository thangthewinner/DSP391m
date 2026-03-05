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

    return {
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
            }
            for seg in segments
        ],
        "total_segments": len(segments),
    }


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

    if session.last_transcript is not None:
        tr = session.last_transcript
        session.last_transcript = None
        events.append({
            "type": "transcript_log",
            "text": tr["text"],
            "confidence": tr["confidence"],
            "similarity": tr["similarity"],
            "timestamp": tr["timestamp"],
            "speaker": tr.get("speaker", ""),
            "speaker_role": tr.get("speaker_role", ""),  # thí sinh / người lạ
        })

    if session.last_slm_alert is not None:
        alert = session.last_slm_alert
        session.last_slm_alert = None
        events.append({
            "type": "slm_alert",
            "text": alert["text"],
            "similarity": alert["similarity"],
            "timestamp": alert["timestamp"],
        })

    if session.last_diarization_result is not None:
        result = session.last_diarization_result
        session.last_diarization_result = None
        events.append({
            "type": "diarization_log",
            "num_speakers": result["num_speakers"],
            "speakers": result["speakers"],
            "segments": result["segments"],
            "confidence": result["confidence"],
            "overlap": result["overlap"],
            "audio_duration": result["audio_duration"],
            "dominant_speaker": result.get("dominant_speaker"),  # Phase 8
            "timestamp": now_ts,
        })

    if session.last_overlap_detected:
        session.last_overlap_detected = False
        events.append({
            "type": "overlap_alert",
            "overlap_count": session.overlap_count,
            "timestamp": now_ts,
        })

    if session.last_verification_failed:
        session.last_verification_failed = False
        events.append({
            "type": "verification_alert",
            "similarity": session.last_verification_similarity,
            "failures_count": session.verification_failures_count,
            "timestamp": now_ts,
        })

    if session.cheating_flag:
        events.append({
            "type": "cheating_alert",
            "timestamp": now_ts,
        })

    return {"events": events, "timestamp": now_ts}


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
