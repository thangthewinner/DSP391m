"""Session state management."""

import asyncio
from datetime import datetime
from typing import Dict, Optional
from uuid import uuid4

from src.core.models import SessionState, SessionStatus


class SessionManager:
    """Manage exam session states."""

    def __init__(self):
        """Initialize session manager."""
        self._sessions: Dict[str, SessionState] = {}
        self._audio_queues: Dict[str, asyncio.Queue] = {}
        self._processing_tasks: Dict[str, asyncio.Task] = {}

    def create_session(
        self, student_id: str, exam_id: str, exam_question: str
    ) -> SessionState:
        """Create a new exam session."""
        session_id = str(uuid4())
        session = SessionState(
            session_id=session_id,
            student_id=student_id,
            exam_id=exam_id,
            exam_question=exam_question,
            status=SessionStatus.ACTIVE,
            started_at=datetime.now(),
        )
        self._sessions[session_id] = session
        self._audio_queues[session_id] = asyncio.Queue(maxsize=100)
        return session

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get session by ID."""
        return self._sessions.get(session_id)

    def get_audio_queue(self, session_id: str) -> Optional[asyncio.Queue]:
        """Get audio queue for session."""
        return self._audio_queues.get(session_id)

    def stop_session(self, session_id: str) -> Optional[SessionState]:
        """Stop an exam session."""
        session = self._sessions.get(session_id)
        if session:
            session.status = SessionStatus.COMPLETED
            session.ended_at = datetime.now()

            # Cancel processing task if exists
            if session_id in self._processing_tasks:
                task = self._processing_tasks[session_id]
                if not task.done():
                    task.cancel()
                del self._processing_tasks[session_id]

        return session

    def set_processing_task(self, session_id: str, task: asyncio.Task) -> None:
        """Set processing task for session."""
        self._processing_tasks[session_id] = task

    def cleanup_session(self, session_id: str) -> None:
        """Clean up session resources."""
        if session_id in self._audio_queues:
            del self._audio_queues[session_id]
        if session_id in self._processing_tasks:
            task = self._processing_tasks[session_id]
            if not task.done():
                task.cancel()
            del self._processing_tasks[session_id]

    def list_active_sessions(self) -> list[SessionState]:
        """List all active sessions."""
        return [s for s in self._sessions.values() if s.status == SessionStatus.ACTIVE]

    def list_all_sessions(self) -> list[SessionState]:
        """List all sessions regardless of status."""
        return list(self._sessions.values())


# Global session manager instance
session_manager = SessionManager()
