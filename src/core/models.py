"""Pydantic data models for API and internal state."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class SessionStatus(str, Enum):
    """Exam session status."""

    ACTIVE = "active"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


class AudioChunkMessage(BaseModel):
    """WebSocket audio chunk message from client."""

    type: str = Field(default="audio_chunk", description="Message type")
    data: str = Field(..., description="Base64 encoded PCM audio data")
    timestamp: float = Field(..., description="Client timestamp")
    sample_rate: int = Field(default=16000, description="Audio sample rate")
    channels: int = Field(default=1, description="Number of audio channels")


class WebSocketResponse(BaseModel):
    """WebSocket response message to client."""

    type: str = Field(..., description="Response type")
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())


class AckResponse(WebSocketResponse):
    """Acknowledgment response."""

    type: str = Field(default="ack")
    chunk_id: Optional[int] = Field(None, description="Chunk ID")
    received_at: float = Field(default_factory=lambda: datetime.now().timestamp())


class StatusUpdateResponse(WebSocketResponse):
    """Status update response."""

    type: str = Field(default="status_update")
    suspicion_score: float = Field(default=0.0, description="Current suspicion score")
    cheating_flag: bool = Field(default=False, description="Cheating detected flag")


class ErrorResponse(WebSocketResponse):
    """Error response."""

    type: str = Field(default="error")
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")


class TranscriptSegment(BaseModel):
    """Speech transcript segment."""

    timestamp_start: float = Field(..., description="Segment start time in seconds")
    timestamp_end: float = Field(..., description="Segment end time in seconds")
    text: str = Field(..., description="Transcribed text")
    confidence: float = Field(..., description="Confidence score (0-1)")


class ExamSessionCreate(BaseModel):
    """Request to create exam session."""

    student_id: str = Field(..., description="Student user ID")
    exam_id: str = Field(default="default", description="Exam identifier")
    exam_question: str = Field(default="", description="Exam question text for semantic analysis")
    duration_minutes: int = Field(default=60, description="Expected exam duration")


class ExamSessionResponse(BaseModel):
    """Exam session response."""

    session_id: str = Field(..., description="Unique session identifier")
    student_id: str = Field(..., description="Student user ID")
    exam_id: str = Field(..., description="Exam identifier")
    status: SessionStatus = Field(..., description="Session status")
    started_at: datetime = Field(..., description="Session start time")
    websocket_url: str = Field(..., description="WebSocket URL for audio streaming")


class ExamSessionStop(BaseModel):
    """Request to stop exam session."""

    session_id: str = Field(..., description="Session identifier")


class ExamSessionStopResponse(BaseModel):
    """Stop exam session response."""

    session_id: str = Field(..., description="Session identifier")
    status: SessionStatus = Field(..., description="Session status")
    ended_at: datetime = Field(..., description="Session end time")
    report_url: str = Field(..., description="Report URL")


class ExamStatusResponse(BaseModel):
    """Exam session status response."""

    session_id: str = Field(..., description="Session identifier")
    status: SessionStatus = Field(..., description="Session status")
    current_suspicion_score: float = Field(default=0.0, description="Current suspicion score")
    cheating_flag: bool = Field(default=False, description="Cheating detected")
    flagged_segments_count: int = Field(default=0, description="Number of flagged segments")
    elapsed_time_seconds: float = Field(..., description="Elapsed time in seconds")
    last_verification_time: Optional[datetime] = Field(
        None, description="Last verification time"
    )
    verification_status: str = Field(default="passed", description="Verification status")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Health status")
    models_loaded: bool = Field(..., description="Models loaded status")
    version: str = Field(..., description="API version")


class SessionState(BaseModel):
    """Internal session state."""

    session_id: str
    student_id: str
    exam_id: str
    exam_question: str
    status: SessionStatus
    started_at: datetime
    ended_at: Optional[datetime] = None
    suspicion_score: float = 0.0
    cheating_flag: bool = False
    transcript_segments: list[TranscriptSegment] = Field(default_factory=list)
    flagged_segments_count: int = 0

    model_config = {"arbitrary_types_allowed": True}
