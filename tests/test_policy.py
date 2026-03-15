"""Policy tests for enrollment, verification, and overlap behavior."""

import asyncio
from collections import deque
from datetime import datetime

import numpy as np
import pytest
from fastapi import HTTPException

from src.api.routes import exam as exam_routes
from src.core.config import settings
from src.core.models import ExamSessionCreate, SessionStatus
from src.core.session import session_manager
from src.processing.pipeline import AudioPipeline
from src.storage.transcript_store import TranscriptStore


class _Dummy:
    pass


class _FakeVerifierNotEnrolled:
    def is_enrolled(self, student_id: str) -> bool:
        return False


class _FakeVerifierAlwaysFail:
    def is_enrolled(self, student_id: str) -> bool:
        return True

    def verify(self, student_id: str, audio: np.ndarray, sample_rate: int) -> tuple[bool, float]:
        return False, 0.2


class _FakeOverlapDetector:
    def __init__(self, on_detect):
        self._on_detect = on_detect

    def detect(self, audio: np.ndarray, sample_rate: int) -> tuple[bool, float, list[dict]]:
        self._on_detect()
        return True, 0.9, []


class _FakePipelineForStart:
    def __init__(self):
        self.verifier = _FakeVerifierNotEnrolled()
        self.embedding = _Dummy()


class _FakePipelineNoVerifier:
    def __init__(self):
        self.verifier = None
        self.embedding = _Dummy()


class _FakeSTTHighConfidence:
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> dict:
        return {"text": "noi dung lien quan de thi", "confidence": 0.95}


class _FakeEmbeddingHighSimilarity:
    def similarity_to_question(self, text: str, question_embedding: object) -> float:
        return 0.95


class _FakeSLMAlwaysYes:
    def predict(self, exam_question: str, text: str) -> bool:
        return True


@pytest.mark.asyncio
async def test_start_exam_requires_enrollment(monkeypatch: pytest.MonkeyPatch):
    """Start endpoint must reject non-enrolled students."""
    monkeypatch.setattr(exam_routes.pipeline_module, "pipeline", _FakePipelineForStart())
    request = ExamSessionCreate(
        student_id="student_policy_not_enrolled",
        exam_id="exam_policy",
        exam_question="test",
    )

    with pytest.raises(HTTPException) as exc:
        await exam_routes.start_exam(request)

    assert exc.value.status_code == 403
    assert "required" in str(exc.value.detail).lower()


@pytest.mark.asyncio
async def test_start_exam_requires_verifier_available(monkeypatch: pytest.MonkeyPatch):
    """Start endpoint must reject when verifier service is unavailable."""
    monkeypatch.setattr(exam_routes.pipeline_module, "pipeline", _FakePipelineNoVerifier())
    request = ExamSessionCreate(
        student_id="student_policy_verifier_missing",
        exam_id="exam_policy",
        exam_question="test",
    )

    with pytest.raises(HTTPException) as exc:
        await exam_routes.start_exam(request)

    assert exc.value.status_code == 503
    assert "required" in str(exc.value.detail).lower()


@pytest.mark.asyncio
async def test_verification_fail_3_times_terminates_session(
    monkeypatch: pytest.MonkeyPatch,
):
    """Session should be cancelled after max verification failures."""
    monkeypatch.setattr(settings, "verification_interval", 0)
    monkeypatch.setattr(settings, "verification_max_failures", 3)
    monkeypatch.setattr(settings, "min_verification_audio_seconds", 0.1)

    pipeline = AudioPipeline(
        vad_processor=_Dummy(),
        stt_processor=_Dummy(),
        embedding_processor=_Dummy(),
        transcript_store=TranscriptStore(),
        speaker_verifier=_FakeVerifierAlwaysFail(),
        overlap_detector=None,
    )

    session = session_manager.create_session(
        student_id="student_policy_verify_fail",
        exam_id="exam_policy",
        exam_question="",
    )
    session_id = session.session_id
    min_samples = int(settings.min_verification_audio_seconds * settings.audio_sample_rate)
    pipeline._recent_audio[session_id] = deque([0.01] * min_samples, maxlen=min_samples)
    pipeline._should_stop[session_id] = False

    await asyncio.wait_for(pipeline._verification_loop(session_id), timeout=2)

    updated = session_manager.get_session(session_id)
    assert updated is not None
    assert updated.status == SessionStatus.CANCELLED
    assert updated.cheating_flag is True
    assert updated.verification_failures_count == 3
    assert any(ev.get("type") == "session_terminated" for ev in updated.events)


@pytest.mark.asyncio
async def test_overlap_detected_does_not_set_cheating_flag(
    monkeypatch: pytest.MonkeyPatch,
):
    """Overlap should not set cheating flag by itself."""
    async def _no_sleep(_: float) -> None:
        return

    monkeypatch.setattr(settings, "diarization_enabled", True)
    monkeypatch.setattr("src.processing.pipeline.asyncio.sleep", _no_sleep)

    session = session_manager.create_session(
        student_id="student_policy_overlap",
        exam_id="exam_policy",
        exam_question="",
    )
    session_id = session.session_id

    pipeline_ref: dict[str, AudioPipeline] = {}

    def _mark_stop() -> None:
        pipeline_ref["p"]._should_stop[session_id] = True

    pipeline = AudioPipeline(
        vad_processor=_Dummy(),
        stt_processor=_Dummy(),
        embedding_processor=_Dummy(),
        transcript_store=TranscriptStore(),
        speaker_verifier=None,
        overlap_detector=_FakeOverlapDetector(_mark_stop),
    )
    pipeline_ref["p"] = pipeline

    window_samples = int(15.0 * settings.audio_sample_rate)
    pipeline._recent_audio[session_id] = deque([0.0] * window_samples, maxlen=window_samples)
    pipeline._should_stop[session_id] = False

    start = datetime.now()
    await asyncio.wait_for(pipeline._diarization_loop(session_id), timeout=2)
    updated = session_manager.get_session(session_id)
    assert updated is not None
    assert updated.overlap_count >= 1
    assert updated.cheating_flag is False
    assert updated.status == SessionStatus.ACTIVE
    assert (datetime.now() - start).total_seconds() < 1.5


@pytest.mark.asyncio
async def test_exam_related_content_from_other_speaker_sets_cheating_flag(
    monkeypatch: pytest.MonkeyPatch,
):
    """Any speaker saying exam-related content should trigger cheating."""
    pipeline = AudioPipeline(
        vad_processor=_Dummy(),
        stt_processor=_FakeSTTHighConfidence(),
        embedding_processor=_FakeEmbeddingHighSimilarity(),
        transcript_store=TranscriptStore(),
        slm_processor=_FakeSLMAlwaysYes(),
        speaker_verifier=None,
        overlap_detector=None,
    )

    session = session_manager.create_session(
        student_id="student_policy_speaker_other",
        exam_id="exam_policy",
        exam_question="Cau hoi de thi",
    )
    session_id = session.session_id
    session.question_embedding = object()

    await pipeline._process_buffer(
        session_id=session_id,
        audio=np.zeros(16000, dtype=np.float32),
        timestamp=12.0,
        speaker="spk_1",
        speaker_role="người lạ",
    )

    updated = session_manager.get_session(session_id)
    assert updated is not None
    assert updated.cheating_flag is True
    assert any(ev.get("type") == "slm_alert" for ev in updated.events)
    cheating_events = [ev for ev in updated.events if ev.get("type") == "cheating_alert"]
    assert len(cheating_events) == 1
    assert cheating_events[0].get("speaker") == "spk_1"
