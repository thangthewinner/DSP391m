"""End-to-end API flow tests with lightweight fake processors."""

import base64
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import enrollment, exam, health
from src.api.websocket import audio_handler
from src.core.config import settings
from src.processing import pipeline as pipeline_module
from src.processing.pipeline import AudioPipeline
from src.storage.transcript_store import TranscriptStore


class _FakeVAD:
    def process_chunk(self, audio: np.ndarray) -> bool:
        return True


class _FakeSTT:
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> dict:
        return {"text": "tra loi lien quan de thi", "confidence": 0.96}


class _FakeEmbedding:
    def embed(self, text: str) -> np.ndarray:
        return np.array([1.0], dtype=np.float32)

    def similarity_to_question(self, text: str, question_embedding: np.ndarray) -> float:
        return 0.92


class _FakeSLM:
    def predict(self, exam_question: str, text: str) -> bool:
        return True


class _FakeVerifier:
    def __init__(self, enrollment_dir: Path):
        self.enrollment_dir = enrollment_dir
        self.enrollment_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, student_id: str) -> Path:
        return self.enrollment_dir / f"{student_id}.npy"

    def enroll(self, student_id: str, audio_samples: list[np.ndarray], sample_rate: int) -> bool:
        if not audio_samples:
            return False
        np.save(self._path(student_id), np.array([1.0], dtype=np.float32))
        return True

    def is_enrolled(self, student_id: str) -> bool:
        return self._path(student_id).exists()

    def verify(self, student_id: str, audio: np.ndarray, sample_rate: int) -> tuple[bool, float]:
        return True, 0.99

    def delete_enrollment(self, student_id: str) -> bool:
        path = self._path(student_id)
        if not path.exists():
            return False
        path.unlink()
        return True


class _FakeVerifierAlwaysFail(_FakeVerifier):
    def verify(self, student_id: str, audio: np.ndarray, sample_rate: int) -> tuple[bool, float]:
        return False, 0.2


def _build_test_app() -> FastAPI:
    app = FastAPI()
    app.include_router(health.router)
    app.include_router(exam.router)
    app.include_router(enrollment.router)
    app.include_router(audio_handler.router)
    return app


def test_e2e_enroll_start_ws_stop_report_history(monkeypatch, tmp_path):
    storage_root = tmp_path / "storage"
    monkeypatch.setattr(settings, "storage_root", storage_root)
    monkeypatch.setattr(settings, "speaker_verification_enabled", False)
    monkeypatch.setattr(settings, "diarization_enabled", False)
    monkeypatch.setattr(settings, "event_schema_version", "1.0.0")
    monkeypatch.setattr(settings, "report_schema_version", "1.0.0")
    settings.ensure_directories()

    verifier = _FakeVerifier(settings.enrollment_dir)
    store = TranscriptStore()
    pipeline = AudioPipeline(
        vad_processor=_FakeVAD(),
        stt_processor=_FakeSTT(),
        embedding_processor=_FakeEmbedding(),
        transcript_store=store,
        slm_processor=_FakeSLM(),
        speaker_verifier=verifier,
        overlap_detector=None,
    )
    monkeypatch.setattr(pipeline_module, "pipeline", pipeline)

    app = _build_test_app()
    client = TestClient(app)

    student_id = "student_e2e"
    audio_i16 = (np.zeros(16000, dtype=np.float32) * 32767).astype(np.int16)
    sample_b64 = base64.b64encode(audio_i16.tobytes()).decode("utf-8")

    enroll_resp = client.post(
        f"/api/enroll/{student_id}",
        json={"audio_samples": [sample_b64], "sample_rate": 16000},
    )
    assert enroll_resp.status_code == 201

    start_resp = client.post(
        "/api/exam/start",
        json={
            "student_id": student_id,
            "exam_id": "exam_e2e",
            "exam_question": "Noi dung de thi la gi?",
            "duration_minutes": 60,
        },
    )
    assert start_resp.status_code == 200
    session_id = start_resp.json()["session_id"]

    events_schema = None
    collected_events = []
    with client.websocket_connect(f"/ws/audio/{session_id}") as ws:
        for i in range(1, 7):
            ws.send_json(
                {
                    "type": "audio_chunk",
                    "data": sample_b64,
                    "timestamp": float(i),
                    "sample_rate": 16000,
                    "channels": 1,
                }
            )
            ack = ws.receive_json()
            assert ack["type"] in {"ack", "status_update"}
            if ack["type"] == "status_update":
                ack = ws.receive_json()
                assert ack["type"] == "ack"
            time.sleep(0.12)
        for _ in range(10):
            events_resp = client.get(f"/api/exam/events/{session_id}")
            assert events_resp.status_code == 200
            payload = events_resp.json()
            events_schema = payload.get("schema_version")
            collected_events.extend(payload.get("events", []))
            if any(e.get("type") == "transcript_log" for e in collected_events):
                break
            time.sleep(0.2)

    assert events_schema == "1.0.0"
    assert any(e.get("type") == "transcript_log" for e in collected_events)
    for event in collected_events:
        assert event.get("schema_version") == "1.0.0"

    stop_resp = client.post(f"/api/exam/stop/{session_id}")
    assert stop_resp.status_code == 200

    report_resp = client.get(f"/api/exam/report/{session_id}")
    assert report_resp.status_code == 200
    report = report_resp.json()
    assert report["schema_version"] == "1.0.0"
    assert report["session_id"] == session_id
    assert report["status"] in {"completed", "cancelled"}
    assert report["total_segments"] >= 1
    assert report["cheating_detected"] is True
    assert report["transcript"][0]["source"] in {"stt_only", "diarization"}

    history_resp = client.get(f"/api/exam/history?student_id={student_id}&limit=10")
    assert history_resp.status_code == 200
    history = history_resp.json()
    assert history["schema_version"] == "1.0.0"
    assert history["total"] >= 1
    assert any(item["session_id"] == session_id for item in history["items"])

    delete_resp = client.delete(f"/api/exam/session/{session_id}")
    assert delete_resp.status_code == 204
    unenroll_resp = client.delete(f"/api/enroll/{student_id}")
    assert unenroll_resp.status_code == 204


def test_e2e_policy_verify_fail_3_terminates_session(monkeypatch, tmp_path):
    storage_root = tmp_path / "storage"
    monkeypatch.setattr(settings, "storage_root", storage_root)
    monkeypatch.setattr(settings, "speaker_verification_enabled", True)
    monkeypatch.setattr(settings, "diarization_enabled", False)
    monkeypatch.setattr(settings, "verification_interval", 0)
    monkeypatch.setattr(settings, "verification_max_failures", 3)
    monkeypatch.setattr(settings, "min_verification_audio_seconds", 0.1)
    monkeypatch.setattr(settings, "event_schema_version", "1.0.0")
    monkeypatch.setattr(settings, "report_schema_version", "1.0.0")
    settings.ensure_directories()

    verifier = _FakeVerifierAlwaysFail(settings.enrollment_dir)
    store = TranscriptStore()
    pipeline = AudioPipeline(
        vad_processor=_FakeVAD(),
        stt_processor=_FakeSTT(),
        embedding_processor=_FakeEmbedding(),
        transcript_store=store,
        slm_processor=_FakeSLM(),
        speaker_verifier=verifier,
        overlap_detector=None,
    )
    monkeypatch.setattr(pipeline_module, "pipeline", pipeline)

    app = _build_test_app()
    client = TestClient(app)
    student_id = "student_fail_policy"

    audio_i16 = (np.ones(16000, dtype=np.float32) * 0.05 * 32767).astype(np.int16)
    sample_b64 = base64.b64encode(audio_i16.tobytes()).decode("utf-8")

    enroll_resp = client.post(
        f"/api/enroll/{student_id}",
        json={"audio_samples": [sample_b64], "sample_rate": 16000},
    )
    assert enroll_resp.status_code == 201

    start_resp = client.post(
        "/api/exam/start",
        json={
            "student_id": student_id,
            "exam_id": "exam_policy_fail",
            "exam_question": "Noi dung de thi",
            "duration_minutes": 60,
        },
    )
    assert start_resp.status_code == 200
    session_id = start_resp.json()["session_id"]

    with client.websocket_connect(f"/ws/audio/{session_id}") as ws:
        for i in range(1, 4):
            ws.send_json(
                {
                    "type": "audio_chunk",
                    "data": sample_b64,
                    "timestamp": float(i),
                    "sample_rate": 16000,
                    "channels": 1,
                }
            )
            ws.receive_json()
            time.sleep(0.05)

        terminated_event = None
        for _ in range(15):
            events_resp = client.get(f"/api/exam/events/{session_id}")
            assert events_resp.status_code == 200
            payload = events_resp.json()
            for ev in payload.get("events", []):
                if ev.get("type") == "session_terminated":
                    terminated_event = ev
                    break
            if terminated_event is not None:
                break
            time.sleep(0.1)

    assert terminated_event is not None
    assert terminated_event.get("failures_count") == 3
    assert terminated_event.get("schema_version") == "1.0.0"

    status_resp = client.get(f"/api/exam/status/{session_id}")
    assert status_resp.status_code == 200
    status_payload = status_resp.json()
    assert status_payload["status"] == "cancelled"
    assert status_payload["cheating_flag"] is True
