"""Integration tests for end-to-end audio pipeline."""

import asyncio

import numpy as np
import pytest

from src.core.config import settings
from src.core.models import TranscriptSegment
from src.processing.buffer import AudioBuffer
from src.processing.pipeline import AudioPipeline
from src.processing.stt import STTProcessor
from src.processing.vad import VADProcessor
from src.storage.transcript_store import TranscriptStore


@pytest.mark.asyncio
@pytest.mark.skipif(
    True,
    reason="Requires models - run manually: pytest tests/test_integration.py -v",
)
async def test_vad_to_buffer_integration():
    """Test VAD + Buffer integration."""
    # Initialize components
    vad = VADProcessor(threshold=0.5)
    vad.load_model()

    buffer = AudioBuffer(buffer_duration=10.0)

    # Generate test audio
    speech_audio = np.random.randn(16000).astype(np.float32) * 0.5
    silent_audio = np.zeros(16000, dtype=np.float32)

    # Process speech
    is_speech = vad.process_chunk(speech_audio)
    if is_speech:
        buffer.add_chunk(speech_audio, timestamp=0.0)

    # Process silence
    is_silence = vad.process_chunk(silent_audio)
    if is_silence:
        buffer.add_chunk(silent_audio, timestamp=1.0)

    # Buffer should only contain speech
    assert len(buffer) > 0


@pytest.mark.asyncio
@pytest.mark.skipif(
    True,
    reason="Requires models - run manually: pytest tests/test_integration.py -v",
)
async def test_buffer_to_stt_integration():
    """Test Buffer + STT integration."""
    # Initialize components
    buffer = AudioBuffer(buffer_duration=10.0)
    stt = STTProcessor(device="cpu", compute_type="int8")
    stt.load_model()

    # Add 5 seconds of audio to buffer
    for i in range(5):
        chunk = np.random.randn(16000).astype(np.float32) * 0.01
        buffer.add_chunk(chunk, timestamp=float(i))

    # Check buffer is ready
    assert buffer.is_ready(min_duration=5.0)

    # Transcribe
    audio_array = buffer.get_buffer_array()
    result = stt.transcribe(audio_array)

    assert "text" in result
    assert "confidence" in result


@pytest.mark.asyncio
async def test_transcript_store_integration():
    """Test transcript storage integration."""
    store = TranscriptStore()

    # Create test segment
    segment = TranscriptSegment(
        timestamp_start=0.0,
        timestamp_end=2.0,
        text="Test transcript",
        confidence=0.95,
    )

    # Save segment
    session_id = "test_session_123"
    await store.save_segment(session_id, segment)

    # Load segments
    segments = await store.load_segments(session_id)
    assert len(segments) == 1
    assert segments[0].text == "Test transcript"

    # Get full transcript
    full_text = await store.get_full_transcript(session_id)
    assert full_text == "Test transcript"

    # Cleanup
    await store.delete_transcript(session_id)


@pytest.mark.asyncio
@pytest.mark.skipif(
    True,
    reason="Requires models - run manually: pytest tests/test_integration.py::test_full_pipeline -v",
)
async def test_full_pipeline():
    """Test full pipeline integration."""
    # Initialize all components
    vad = VADProcessor(threshold=0.5)
    vad.load_model()

    stt = STTProcessor(device="cpu", compute_type="int8")
    stt.load_model()

    transcript_store = TranscriptStore()

    pipeline = AudioPipeline(
        vad_processor=vad,
        stt_processor=stt,
        transcript_store=transcript_store,
    )

    # Test pipeline components are initialized
    assert pipeline.vad is not None
    assert pipeline.stt is not None
    assert pipeline.transcript_store is not None

    print("✓ Full pipeline initialized successfully")


def test_settings_configuration():
    """Test settings are properly configured."""
    assert settings.audio_sample_rate == 16000
    assert settings.buffer_size == 10.0
    assert settings.vad_threshold == 0.5
    assert settings.api_port == 8000


def test_directory_creation():
    """Test directory creation."""
    settings.ensure_directories()

    assert settings.model_cache_dir.exists()
    assert settings.storage_root.exists()
    assert settings.transcripts_dir.exists()
    assert settings.logs_dir.exists()
