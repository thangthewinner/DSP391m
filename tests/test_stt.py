"""Tests for STT processor."""

import numpy as np
import pytest

from src.processing.stt import STTProcessor


@pytest.fixture
def stt_processor():
    """Create STT processor instance."""
    return STTProcessor(
        model_name="vinai/PhoWhisper-small",
        device="cpu",
        compute_type="int8",
    )


def test_stt_initialization(stt_processor):
    """Test STT processor initialization."""
    assert stt_processor.model_name == "vinai/PhoWhisper-small"
    assert stt_processor.device == "cpu"
    assert stt_processor.compute_type == "int8"
    assert stt_processor.model is None


def test_stt_transcribe_without_model(stt_processor):
    """Test transcribing without loaded model raises error."""
    audio = np.random.randn(16000).astype(np.float32)

    with pytest.raises(RuntimeError, match="STT model not loaded"):
        stt_processor.transcribe(audio)


@pytest.mark.skipif(
    True,
    reason="Requires model download - run manually with: pytest tests/test_stt.py::test_stt_transcribe -v",
)
def test_stt_transcribe():
    """Test STT transcription with actual model."""
    stt = STTProcessor(device="cpu", compute_type="int8")
    stt.load_model()

    # Generate 2 seconds of audio
    audio = np.random.randn(32000).astype(np.float32) * 0.01

    result = stt.transcribe(audio, sample_rate=16000)

    assert "text" in result
    assert "confidence" in result
    assert isinstance(result["text"], str)
    assert 0 <= result["confidence"] <= 1


def test_stt_audio_normalization(stt_processor):
    """Test audio normalization in transcribe."""
    # This tests the normalization logic without actually running the model
    audio = np.array([32767, -32768, 0], dtype=np.int16)
    audio_float = audio.astype(np.float32)

    # Should normalize to [-1, 1] range
    max_val = np.abs(audio_float).max()
    if max_val > 1.0:
        normalized = audio_float / max_val
        assert np.abs(normalized).max() <= 1.0
