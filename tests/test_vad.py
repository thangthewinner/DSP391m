"""Tests for VAD processor."""

import numpy as np
import pytest

from src.processing.vad import VADProcessor


@pytest.fixture
def vad_processor():
    """Create VAD processor instance."""
    return VADProcessor(threshold=0.5, sample_rate=16000)


def test_vad_initialization(vad_processor):
    """Test VAD processor initialization."""
    assert vad_processor.threshold == 0.5
    assert vad_processor.sample_rate == 16000
    assert vad_processor.model is None


def test_vad_process_chunk_without_model(vad_processor):
    """Test processing chunk without loaded model raises error."""
    audio = np.random.randn(16000).astype(np.float32)

    with pytest.raises(RuntimeError, match="VAD model not loaded"):
        vad_processor.process_chunk(audio)


def test_vad_load_model(vad_processor):
    """Test VAD model loading."""
    vad_processor.load_model()
    assert vad_processor.model is not None


@pytest.mark.skipif(
    True,
    reason="Requires model download - run manually with: pytest tests/test_vad.py::test_vad_speech_detection -v",
)
def test_vad_speech_detection():
    """Test VAD speech detection with actual model."""
    vad = VADProcessor(threshold=0.5)
    vad.load_model()

    # Generate silent audio
    silent_audio = np.zeros(16000, dtype=np.float32)
    is_speech_silent = vad.process_chunk(silent_audio)

    # Generate noisy audio (simulating speech)
    noisy_audio = np.random.randn(16000).astype(np.float32) * 0.5
    is_speech_noisy = vad.process_chunk(noisy_audio)

    # Silent audio should not be detected as speech
    assert not is_speech_silent

    # Note: Random noise might not be detected as speech either
    # This is just a basic test structure
