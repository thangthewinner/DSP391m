"""Tests for audio buffer."""

import numpy as np
import pytest

from src.processing.buffer import AudioBuffer, SpeechBuffer


@pytest.fixture
def audio_buffer():
    """Create audio buffer instance."""
    return AudioBuffer(buffer_duration=10.0, sample_rate=16000)


def test_buffer_initialization(audio_buffer):
    """Test buffer initialization."""
    assert audio_buffer.buffer_duration == 10.0
    assert audio_buffer.sample_rate == 16000
    assert audio_buffer.max_samples == 160000
    assert len(audio_buffer) == 0


def test_buffer_add_chunk(audio_buffer):
    """Test adding audio chunk to buffer."""
    chunk = np.random.randn(16000).astype(np.float32)
    audio_buffer.add_chunk(chunk, timestamp=0.0)

    assert len(audio_buffer) == 16000
    assert audio_buffer.duration == 1.0


def test_buffer_capacity(audio_buffer):
    """Test buffer capacity limit."""
    # Add 15 seconds of audio (exceeds 10s capacity)
    for i in range(15):
        chunk = np.random.randn(16000).astype(np.float32)
        audio_buffer.add_chunk(chunk, timestamp=float(i))

    # Should only keep last 10 seconds
    assert len(audio_buffer) == 160000
    assert audio_buffer.is_full


def test_buffer_get_array(audio_buffer):
    """Test getting buffer as array."""
    chunk = np.random.randn(16000).astype(np.float32)
    audio_buffer.add_chunk(chunk, timestamp=0.0)

    array = audio_buffer.get_buffer_array()
    assert isinstance(array, np.ndarray)
    assert len(array) == 16000


def test_buffer_get_last_n_seconds(audio_buffer):
    """Test getting last N seconds."""
    # Add 5 seconds
    for i in range(5):
        chunk = np.random.randn(16000).astype(np.float32)
        audio_buffer.add_chunk(chunk, timestamp=float(i))

    # Get last 2 seconds
    last_2s = audio_buffer.get_last_n_seconds(2.0)
    assert len(last_2s) == 32000


def test_buffer_is_ready(audio_buffer):
    """Test buffer ready check."""
    assert not audio_buffer.is_ready(min_duration=5.0)

    # Add 6 seconds
    for i in range(6):
        chunk = np.random.randn(16000).astype(np.float32)
        audio_buffer.add_chunk(chunk, timestamp=float(i))

    assert audio_buffer.is_ready(min_duration=5.0)


def test_buffer_clear(audio_buffer):
    """Test buffer clearing."""
    chunk = np.random.randn(16000).astype(np.float32)
    audio_buffer.add_chunk(chunk, timestamp=0.0)

    assert len(audio_buffer) > 0

    audio_buffer.clear()
    assert len(audio_buffer) == 0


def test_speech_buffer():
    """Test speech buffer with VAD filtering."""
    speech_buffer = SpeechBuffer(buffer_duration=10.0)

    # Add speech chunk
    chunk1 = np.random.randn(16000).astype(np.float32)
    added1 = speech_buffer.add_chunk_if_speech(chunk1, 0.0, is_speech=True)
    assert added1
    assert len(speech_buffer) == 16000

    # Add non-speech chunk
    chunk2 = np.random.randn(16000).astype(np.float32)
    added2 = speech_buffer.add_chunk_if_speech(chunk2, 1.0, is_speech=False)
    assert not added2
    assert len(speech_buffer) == 16000  # Should not change

    # Check speech ratio
    assert speech_buffer.speech_ratio == 0.5  # 1 speech out of 2 total
