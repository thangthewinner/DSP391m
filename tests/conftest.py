"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture(scope="session")
def sample_audio_16khz():
    """Generate sample audio at 16kHz."""
    import numpy as np

    # 1 second of audio
    return np.random.randn(16000).astype(np.float32) * 0.1


@pytest.fixture(scope="session")
def silent_audio_16khz():
    """Generate silent audio at 16kHz."""
    import numpy as np

    # 1 second of silence
    return np.zeros(16000, dtype=np.float32)
