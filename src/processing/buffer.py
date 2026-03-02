"""Audio buffering with sliding window."""

import logging
from collections import deque
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class AudioBuffer:
    """Audio buffer with sliding window for speech processing."""

    def __init__(self, buffer_duration: float = 10.0, sample_rate: int = 16000):
        """
        Initialize audio buffer.

        Args:
            buffer_duration: Buffer duration in seconds
            sample_rate: Audio sample rate in Hz
        """
        self.buffer_duration = buffer_duration
        self.sample_rate = sample_rate
        self.max_samples = int(buffer_duration * sample_rate)

        # Use deque with maxlen for automatic size management
        self.buffer: deque = deque(maxlen=self.max_samples)
        self.timestamps: deque = deque()

        # Cache for optimization
        self._cached_array: Optional[np.ndarray] = None
        self._cache_valid = False

        logger.info(
            f"AudioBuffer initialized: duration={buffer_duration}s, "
            f"sample_rate={sample_rate}Hz, max_samples={self.max_samples}"
        )

    def add_chunk(self, audio_chunk: np.ndarray, timestamp: float) -> None:
        """
        Add audio chunk to buffer.

        Args:
            audio_chunk: Audio samples to add
            timestamp: Timestamp of the chunk
        """
        # Add samples to buffer
        for sample in audio_chunk:
            self.buffer.append(sample)

        # Track timestamp
        self.timestamps.append((timestamp, len(audio_chunk)))

        # Remove old timestamps that are no longer in buffer
        total_samples = sum(count for _, count in self.timestamps)
        while total_samples > self.max_samples and len(self.timestamps) > 1:
            _, count = self.timestamps.popleft()
            total_samples -= count

        # Invalidate cache
        self._cache_valid = False

        logger.debug(
            f"Added {len(audio_chunk)} samples at t={timestamp:.2f}s, "
            f"buffer size: {len(self.buffer)}/{self.max_samples}"
        )

    def get_buffer_array(self) -> np.ndarray:
        """
        Get current buffer as numpy array.

        Returns:
            Numpy array of audio samples
        """
        if not self._cache_valid:
            self._cached_array = np.array(list(self.buffer), dtype=np.float32)
            self._cache_valid = True
        return self._cached_array

    def get_last_n_seconds(self, n_seconds: float) -> np.ndarray:
        """
        Get last N seconds from buffer.

        Args:
            n_seconds: Number of seconds to retrieve

        Returns:
            Numpy array of audio samples
        """
        n_samples = int(n_seconds * self.sample_rate)
        if len(self.buffer) < n_samples:
            return self.get_buffer_array()

        # Get last n_samples
        return np.array(list(self.buffer)[-n_samples:], dtype=np.float32)

    def is_ready(self, min_duration: float = 5.0) -> bool:
        """
        Check if buffer has enough data for processing.

        Args:
            min_duration: Minimum duration in seconds

        Returns:
            True if buffer has enough data
        """
        min_samples = int(min_duration * self.sample_rate)
        return len(self.buffer) >= min_samples

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
        self.timestamps.clear()
        self._cache_valid = False
        logger.debug("Buffer cleared")

    @property
    def duration(self) -> float:
        """Get current buffer duration in seconds."""
        return len(self.buffer) / self.sample_rate

    @property
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return len(self.buffer) >= self.max_samples

    def __len__(self) -> int:
        """Get number of samples in buffer."""
        return len(self.buffer)


class SpeechBuffer(AudioBuffer):
    """Audio buffer that only stores speech segments (VAD-filtered)."""

    def __init__(
        self,
        buffer_duration: float = 10.0,
        sample_rate: int = 16000,
        vad_threshold: float = 0.5,
    ):
        """
        Initialize speech buffer.

        Args:
            buffer_duration: Buffer duration in seconds
            sample_rate: Audio sample rate in Hz
            vad_threshold: VAD threshold for speech detection
        """
        super().__init__(buffer_duration, sample_rate)
        self.vad_threshold = vad_threshold
        self.total_chunks_received = 0
        self.speech_chunks_kept = 0

    def add_chunk_if_speech(
        self, audio_chunk: np.ndarray, timestamp: float, is_speech: bool
    ) -> bool:
        """
        Add chunk only if it contains speech.

        Args:
            audio_chunk: Audio samples
            timestamp: Timestamp of the chunk
            is_speech: Whether VAD detected speech

        Returns:
            True if chunk was added, False otherwise
        """
        self.total_chunks_received += 1

        if is_speech:
            self.add_chunk(audio_chunk, timestamp)
            self.speech_chunks_kept += 1
            return True

        return False

    @property
    def speech_ratio(self) -> float:
        """Get ratio of speech chunks to total chunks."""
        if self.total_chunks_received == 0:
            return 0.0
        return self.speech_chunks_kept / self.total_chunks_received
