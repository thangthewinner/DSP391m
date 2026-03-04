"""Voice Activity Detection using Silero VAD."""

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class VADProcessor:
    """Voice Activity Detection processor using Silero VAD."""

    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000):
        """
        Initialize VAD processor.

        Args:
            threshold: VAD threshold (0-1). Higher = more strict.
            sample_rate: Audio sample rate in Hz.
        """
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.model: Optional[torch.nn.Module] = None
        self.utils: Optional[tuple] = None

    def load_model(self) -> None:
        """Load Silero VAD model from torch hub."""
        try:
            logger.info("Loading Silero VAD model...")
            self.model, self.utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
                trust_repo=True,  # Trust the repo to avoid confirmation
            )
            self.model.eval()
            logger.info("Silero VAD model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD model: {e}")
            logger.info("Retrying with force_reload=True...")
            try:
                self.model, self.utils = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=True,
                    onnx=False,
                    trust_repo=True,
                )
                self.model.eval()
                logger.info("Silero VAD model loaded successfully on retry")
            except Exception as e2:
                logger.error(f"Failed to load Silero VAD model on retry: {e2}")
                raise

    def process_chunk(self, audio: np.ndarray) -> bool:
        """
        Process audio chunk and detect speech.

        Silero VAD requires exactly 512 samples per frame at 16kHz.
        We split the chunk into frames and return True if any frame has speech.

        Args:
            audio: Audio array (mono, sample_rate Hz)

        Returns:
            True if speech detected, False otherwise
        """
        if self.model is None:
            raise RuntimeError("VAD model not loaded. Call load_model() first.")

        # Silero VAD v4 requires exactly 512 samples at 16kHz (or 256 at 8kHz)
        frame_size = 512 if self.sample_rate == 16000 else 256

        try:
            n = len(audio)
            if n < frame_size:
                # Too short — pad and process as single frame
                padded = np.zeros(frame_size, dtype=np.float32)
                padded[:n] = audio
                frames = [padded]
            else:
                # Split into non-overlapping frames; drop the last incomplete frame
                num_frames = n // frame_size
                frames = [audio[i * frame_size:(i + 1) * frame_size] for i in range(num_frames)]

            max_prob = 0.0
            with torch.no_grad():
                for frame in frames:
                    tensor = torch.from_numpy(frame).float()
                    prob = self.model(tensor, self.sample_rate).item()
                    if prob > max_prob:
                        max_prob = prob

            is_speech = max_prob >= self.threshold
            logger.debug(
                "VAD: max_prob=%.3f over %d frames, threshold=%.2f, is_speech=%s",
                max_prob, len(frames), self.threshold, is_speech,
            )
            return is_speech

        except Exception as e:
            logger.error(f"Error processing audio chunk in VAD: {e}")
            # Fallback: assume speech to avoid dropping audio
            return True

