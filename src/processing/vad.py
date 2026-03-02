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

        Args:
            audio: Audio array (mono, sample_rate Hz)

        Returns:
            True if speech detected, False otherwise
        """
        if self.model is None:
            raise RuntimeError("VAD model not loaded. Call load_model() first.")

        try:
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio).float()

            # Get speech probability
            with torch.no_grad():
                speech_prob = self.model(audio_tensor, self.sample_rate).item()

            # Check against threshold
            is_speech = speech_prob >= self.threshold

            logger.debug(
                f"VAD: speech_prob={speech_prob:.3f}, threshold={self.threshold}, "
                f"is_speech={is_speech}"
            )

            return is_speech

        except Exception as e:
            logger.error(f"Error processing audio chunk in VAD: {e}")
            # Fallback: assume speech to avoid dropping audio
            return True

