"""
Overlap Detector — Phase 6.

Uses NVIDIA NeMo Streaming Sortformer to detect when multiple speakers
are talking simultaneously during an exam session.

Model: nvidia/diar_streaming_sortformer_4spk-v2
- Streaming-capable, chunk-based diarization
- Supports numpy array input
- Detects up to 4 speakers

Flow:
    rolling audio buffer (10-30s)
    → SortformerEncLabelModel.diarize(audio_np)
    → 2+ unique speakers in window → overlap_detected = True
    → Decision Engine +1pt
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Streaming Sortformer configuration (in 80ms frames)
CHUNK_LEN = 340           # ~27s context window
CHUNK_RIGHT_CONTEXT = 40  # ~3.2s right context
FIFO_LEN = 40             # ~3.2s FIFO buffer
SPKCACHE_UPDATE_PERIOD = 300  # speaker cache update period


class OverlapDetector:
    """
    Speaker overlap detection using NeMo Streaming Sortformer.

    Detects when 2+ speakers are present in the audio buffer.
    Runs on a rolling audio buffer (not per-chunk) for accuracy.
    Requires at least MIN_AUDIO_SECONDS of audio to produce reliable results.
    """

    MODEL_NAME = "nvidia/diar_streaming_sortformer_4spk-v2"

    def __init__(
        self,
        model_path: Path,
        device: str = "cpu",
        min_audio_seconds: float = 10.0,
    ):
        self.model_path = model_path
        self.device = device
        self.min_audio_seconds = min_audio_seconds
        self.model: Any = None

    def load_model(self) -> None:
        """Load NeMo Streaming Sortformer model."""
        from nemo.collections.asr.models import SortformerEncLabelModel  # type: ignore[import-untyped]

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Diarization model not found: {self.model_path}\n"
                f"Run: uv run python scripts/download_models.py --diarization"
            )

        logger.info("Loading NeMo Sortformer: %s", self.model_path.name)
        logger.info("  device=%s", self.device)

        self.model = SortformerEncLabelModel.restore_from(
            restore_path=str(self.model_path),
            map_location=self.device,
            strict=False,
        )
        self.model.eval()

        # Configure streaming parameters
        self.model.sortformer_modules.chunk_len = CHUNK_LEN
        self.model.sortformer_modules.chunk_right_context = CHUNK_RIGHT_CONTEXT
        self.model.sortformer_modules.fifo_len = FIFO_LEN
        self.model.sortformer_modules.spkcache_update_period = SPKCACHE_UPDATE_PERIOD

        logger.info("✓ NeMo Sortformer loaded successfully: %s", self.model_path.name)

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> tuple[bool, float]:
        """
        Detect if multiple speakers are present in the audio.

        Args:
            audio: Float32 numpy array, shape (N,), values in [-1, 1]
            sample_rate: Audio sample rate (must be 16000)

        Returns:
            (overlap_detected, confidence)
            overlap_detected = True if 2+ speakers found
            confidence = 0.0-1.0 (normalized speaker count)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        min_samples = int(self.min_audio_seconds * sample_rate)
        if len(audio) < min_samples:
            logger.debug(
                "Audio too short for diarization: %.1fs < %.1fs required",
                len(audio) / sample_rate,
                self.min_audio_seconds,
            )
            return False, 0.0

        try:
            predicted_segments = self.model.diarize(
                audio=[audio],
                batch_size=1,
                sample_rate=sample_rate,
            )

            if not predicted_segments or not predicted_segments[0]:
                return False, 0.0

            # Count unique speakers in the output
            unique_speakers: set[str] = set()
            for seg in predicted_segments[0]:
                if isinstance(seg, dict):
                    speaker = seg.get("speaker", seg.get("label", ""))
                else:
                    # Segment may be a string like "speaker_0 0.0 5.0"
                    speaker = str(seg).split()[0] if seg else ""
                if speaker:
                    unique_speakers.add(speaker)

            num_speakers = len(unique_speakers)
            overlap_detected = num_speakers >= 2

            # Confidence: 0 for 1 speaker, scales up with more speakers
            confidence = min(1.0, (num_speakers - 1) / 3.0) if num_speakers >= 2 else 0.0

            if overlap_detected:
                logger.info(
                    "Overlap detected: %d speakers (conf=%.2f)",
                    num_speakers,
                    confidence,
                )

            return overlap_detected, confidence

        except Exception as e:  # noqa: BLE001  # NeMo can raise various runtime errors
            logger.warning("Diarization failed, defaulting to no overlap: %s", e)
            return False, 0.0
