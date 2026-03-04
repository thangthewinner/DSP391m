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
    ) -> tuple[bool, float, list[dict]]:
        """
        Detect if multiple speakers are present in the audio.

        Args:
            audio: Float32 numpy array, shape (N,), values in [-1, 1]
            sample_rate: Audio sample rate (must be 16000)

        Returns:
            (overlap_detected, confidence, segments)
            overlap_detected = True if 2+ speakers found
            confidence = 0.0-1.0 (normalized speaker count)
            segments = list of {speaker, start, end} dicts
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
            return False, 0.0, []

        try:
            audio_duration = len(audio) / sample_rate
            logger.debug(
                "Running diarization on %.1fs audio buffer (%d samples)",
                audio_duration,
                len(audio),
            )

            predicted_segments = self.model.diarize(
                audio=[audio],
                batch_size=1,
                sample_rate=sample_rate,
            )

            if not predicted_segments or not predicted_segments[0]:
                logger.debug("Diarization returned no segments")
                return False, 0.0, []

            # Parse segments and collect unique speakers
            unique_speakers: set[str] = set()
            parsed_segments: list[dict] = []

            for seg in predicted_segments[0]:
                if isinstance(seg, dict):
                    speaker = str(seg.get("speaker", seg.get("label", "")))
                    try:
                        start = float(seg.get("start", seg.get("start_time", 0.0)))
                        end = float(seg.get("end", seg.get("end_time", 0.0)))
                    except (TypeError, ValueError):
                        start, end = 0.0, 0.0
                elif isinstance(seg, (list, tuple)) and len(seg) >= 3:
                    # (start, end, speaker) or (speaker, start, end)
                    try:
                        start = float(seg[0])
                        end = float(seg[1])
                        speaker = str(seg[2])
                    except (TypeError, ValueError):
                        try:
                            speaker = str(seg[0])
                            start = float(seg[1])
                            end = float(seg[2])
                        except (TypeError, ValueError):
                            continue
                else:
                    # String format: "speaker_0 0.00 5.00" or "0.00 5.00 speaker_0"
                    parts = str(seg).split() if seg else []
                    if len(parts) < 3:
                        continue
                    # Try "speaker start end" format first
                    try:
                        float(parts[0])  # if first token is a number → "start end speaker"
                        start = float(parts[0])
                        end = float(parts[1])
                        speaker = parts[2]
                    except ValueError:
                        # First token is speaker name → "speaker start end"
                        speaker = parts[0]
                        try:
                            start = float(parts[1])
                            end = float(parts[2])
                        except (ValueError, IndexError):
                            start, end = 0.0, 0.0

                if speaker:
                    unique_speakers.add(speaker)
                    parsed_segments.append({"speaker": speaker, "start": start, "end": end})

            num_speakers = len(unique_speakers)
            overlap_detected = num_speakers >= 2
            confidence = min(1.0, (num_speakers - 1) / 3.0) if num_speakers >= 2 else 0.0

            # Detailed segment log
            logger.info(
                "[Diarization] %.1fs audio → %d speaker(s) detected: %s",
                audio_duration,
                num_speakers,
                ", ".join(sorted(unique_speakers)) if unique_speakers else "none",
            )
            for seg in parsed_segments:
                logger.debug(
                    "  [%s] %.2fs → %.2fs (%.1fs)",
                    seg["speaker"],
                    seg["start"],
                    seg["end"],
                    seg["end"] - seg["start"],
                )

            if overlap_detected:
                logger.warning(
                    "[Diarization] OVERLAP DETECTED — %d speakers, conf=%.2f",
                    num_speakers,
                    confidence,
                )
            else:
                logger.info(
                    "[Diarization] Single speaker — no overlap (conf=0.00)"
                )

            return overlap_detected, confidence, parsed_segments

        except Exception as e:  # noqa: BLE001  # NeMo can raise various runtime errors
            logger.warning("Diarization failed, defaulting to no overlap: %s", e)
            return False, 0.0, []
