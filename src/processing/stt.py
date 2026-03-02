"""Speech-to-Text using PhoWhisper (faster-whisper backend)."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class STTProcessor:
    """Speech-to-Text processor using PhoWhisper with faster-whisper backend."""

    def __init__(
        self,
        model_name: str = "vinai/PhoWhisper-small",
        device: str = "cuda",
        compute_type: str = "float16",
    ):
        """
        Initialize STT processor.

        Args:
            model_name: Model name or path
            device: Device to use ('cuda' or 'cpu')
            compute_type: Compute type ('float16', 'int8', 'float32')
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.model: Optional[WhisperModel] = None

        # Adjust compute type for CPU
        if device == "cpu" and compute_type == "float16":
            self.compute_type = "int8"
            logger.info("Using int8 compute type for CPU")

    def load_model(self, model_path: Optional[Path] = None) -> None:
        """
        Load PhoWhisper model in CTranslate2 format.
        
        Args:
            model_path: Path to CTranslate2 model directory (must contain model.bin)
        """
        try:
            if model_path is None:
                model_source = self.model_name
                logger.info(f"Loading PhoWhisper from HuggingFace: {model_source}")
            else:
                model_source = str(model_path)
                logger.info(f"Loading PhoWhisper from local: {model_source}")
            
            logger.info(f"Device: {self.device}, Compute type: {self.compute_type}")

            # Load CTranslate2 model
            self.model = WhisperModel(
                model_source,
                device=self.device,
                compute_type=self.compute_type,
            )

            logger.info("✓ PhoWhisper model loaded successfully")

        except Exception as e:
            logger.error(f"✗ Failed to load PhoWhisper model: {e}")
            if model_path:
                logger.error(
                    "Troubleshooting:\n"
                    f"  1. Check if model.bin exists in {model_path}\n"
                    "  2. If not, run: python scripts/convert_phowhisper.py\n"
                    "  3. Or convert manually with ct2-transformers-converter"
                )
            else:
                logger.error(
                    "Troubleshooting:\n"
                    "  1. Check internet connection\n"
                    "  2. Ensure model is in CTranslate2 format"
                )
            raise

    def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text.

        Args:
            audio: Audio array (mono, float32)
            sample_rate: Audio sample rate

        Returns:
            Dictionary with 'text' and 'confidence'
        """
        if self.model is None:
            raise RuntimeError("STT model not loaded. Call load_model() first.")

        try:
            # Ensure audio is float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Normalize audio to [-1, 1] range if needed
            if np.abs(audio).max() > 1.0:
                audio = audio / np.abs(audio).max()

            # Transcribe
            segments, info = self.model.transcribe(
                audio,
                language="vi",  # Vietnamese
                beam_size=1,  # Greedy decoding for speed
                vad_filter=False,  # We already did VAD
                without_timestamps=False,
            )

            # Collect segments
            text_parts = []
            confidences = []

            for segment in segments:
                text_parts.append(segment.text.strip())
                # avg_logprob is negative, convert to confidence
                confidence = np.exp(segment.avg_logprob)
                confidences.append(confidence)

            # Combine results
            full_text = " ".join(text_parts)
            avg_confidence = np.mean(confidences) if confidences else 0.0

            result = {
                "text": full_text,
                "confidence": float(min(avg_confidence, 1.0)),
                "language": info.language,
                "language_probability": info.language_probability,
            }

            logger.debug(
                f"STT: transcribed {len(audio)/sample_rate:.2f}s audio, "
                f"text_length={len(full_text)}, confidence={avg_confidence:.3f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "error": str(e),
            }

    def transcribe_with_timestamps(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> list[Dict[str, Any]]:
        """
        Transcribe audio with word-level timestamps.

        Args:
            audio: Audio array (mono, float32)
            sample_rate: Audio sample rate

        Returns:
            List of segment dictionaries with timestamps
        """
        if self.model is None:
            raise RuntimeError("STT model not loaded. Call load_model() first.")

        try:
            # Ensure audio is float32 and normalized
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            if np.abs(audio).max() > 1.0:
                audio = audio / np.abs(audio).max()

            # Transcribe with timestamps
            segments, info = self.model.transcribe(
                audio,
                language="vi",
                beam_size=1,
                vad_filter=False,
                word_timestamps=True,
            )

            # Collect segments with timestamps
            results = []
            for segment in segments:
                results.append(
                    {
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text.strip(),
                        "confidence": float(np.exp(segment.avg_logprob)),
                    }
                )

            return results

        except Exception as e:
            logger.error(f"Error transcribing audio with timestamps: {e}")
            return []
