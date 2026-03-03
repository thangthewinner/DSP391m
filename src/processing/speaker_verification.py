"""
Speaker Verification — Phase 5.

Uses SpeechBrain ECAPA-TDNN to verify speaker identity during exam.

Flow:
    Enrollment: 3-5 audio samples → average embedding → save to disk
    Verification (every N minutes): current audio → embedding → cosine similarity
    similarity < threshold → verification_failed → +3pt Decision Engine
"""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_THRESHOLD = 0.75
MIN_AUDIO_SECONDS = 3.0


class SpeakerVerifier:
    """
    Speaker verification using SpeechBrain ECAPA-TDNN.

    Enrollment: compute average embedding from multiple samples, save to disk.
    Verification: compare current audio embedding against stored enrollment.
    """

    def __init__(
        self,
        enrollment_dir: Path,
        threshold: float = DEFAULT_THRESHOLD,
        device: str = "cpu",
    ):
        self.enrollment_dir = enrollment_dir
        self.threshold = threshold
        self.device = device
        self.model: Any = None
        self._classifier: Any = None

    def load_model(self) -> None:
        """Load ECAPA-TDNN model from SpeechBrain."""
        import torchaudio

        # Compatibility patch: torchaudio >= 2.x removed list_audio_backends
        # SpeechBrain 1.x still calls it — provide a no-op stub
        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: []  # type: ignore[attr-defined]
            logger.debug("Patched torchaudio.list_audio_backends for SpeechBrain compatibility")

        try:
            # speechbrain >= 1.0
            from speechbrain.inference.classifiers import EncoderClassifier  # type: ignore[import-untyped]
        except ImportError:
            # speechbrain 0.5.x
            from speechbrain.pretrained import EncoderClassifier  # type: ignore[import-untyped]

        logger.info("Loading SpeechBrain ECAPA-TDNN: %s", MODEL_NAME)

        self._classifier = EncoderClassifier.from_hparams(
            source=MODEL_NAME,
            run_opts={"device": self.device},
            savedir=str(self.enrollment_dir.parent / ".speechbrain_cache"),
        )
        self.model = self._classifier

        logger.info("✓ SpeechBrain ECAPA-TDNN loaded successfully")

    def extract_embedding(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Extract 192-dim speaker embedding from audio.

        Args:
            audio: Float32 numpy array, shape (N,), values in [-1, 1]
            sample_rate: Audio sample rate (must be 16000)

        Returns:
            192-dim numpy array
        """
        import torch

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if len(audio) < int(MIN_AUDIO_SECONDS * sample_rate):
            raise ValueError(
                f"Audio too short: {len(audio)/sample_rate:.1f}s < {MIN_AUDIO_SECONDS}s required"
            )

        # SpeechBrain expects (batch, time) tensor
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        audio_len = torch.tensor([1.0])  # relative length (full)

        with torch.no_grad():
            embedding = self._classifier.encode_batch(audio_tensor, audio_len)

        # Shape: (1, 1, 192) → (192,)
        return embedding.squeeze().cpu().numpy().astype(np.float32)

    def enroll(
        self,
        student_id: str,
        audio_samples: list[np.ndarray],
        sample_rate: int = 16000,
    ) -> bool:
        """
        Enroll a student by computing and saving average speaker embedding.

        Args:
            student_id: Student identifier
            audio_samples: List of audio arrays (3-5 samples recommended)
            sample_rate: Audio sample rate

        Returns:
            True if enrollment succeeded
        """
        if not audio_samples:
            logger.error("No audio samples provided for enrollment")
            return False

        embeddings = []
        for i, audio in enumerate(audio_samples):
            try:
                emb = self.extract_embedding(audio, sample_rate)
                embeddings.append(emb)
                logger.debug("Enrollment sample %d/%d extracted", i + 1, len(audio_samples))
            except ValueError as e:
                logger.warning("Skipping enrollment sample %d: %s", i + 1, e)

        if not embeddings:
            logger.error("All enrollment samples failed — audio too short?")
            return False

        # Average embedding (L2-normalized)
        avg_embedding = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm

        # Save to disk
        self.enrollment_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.enrollment_dir / f"{student_id}.npy"
        np.save(str(save_path), avg_embedding)

        logger.info(
            "✓ Enrolled student %s (%d samples) → %s",
            student_id,
            len(embeddings),
            save_path,
        )
        return True

    def verify(
        self,
        student_id: str,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> tuple[bool, float]:
        """
        Verify speaker identity against stored enrollment.

        Args:
            student_id: Student identifier
            audio: Current audio to verify
            sample_rate: Audio sample rate

        Returns:
            (passed, similarity_score)
            passed = True if similarity >= threshold
        """
        enrollment = self.load_enrollment(student_id)
        if enrollment is None:
            logger.warning("No enrollment found for student %s — skipping verification", student_id)
            return True, 1.0  # Pass by default if not enrolled

        try:
            current_embedding = self.extract_embedding(audio, sample_rate)
        except ValueError as e:
            logger.warning("Verification skipped for %s: %s", student_id, e)
            return True, 1.0  # Pass if audio too short

        # Cosine similarity (both embeddings are L2-normalized)
        similarity = float(np.dot(enrollment, current_embedding))
        similarity = max(0.0, min(1.0, similarity))

        passed = similarity >= self.threshold

        logger.info(
            "[%s] Speaker verification: similarity=%.3f, threshold=%.2f → %s",
            student_id,
            similarity,
            self.threshold,
            "PASS ✓" if passed else "FAIL ⚠️",
        )

        return passed, similarity

    def load_enrollment(self, student_id: str) -> Optional[np.ndarray]:
        """Load stored enrollment embedding for a student."""
        path = self.enrollment_dir / f"{student_id}.npy"
        if not path.exists():
            return None
        return np.load(str(path)).astype(np.float32)

    def is_enrolled(self, student_id: str) -> bool:
        """Check if a student has an enrollment."""
        return (self.enrollment_dir / f"{student_id}.npy").exists()

    def delete_enrollment(self, student_id: str) -> bool:
        """Delete enrollment for a student."""
        path = self.enrollment_dir / f"{student_id}.npy"
        if path.exists():
            path.unlink()
            logger.info("Deleted enrollment for student %s", student_id)
            return True
        return False
