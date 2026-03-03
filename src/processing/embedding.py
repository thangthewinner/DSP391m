"""Text embedding using Vietnamese SBERT for semantic similarity detection."""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingProcessor:
    """Compute text embeddings and cosine similarity using Vietnamese SBERT."""

    MODEL_NAME = "keepitreal/vietnamese-sbert"

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None

    def load_model(self) -> None:
        """Load Vietnamese SBERT model."""
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading Vietnamese SBERT: {self.MODEL_NAME}")
        self.model = SentenceTransformer(self.MODEL_NAME, device=self.device)
        logger.info("✓ Vietnamese SBERT loaded successfully")

    def embed(self, text: str) -> np.ndarray:
        """
        Compute embedding for a single text.

        Returns:
            1D numpy array of shape (768,)
        """
        if self.model is None:
            raise RuntimeError("Embedding model not loaded. Call load_model() first.")

        if not text or not text.strip():
            return np.zeros(768, dtype=np.float32)

        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.astype(np.float32)

    def similarity(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """
        Cosine similarity between two normalized embeddings.

        Since embeddings are L2-normalized, dot product = cosine similarity.

        Returns:
            float in [0, 1]
        """
        score = float(np.dot(embedding_a, embedding_b))
        # Clamp to [0, 1] — normalized embeddings can give tiny negatives due to float precision
        return max(0.0, min(1.0, score))

    def similarity_to_question(self, transcript: str, question_embedding: np.ndarray) -> float:
        """
        Compute similarity between transcript text and pre-computed question embedding.

        Args:
            transcript: Transcribed speech text
            question_embedding: Pre-computed embedding of exam question

        Returns:
            Similarity score in [0, 1]
        """
        if not transcript or not transcript.strip():
            return 0.0

        transcript_embedding = self.embed(transcript)
        return self.similarity(transcript_embedding, question_embedding)
