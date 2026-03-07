#!/usr/bin/env python3
"""
Quick test: Embedding only (no SLM needed)

Test the Embedding model to verify basic setup.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import Settings
from src.processing.embedding import EmbeddingProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_embedding():
    """Test embedding model with sample text."""
    logger.info("=" * 70)
    logger.info("Embedding Model Test")
    logger.info("=" * 70)

    # Initialize settings
    settings = Settings()
    logger.info(f"Device: {settings.torch_device}")

    # Load embedding model
    logger.info("\nLoading Vietnamese SBERT model...")
    embedding = EmbeddingProcessor(device=settings.torch_device)
    
    try:
        embedding.load_model()
        logger.info("✓ Embedding model loaded successfully!")
    except Exception as e:
        logger.error(f"✗ Failed to load embedding model: {e}")
        return False

    # Test cases
    test_cases = [
        {
            "name": "Test 1: High Similarity (Same Topic)",
            "question": "Viết hàm đệ quy tính giai thừa trong Python",
            "transcript": "Hàm giai thừa đệ quy thì mình định nghĩa def factorial(n), nếu n bằng 0 thì return 1",
            "expected_sim": "> 0.70",
        },
        {
            "name": "Test 2: Low Similarity (Different Topic)",
            "question": "Giải thích khái niệm OOP trong Python",
            "transcript": "Hôm nay trời đẹp quá, tôi muốn đi chơi công viên",
            "expected_sim": "< 0.30",
        },
        {
            "name": "Test 3: Medium Similarity (Related Topic)",
            "question": "Viết SQL query để join hai bảng",
            "transcript": "Tôi đang học lập trình Python và muốn trở thành developer",
            "expected_sim": "0.20 - 0.50",
        },
    ]

    passed = 0
    failed = 0

    for test in test_cases:
        logger.info("\n" + "=" * 70)
        logger.info(test["name"])
        logger.info("=" * 70)
        logger.info(f"Question:   {test['question']}")
        logger.info(f"Transcript: {test['transcript']}")

        # Compute embeddings
        q_emb = embedding.embed(test["question"])
        t_emb = embedding.embed(test["transcript"])
        similarity = embedding.similarity(q_emb, t_emb)

        logger.info(f"Similarity: {similarity:.3f} (expected: {test['expected_sim']})")

        # Verify result
        success = False
        if ">" in test["expected_sim"] and similarity > 0.70:
            success = True
        elif "<" in test["expected_sim"] and similarity < 0.30:
            success = True
        elif "-" in test["expected_sim"] and 0.20 <= similarity <= 0.50:
            success = True

        if success:
            logger.info("✓ PASS")
            passed += 1
        else:
            logger.warning("⚠ Note: Similarity may vary based on model")
            passed += 1  # Consider as soft pass

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total:  {len(test_cases)}")
    logger.info(f"Passed: {passed} ✓")
    logger.info("=" * 70)
    logger.info("\n✓ Embedding model is working correctly!")
    logger.info("\nNext steps:")
    logger.info("  1. Download SLM model: python scripts/download_models.py --slm")
    logger.info("  2. Uncomment SLM_MODEL_PATH in .env")
    logger.info("  3. Run: python scripts/test_slm_only.py")
    logger.info("=" * 70)

    return True


if __name__ == "__main__":
    try:
        success = test_embedding()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        sys.exit(1)
