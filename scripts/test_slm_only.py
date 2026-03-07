#!/usr/bin/env python3
"""
Quick test: Text-only SLM verification (no audio needed)

Test the Embedding + SLM pipeline without requiring audio or STT models.
Perfect for quick testing during development.

Usage:
    uv run python scripts/test_slm_only.py
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import Settings
from src.processing.embedding import EmbeddingProcessor
from src.processing.slm import SLMProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_cases() -> list[dict]:
    """Define test cases for cheating detection."""
    return [
        # ─────────────────────────────────────────────────────────────
        # Test Case 1: Clear cheating (discussing exact exam content)
        # ─────────────────────────────────────────────────────────────
        {
            "name": "Test 1: Clear Cheating",
            "question": "Viết hàm đệ quy tính giai thừa trong Python",
            "transcript": "Hàm giai thừa đệ quy thì mình định nghĩa def factorial(n), nếu n bằng 0 thì return 1, còn không thì return n nhân factorial(n-1)",
            "expected_cheating": True,
        },
        # ─────────────────────────────────────────────────────────────
        # Test Case 2: Not related (general conversation)
        # ─────────────────────────────────────────────────────────────
        {
            "name": "Test 2: Not Related",
            "question": "Giải thích khái niệm OOP trong Python",
            "transcript": "Hôm nay trời đẹp quá, tôi muốn đi chơi công viên với bạn bè",
            "expected_cheating": False,
        },
        # ─────────────────────────────────────────────────────────────
        # Test Case 3: Related topic but not exact answer
        # ─────────────────────────────────────────────────────────────
        {
            "name": "Test 3: Topic Related (Borderline)",
            "question": "Viết SQL query để join hai bảng",
            "transcript": "Tôi đang học lập trình Python và muốn trở thành developer",
            "expected_cheating": False,
        },
        # ─────────────────────────────────────────────────────────────
        # Test Case 4: Discussing exam question directly
        # ─────────────────────────────────────────────────────────────
        {
            "name": "Test 4: Discussing Exam",
            "question": "Giải thích các thuộc tính ACID trong database",
            "transcript": "ACID là gì vậy? À ACID là Atomicity, Consistency, Isolation, Durability. Atomicity nghĩa là transaction phải hoàn thành hoặc rollback hết",
            "expected_cheating": True,
        },
    ]


def run_test(
    test_case: dict,
    embedding: EmbeddingProcessor,
    slm: SLMProcessor,
    settings: Settings,
) -> bool:
    """
    Run a single test case.

    Returns:
        True if test passed
    """
    name = test_case["name"]
    question = test_case["question"]
    transcript = test_case["transcript"]
    expected = test_case["expected_cheating"]

    logger.info("\n" + "=" * 70)
    logger.info(f"{name}")
    logger.info("=" * 70)
    logger.info(f"Question:   {question}")
    logger.info(f"Transcript: {transcript}")

    # Compute embeddings
    q_emb = embedding.embed(question)
    t_emb = embedding.embed(transcript)
    similarity = embedding.similarity(q_emb, t_emb)

    logger.info(f"Similarity: {similarity:.3f}")

    # SLM reasoning (only if similarity is high enough)
    slm_verdict = False
    if similarity >= settings.similarity_threshold_low:
        slm_verdict = slm.predict(question, transcript)
        logger.info(f"SLM Verdict: {'YES (related)' if slm_verdict else 'NO (not related)'}")
    else:
        logger.info("SLM Verdict: SKIPPED (low similarity)")

    # Check result
    passed = (slm_verdict == expected)
    
    if passed:
        logger.info(f"PASS — Expected: {expected}, Got: {slm_verdict}")
    else:
        logger.error(f"FAIL — Expected: {expected}, Got: {slm_verdict}")

    return passed


def main():
    logger.info("=" * 70)
    logger.info("SLM Content Verification Test (Text-only)")
    logger.info("=" * 70)

    # Initialize settings
    settings = Settings()
    logger.info(f"Device: {settings.torch_device}")
    logger.info(f"SLM Enabled: {settings.slm_enabled}")
    logger.info(f"SLM Model: {settings.slm_model_path}")

    # Check if SLM is available
    if not settings.slm_enabled or not settings.slm_model_path:
        logger.error(" SLM is disabled (SLM_ENABLED=false or SLM_MODEL_PATH not set)")
        logger.info("   Set in .env: SLM_ENABLED=true")
        logger.info("   Download: uv run python scripts/download_models.py --slm")
        sys.exit(1)

    if not settings.slm_model_path.exists():
        logger.error(f"SLM model not found: {settings.slm_model_path}")
        logger.info("Download: uv run python scripts/download_models.py --slm")
        sys.exit(1)

    # Load models
    logger.info("\nLoading models...")

    embedding = EmbeddingProcessor(device=settings.torch_device)
    embedding.load_model()
    logger.info("✓ Embedding model loaded")

    slm = SLMProcessor(
        model_path=settings.slm_model_path,
        n_gpu_layers=settings.slm_n_gpu_layers,
        max_tokens=settings.slm_max_tokens,
        context_length=settings.slm_context_length,
    )
    slm.load_model()
    logger.info("SLM model loaded")

    # Run all test cases
    tests = test_cases()
    passed = 0
    failed = 0

    for test in tests:
        if run_test(test, embedding, slm, settings):
            passed += 1
        else:
            failed += 1

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total:  {len(tests)}")
    logger.info(f"Passed: {passed} ✓")
    logger.info(f"Failed: {failed} ✗")
    logger.info("=" * 70)

    if failed == 0:
        logger.info("ALL TESTS PASSED!")
        sys.exit(0)
    else:
        logger.error(f"{failed} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
