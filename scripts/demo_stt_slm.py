#!/usr/bin/env python3
"""
Demo: Speech to Text + SLM Content Verification

Demonstrates the STT → Embedding → SLM pipeline for cheating detection.

Usage:
    # Test with sample audio file:
    uv run python scripts/demo_stt_slm.py --audio test_audio.wav --question "Giải thích OOP trong Python"

    # Test with microphone (record 10 seconds):
    uv run python scripts/demo_stt_slm.py --record 10 --question "Viết hàm đệ quy tính giai thừa"

    # Test with text input (skip STT):
    uv run python scripts/demo_stt_slm.py --text "Hàm đệ quy là hàm tự gọi chính nó" --question "Viết hàm đệ quy"
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.core.config import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_audio_file(audio_path: Path) -> tuple[np.ndarray, int]:
    """
    Load audio file and convert to float32 mono at 16kHz.

    Returns:
        (audio_array, sample_rate)
    """
    try:
        import soundfile as sf

        audio, sr = sf.read(str(audio_path))

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Convert to float32 in [-1, 1]
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            audio = audio.astype(np.float32)

        logger.info(f"✓ Loaded audio: {len(audio)/sr:.2f}s, {sr}Hz")
        return audio, sr

    except ImportError:
        logger.error("soundfile not installed. Run: pip install soundfile")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load audio: {e}")
        sys.exit(1)


def record_audio(duration: float, sample_rate: int = 16000) -> np.ndarray:
    """
    Record audio from microphone.

    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        audio_array (float32)
    """
    try:
        import sounddevice as sd

        logger.info(f"🎤 Recording {duration}s from microphone...")
        logger.info("   Speak now!")

        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()

        logger.info("✓ Recording complete")
        return audio.flatten()

    except ImportError:
        logger.error("sounddevice not installed. Run: pip install sounddevice")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to record audio: {e}")
        sys.exit(1)


def demo_pipeline(
    audio: np.ndarray,
    exam_question: str,
    settings: Settings,
) -> dict:
    """
    Run the full STT → Embedding → SLM pipeline.

    Args:
        audio: Audio array (float32, mono, 16kHz)
        exam_question: Exam question text
        settings: Application settings

    Returns:
        Result dictionary with transcript, similarity, slm_verdict
    """
    from src.processing.embedding import EmbeddingProcessor
    from src.processing.slm import SLMProcessor
    from src.processing.stt import STTProcessor
    from src.processing.vad import VADProcessor

    result = {
        "transcript": "",
        "confidence": 0.0,
        "similarity": 0.0,
        "slm_verdict": False,
        "cheating_detected": False,
    }

    # ─────────────────────────────────────────────────────────────────
    # Step 1: VAD (Voice Activity Detection)
    # ─────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: VAD (Voice Activity Detection)")
    logger.info("=" * 70)

    vad = VADProcessor(threshold=settings.vad_threshold)
    vad.load_model()

    is_speech = vad.process_chunk(audio)
    if not is_speech:
        logger.warning("⚠️  No speech detected in audio (silence)")
        return result

    logger.info("✓ Speech detected")

    # ─────────────────────────────────────────────────────────────────
    # Step 2: STT (Speech to Text)
    # ─────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: STT (Speech → Text)")
    logger.info("=" * 70)

    stt = STTProcessor(
        model_name=settings.stt_model_name,
        device=settings.torch_device,
        compute_type="int8" if settings.torch_device == "cpu" else "float16",
    )

    # Use custom model if specified
    if settings.stt_model_override:
        model_path = settings.stt_model_override
    else:
        model_path = settings.model_cache_dir / "stt" / "phowhisper-small-ct2"

    logger.info(f"Loading STT model: {model_path}")
    stt.load_model(model_path=model_path)

    stt_result = stt.transcribe(audio, sample_rate=16000)
    transcript = stt_result.get("text", "").strip()
    confidence = stt_result.get("confidence", 0.0)

    result["transcript"] = transcript
    result["confidence"] = confidence

    if not transcript:
        logger.warning("⚠️  Empty transcript")
        return result

    logger.info(f"\n📝 Transcript: \"{transcript}\"")
    logger.info(f"   Confidence: {confidence:.3f}")

    # ─────────────────────────────────────────────────────────────────
    # Step 3: Embedding (Semantic Similarity)
    # ─────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Embedding (Semantic Similarity)")
    logger.info("=" * 70)

    embedding = EmbeddingProcessor(device=settings.torch_device)
    embedding.load_model()

    # Compute embeddings
    question_embedding = embedding.embed(exam_question)
    transcript_embedding = embedding.embed(transcript)

    similarity = embedding.similarity(question_embedding, transcript_embedding)
    result["similarity"] = similarity

    logger.info(f"📊 Exam Question: \"{exam_question}\"")
    logger.info(f"   Similarity Score: {similarity:.3f}")

    if similarity < settings.similarity_threshold_low:
        logger.info("✓ Low similarity — content not related to exam")
        return result

    logger.warning(f"⚠️  Similarity ≥ {settings.similarity_threshold_low} — checking with SLM...")

    # ─────────────────────────────────────────────────────────────────
    # Step 4: SLM (Small Language Model Reasoning)
    # ─────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: SLM Reasoning (YES/NO)")
    logger.info("=" * 70)

    if not settings.slm_enabled or not settings.slm_model_path:
        logger.warning("⚠️  SLM disabled — skipping reasoning")
        return result

    if not settings.slm_model_path.exists():
        logger.warning(f"⚠️  SLM model not found: {settings.slm_model_path}")
        logger.info("   Run: uv run python scripts/download_models.py --slm")
        return result

    slm = SLMProcessor(
        model_path=settings.slm_model_path,
        n_gpu_layers=settings.slm_n_gpu_layers,
        max_tokens=settings.slm_max_tokens,
        context_length=settings.slm_context_length,
    )
    slm.load_model()

    slm_verdict = slm.predict(exam_question, transcript)
    result["slm_verdict"] = slm_verdict

    if slm_verdict:
        logger.error("🚨 SLM: YES — Content IS related to exam question!")
        logger.error("   → CHEATING DETECTED")
        result["cheating_detected"] = True
    else:
        logger.info("✓ SLM: NO — Content not related to exam question")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Demo: Speech to Text + SLM Content Verification"
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--audio", type=Path, help="Path to audio file (.wav)")
    input_group.add_argument("--record", type=float, help="Record N seconds from mic")
    input_group.add_argument("--text", type=str, help="Test with text input (skip STT)")

    # Required exam question
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Exam question text for comparison",
    )

    # Optional settings
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for inference (default: cpu)",
    )

    args = parser.parse_args()

    # Initialize settings
    settings = Settings()
    settings.torch_device = args.device

    logger.info("\n" + "=" * 70)
    logger.info("AI Exam Proctoring System — STT + SLM Demo")
    logger.info("=" * 70)
    logger.info(f"Device: {settings.torch_device}")
    logger.info(f"Exam Question: \"{args.question}\"")

    # ─────────────────────────────────────────────────────────────────
    # Load or record audio
    # ─────────────────────────────────────────────────────────────────
    if args.text:
        # Text-only mode (skip STT)
        logger.info("\n📝 Text input mode (skipping STT)")
        logger.info(f"   Input: \"{args.text}\"")

        from src.processing.embedding import EmbeddingProcessor
        from src.processing.slm import SLMProcessor

        embedding = EmbeddingProcessor(device=settings.torch_device)
        embedding.load_model()

        question_embedding = embedding.embed(args.question)
        text_embedding = embedding.embed(args.text)
        similarity = embedding.similarity(question_embedding, text_embedding)

        logger.info(f"\n📊 Similarity: {similarity:.3f}")

        if similarity >= settings.similarity_threshold_low and settings.slm_enabled and settings.slm_model_path and settings.slm_model_path.exists():
            slm = SLMProcessor(
                model_path=settings.slm_model_path,
                n_gpu_layers=settings.slm_n_gpu_layers,
                max_tokens=settings.slm_max_tokens,
                context_length=settings.slm_context_length,
            )
            slm.load_model()
            verdict = slm.predict(args.question, args.text)
            logger.info(f"🤖 SLM Verdict: {'YES (related)' if verdict else 'NO (not related)'}")
            if verdict:
                logger.error("🚨 CHEATING DETECTED")

        return

    if args.audio:
        audio, sr = load_audio_file(args.audio)
        if sr != 16000:
            logger.info(f"   Resampling from {sr}Hz to 16000Hz...")
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    elif args.record:
        audio = record_audio(args.record, sample_rate=16000)

    # ─────────────────────────────────────────────────────────────────
    # Run pipeline
    # ─────────────────────────────────────────────────────────────────
    start_time = time.time()
    result = demo_pipeline(audio, args.question, settings)
    elapsed = time.time() - start_time

    # ─────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Transcript:     \"{result['transcript']}\"")
    logger.info(f"Confidence:     {result['confidence']:.3f}")
    logger.info(f"Similarity:     {result['similarity']:.3f}")
    logger.info(f"SLM Verdict:    {'YES' if result['slm_verdict'] else 'NO'}")
    logger.info(f"Cheating:       {'🚨 DETECTED' if result['cheating_detected'] else '✓ Not detected'}")
    logger.info(f"Processing Time: {elapsed:.2f}s")
    logger.info("=" * 70)

    # Exit with status code
    sys.exit(1 if result["cheating_detected"] else 0)


if __name__ == "__main__":
    main()
