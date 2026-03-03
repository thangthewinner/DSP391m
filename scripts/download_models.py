#!/usr/bin/env python3
"""Download all required models for the proctoring system."""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from huggingface_hub import hf_hub_download, snapshot_download

from src.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def download_silero_vad():
    """
    Download Silero VAD model with retry logic.
    
    Note: We don't save the JIT model because it can't be serialized.
    torch.hub will cache it automatically in ~/.cache/torch/hub/
    """
    logger.info("Downloading Silero VAD model...")
    logger.info("Note: Model will be cached by torch.hub (not saved to models/)")
    
    # Try multiple times with different approaches
    for attempt in range(3):
        try:
            if attempt > 0:
                logger.info(f"Retry attempt {attempt + 1}/3...")
            
            # Try with trust_repo=True to avoid confirmation
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=(attempt > 0),  # Force reload on retry
                onnx=False,
                trust_repo=True,  # Trust the repo to avoid warnings
            )

            # Verify model loaded successfully (don't try to save JIT model)
            if model is not None:
                logger.info("✓ Silero VAD loaded and cached successfully")
                logger.info("  Cache location: ~/.cache/torch/hub/")
                return True

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                import time
                time.sleep(2)  # Wait before retry
            continue
    
    logger.error("✗ Failed to download Silero VAD after 3 attempts")
    logger.info("Note: The system will auto-download on first run")
    return False


def download_phowhisper():
    """Download PhoWhisper model."""
    logger.info("Downloading PhoWhisper model...")
    logger.info("Note: This downloads the PyTorch model.")
    logger.info("faster-whisper will auto-convert to CTranslate2 on first use (1-2 min).")
    
    try:
        # Download to local directory for faster-whisper
        model_path = settings.stt_model_path / "phowhisper-small"
        model_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading to {model_path}...")
        
        # Download with all files
        snapshot_download(
            repo_id="vinai/PhoWhisper-small",
            local_dir=str(model_path),
            local_dir_use_symlinks=False,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],  # Skip unnecessary files
        )

        logger.info(f"✓ PhoWhisper saved to {model_path}")
        logger.info("  Note: Model will be converted to CTranslate2 format on first server start")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to download PhoWhisper: {e}")
        logger.info("  Fallback: faster-whisper will download directly from HuggingFace on first run")
        return False


def download_slm(model_size: str = "3b") -> bool:
    """
    Download Qwen2.5 GGUF model for SLM reasoning layer.

    Args:
        model_size: "1.5b" or "3b" (default: "3b")
    """
    size_map = {
        "1.5b": {
            "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        },
        "3b": {
            "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
            "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
        },
    }

    if model_size not in size_map:
        logger.error(f"Unknown model size: {model_size}. Choose '1.5b' or '3b'")
        return False

    info = size_map[model_size]
    dest_dir = settings.slm_model_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / info["filename"]

    if dest_path.exists():
        logger.info(f"✓ SLM model already exists: {dest_path}")
        return True

    logger.info(f"Downloading SLM: {info['repo']} / {info['filename']}")
    logger.info(f"  Destination: {dest_path}")
    logger.info(f"  Size: ~{'1.1GB' if model_size == '1.5b' else '2.0GB'}")

    try:
        hf_hub_download(
            repo_id=info["repo"],
            filename=info["filename"],
            local_dir=str(dest_dir),
            local_dir_use_symlinks=False,
        )
        logger.info(f"✓ SLM model saved to {dest_path}")
        logger.info(f"  Set SLM_MODEL_PATH={dest_path} in .env to use it")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to download SLM model: {e}")
        return False


def download_diarization() -> bool:
    """
    Download NeMo Streaming Sortformer diarization model.

    Model: nvidia/diar_streaming_sortformer_4spk-v2
    Size: ~988MB (.nemo format)
    """
    repo = "nvidia/diar_streaming_sortformer_4spk-v2"
    filename = "diar_streaming_sortformer_4spk-v2.nemo"

    dest_dir = settings.diarization_model_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename

    if dest_path.exists():
        logger.info(f"✓ Diarization model already exists: {dest_path}")
        return True

    logger.info(f"Downloading diarization model: {repo}")
    logger.info(f"  Destination: {dest_path}")
    logger.info("  Size: ~988MB")

    try:
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=str(dest_dir),
            local_dir_use_symlinks=False,
        )
        logger.info(f"✓ Diarization model saved to {dest_path}")
        logger.info(f"  Set DIARIZATION_MODEL_PATH={dest_path} in .env to use it")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to download diarization model: {e}")
        return False


def main():
    """Main download function."""
    parser = argparse.ArgumentParser(description="Download models for AI Proctoring System")
    parser.add_argument("--slm", action="store_true", help="Download SLM model (Phase 3)")
    parser.add_argument(
        "--slm-size",
        choices=["1.5b", "3b"],
        default="3b",
        help="SLM model size (default: 3b)",
    )
    parser.add_argument(
        "--diarization", action="store_true", help="Download diarization model (Phase 6)"
    )
    parser.add_argument("--all", action="store_true", help="Download all models")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Model Download Script for AI Proctoring System")
    logger.info("=" * 60)

    # Ensure directories exist
    settings.ensure_directories()
    logger.info(f"Model cache directory: {settings.model_cache_dir}")

    results = []

    if not args.slm and not args.diarization:
        # Default: download base models
        logger.info("\n[1/2] Downloading Silero VAD...")
        results.append(("Silero VAD", download_silero_vad()))

        logger.info("\n[2/2] Downloading PhoWhisper...")
        results.append(("PhoWhisper", download_phowhisper()))

    if args.slm or args.all:
        logger.info(f"\n[SLM] Downloading Qwen2.5-{args.slm_size.upper()}-Instruct-GGUF...")
        results.append((f"SLM (Qwen2.5-{args.slm_size})", download_slm(args.slm_size)))

    if args.diarization or args.all:
        logger.info("\n[Diarization] Downloading NeMo Streaming Sortformer...")
        results.append(("Diarization (NeMo Sortformer)", download_diarization()))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Download Summary")
    logger.info("=" * 60)

    for model_name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"{model_name}: {status}")

    all_success = all(success for _, success in results)

    if all_success:
        logger.info("\n✓ All models downloaded successfully!")
        hints = []
        if args.slm or args.all:
            hints.append("Set SLM_MODEL_PATH in .env")
        if args.diarization or args.all:
            hints.append("Set DIARIZATION_MODEL_PATH in .env")
        if hints:
            logger.info("\nNext steps: " + " | ".join(hints))
        else:
            logger.info("\nYou can now start the server with:")
            logger.info("  uvicorn src.api.main:app --reload")
        return 0
    else:
        logger.error("\n✗ Some models failed to download.")
        logger.error("Please check the errors above and try again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
