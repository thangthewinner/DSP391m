#!/usr/bin/env python3
"""Download all required models for the proctoring system."""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from huggingface_hub import snapshot_download

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


def main():
    """Main download function."""
    logger.info("=" * 60)
    logger.info("Model Download Script for AI Proctoring System")
    logger.info("=" * 60)

    # Ensure directories exist
    settings.ensure_directories()
    logger.info(f"Model cache directory: {settings.model_cache_dir}")

    # Download models
    results = []

    logger.info("\n[1/2] Downloading Silero VAD...")
    results.append(("Silero VAD", download_silero_vad()))

    logger.info("\n[2/2] Downloading PhoWhisper...")
    results.append(("PhoWhisper", download_phowhisper()))

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
        logger.info("\nYou can now start the server with:")
        logger.info("  uvicorn src.api.main:app --reload")
        return 0
    else:
        logger.error("\n✗ Some models failed to download.")
        logger.error("Please check the errors above and try again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
