"""
Convert PhoWhisper PyTorch model to CTranslate2 format.

This script manually converts the model that faster-whisper needs.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def convert_model():
    """Convert PhoWhisper to CTranslate2 format."""
    try:
        import ctranslate2
        from transformers import WhisperProcessor
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install with: uv pip install ctranslate2 transformers")
        return False

    model_name = "vinai/PhoWhisper-small"
    source_path = settings.model_cache_dir / "stt" / "phowhisper-small"
    output_path = settings.model_cache_dir / "stt" / "phowhisper-small-ct2"

    logger.info("=" * 60)
    logger.info("PhoWhisper Model Conversion")
    logger.info("=" * 60)
    logger.info(f"Source: {source_path}")
    logger.info(f"Output: {output_path}")
    logger.info("")

    # Check if source exists
    if not source_path.exists():
        logger.error(f"Source model not found: {source_path}")
        logger.error("Run: python scripts/download_models.py first")
        return False

    # Check if already converted
    if output_path.exists() and (output_path / "model.bin").exists():
        logger.info("✓ Model already converted!")
        logger.info(f"  Location: {output_path}")
        return True

    # Convert
    try:
        logger.info("Converting model (this takes 1-2 minutes)...")
        output_path.mkdir(parents=True, exist_ok=True)

        converter = ctranslate2.converters.TransformersConverter(
            str(source_path),
            activation_scales=None,
            copy_files=["tokenizer.json", "preprocessor_config.json"],
        )

        converter.convert(
            output_dir=str(output_path),
            quantization="int8",  # Use int8 for CPU
            force=True,
        )

        logger.info("✓ Conversion successful!")
        logger.info(f"  Output: {output_path}")
        logger.info("")
        logger.info("Update your .env file:")
        logger.info(f"  STT_MODEL_PATH={output_path}")
        return True

    except Exception as e:
        logger.error(f"✗ Conversion failed: {e}")
        logger.error("")
        logger.error("Troubleshooting:")
        logger.error("  1. Ensure source model is complete")
        logger.error("  2. Check disk space (~2GB needed)")
        logger.error("  3. Try: uv pip install --upgrade ctranslate2 transformers")
        return False


if __name__ == "__main__":
    success = convert_model()
    sys.exit(0 if success else 1)
