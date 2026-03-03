"""FastAPI main application."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src import __version__
from src.api.routes import enrollment, exam, health
from src.api.websocket import audio_handler
from src.core.config import settings
from src.processing import pipeline as pipeline_module
from src.processing.embedding import EmbeddingProcessor
from src.processing.overlap_detector import OverlapDetector
from src.processing.pipeline import AudioPipeline
from src.processing.slm import SLMProcessor
from src.processing.speaker_verification import SpeakerVerifier
from src.processing.stt import STTProcessor
from src.processing.vad import VADProcessor
from src.storage.transcript_store import TranscriptStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting AI Proctoring System...")
    logger.info(f"Version: {__version__}")

    # Ensure directories exist
    settings.ensure_directories()
    logger.info("Storage directories initialized")

    # Load models
    try:
        logger.info("Loading models...")

        # Initialize VAD
        vad = VADProcessor(
            threshold=settings.vad_threshold,
            sample_rate=settings.audio_sample_rate,
        )
        vad.load_model()

        # Initialize STT
        stt = STTProcessor(
            model_name=settings.stt_model_name,
            device=settings.torch_device,
        )
        # Use custom model if specified, otherwise use default CPU model
        if settings.stt_model_override:
            model_path = settings.stt_model_override
            logger.info(f"Using custom STT model: {model_path}")
        else:
            # Default to CPU model
            model_path = settings.model_cache_dir / "stt" / "phowhisper-small-ct2"
            logger.info(f"Using default STT model: {model_path}")
        
        stt.load_model(model_path=model_path)

        # Initialize Embedding
        embedding = EmbeddingProcessor(device=settings.torch_device)
        embedding.load_model()

        # Initialize SLM (optional — only if enabled and model path is set)
        slm = None
        if settings.slm_enabled and settings.slm_model_path:
            try:
                slm = SLMProcessor(
                    model_path=settings.slm_model_path,
                    n_gpu_layers=settings.slm_n_gpu_layers,
                    max_tokens=settings.slm_max_tokens,
                    context_length=settings.slm_context_length,
                )
                slm.load_model()
            except FileNotFoundError as e:
                logger.warning(f"SLM model not found, running without SLM: {e}")
                slm = None
            except Exception as e:
                logger.warning(f"Failed to load SLM, running without SLM: {e}")
                slm = None
        elif settings.slm_enabled:
            logger.info("SLM enabled but SLM_MODEL_PATH not set — skipping SLM")
        else:
            logger.info("SLM disabled (SLM_ENABLED=false)")

        # Initialize Speaker Verifier (Phase 5)
        verifier = None
        if settings.speaker_verification_enabled:
            try:
                verifier = SpeakerVerifier(
                    enrollment_dir=settings.enrollment_dir,
                    threshold=settings.speaker_verification_threshold,
                    device=settings.torch_device,
                )
                verifier.load_model()
            except Exception as e:
                logger.warning(f"Failed to load Speaker Verifier, running without it: {e}")
                verifier = None
        else:
            logger.info("Speaker verification disabled (SPEAKER_VERIFICATION_ENABLED=false)")

        # Initialize Overlap Detector / Diarization (Phase 6)
        overlap_detector = None
        if settings.diarization_enabled and settings.diarization_model_path:
            try:
                overlap_detector = OverlapDetector(
                    model_path=settings.diarization_model_path,
                    device=settings.torch_device,
                    min_audio_seconds=settings.min_diarization_audio_seconds,
                )
                overlap_detector.load_model()
            except FileNotFoundError as e:
                logger.warning(f"Diarization model not found, running without it: {e}")
                overlap_detector = None
            except Exception as e:
                logger.warning(f"Failed to load Diarization model, running without it: {e}")
                overlap_detector = None
        elif settings.diarization_enabled:
            logger.info("Diarization enabled but DIARIZATION_MODEL_PATH not set — skipping")
        else:
            logger.info("Diarization disabled (DIARIZATION_ENABLED=false)")

        # Initialize transcript store
        transcript_store = TranscriptStore()

        # Initialize pipeline
        pipeline_module.pipeline = AudioPipeline(
            vad_processor=vad,
            stt_processor=stt,
            embedding_processor=embedding,
            transcript_store=transcript_store,
            slm_processor=slm,
            speaker_verifier=verifier,
            overlap_detector=overlap_detector,
        )

        slm_status = f"loaded ({settings.slm_model_path.name})" if slm else "disabled/not loaded"
        verifier_status = "loaded" if verifier else "disabled/not loaded"
        diar_status = (
            f"loaded ({settings.diarization_model_path.name})"
            if overlap_detector
            else "disabled/not loaded"
        )
        logger.info(
            f"All models loaded successfully "
            f"(SLM: {slm_status}, Verifier: {verifier_status}, Diarization: {diar_status})"
        )

    except Exception as e:
        logger.error(f"Failed to load models: {e}", exc_info=True)
        raise

    logger.info(f"Server starting on {settings.api_host}:{settings.api_port}")

    yield

    # Shutdown
    logger.info("Shutting down AI Proctoring System...")


# Create FastAPI app
app = FastAPI(
    title="AI-Powered Exam Proctoring System",
    description="Real-time Vietnamese speech monitoring for online exams",
    version=__version__,
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(exam.router)
app.include_router(enrollment.router)
app.include_router(audio_handler.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "AI-Powered Exam Proctoring System",
        "version": __version__,
        "status": "running",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info",
    )
