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
from src.processing.pipeline import AudioPipeline
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

        # Initialize transcript store
        transcript_store = TranscriptStore()

        # Initialize pipeline
        pipeline_module.pipeline = AudioPipeline(
            vad_processor=vad,
            stt_processor=stt,
            embedding_processor=embedding,
            transcript_store=transcript_store,
        )

        logger.info("All models loaded successfully")

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
