"""Health check endpoint."""

import logging

from fastapi import APIRouter

from src import __version__
from src.core.models import HealthResponse
from src.processing import pipeline as pipeline_module

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    p = pipeline_module.pipeline
    models_loaded = (
        p is not None
        and p.vad.model is not None
        and p.stt.model is not None
    )
    slm_loaded = p is not None and p.slm is not None
    verifier_loaded = p is not None and p.verifier is not None
    diarization_loaded = p is not None and p.overlap_detector is not None

    return HealthResponse(
        status="healthy" if models_loaded else "initializing",
        models_loaded=models_loaded,
        version=__version__,
        slm_loaded=slm_loaded,
        verifier_loaded=verifier_loaded,
        diarization_loaded=diarization_loaded,
    )
