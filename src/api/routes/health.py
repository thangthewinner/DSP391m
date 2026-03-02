"""Health check endpoint."""

import logging

from fastapi import APIRouter

from src import __version__
from src.core.models import HealthResponse
from src.processing.pipeline import pipeline

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns:
        Health status and model loading status
    """
    models_loaded = pipeline is not None and pipeline.vad.model is not None and pipeline.stt.model is not None

    return HealthResponse(
        status="healthy" if models_loaded else "initializing",
        models_loaded=models_loaded,
        version=__version__,
    )
