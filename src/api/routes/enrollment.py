"""Speaker enrollment endpoint (stub for Phase 5)."""

import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/enroll", tags=["enrollment"])


class EnrollmentRequest(BaseModel):
    """Speaker enrollment request."""

    user_id: str
    audio_samples: list[str]  # Base64 encoded
    sample_rate: int = 16000


class EnrollmentResponse(BaseModel):
    """Speaker enrollment response."""

    user_id: str
    enrollment_status: str
    message: str


@router.post("/", response_model=EnrollmentResponse, status_code=status.HTTP_201_CREATED)
async def enroll_speaker(request: EnrollmentRequest):
    """
    Enroll a speaker (stub for Phase 5).

    Args:
        request: Enrollment request

    Returns:
        Enrollment status
    """
    logger.info(f"Enrollment request for user {request.user_id} (Phase 5 - not implemented)")

    return EnrollmentResponse(
        user_id=request.user_id,
        enrollment_status="pending",
        message="Speaker enrollment will be implemented in Phase 5",
    )
