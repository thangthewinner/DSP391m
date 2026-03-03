"""Speaker enrollment endpoints — Phase 5."""

import base64
import logging
from datetime import datetime

import numpy as np
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.processing import pipeline as pipeline_module

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/enroll", tags=["enrollment"])


class EnrollmentRequest(BaseModel):
    """Speaker enrollment request."""

    audio_samples: list[str] = Field(
        ..., description="Base64-encoded PCM int16 audio samples (3-5 recommended)"
    )
    sample_rate: int = Field(default=16000, description="Audio sample rate in Hz")


class EnrollmentResponse(BaseModel):
    """Speaker enrollment response."""

    student_id: str
    status: str
    samples_used: int
    message: str


class EnrollmentStatusResponse(BaseModel):
    """Enrollment status response."""

    student_id: str
    enrolled: bool
    enrolled_at: str = ""


@router.post(
    "/{student_id}",
    response_model=EnrollmentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def enroll_speaker(student_id: str, request: EnrollmentRequest):
    """
    Enroll a student's voice for speaker verification.

    Requires 3-5 audio samples (each at least 3 seconds of speech).
    Computes average ECAPA-TDNN embedding and saves to disk.
    """
    p = pipeline_module.pipeline
    if p is None or p.verifier is None:  # type: ignore[union-attr]
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Speaker verification model not loaded",
        )
    verifier = p.verifier  # type: ignore[union-attr]

    if not request.audio_samples:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No audio samples provided",
        )

    # Decode base64 audio samples
    audio_arrays = []
    for i, b64 in enumerate(request.audio_samples):
        try:
            raw = base64.b64decode(b64)
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            audio_arrays.append(arr)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to decode enrollment sample %d: %s", i + 1, e)

    if not audio_arrays:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="All audio samples failed to decode",
        )

    import asyncio

    loop = asyncio.get_event_loop()
    success = await loop.run_in_executor(
        None,
        verifier.enroll,
        student_id,
        audio_arrays,
        request.sample_rate,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Enrollment failed — audio samples may be too short or silent",
        )

    return EnrollmentResponse(
        student_id=student_id,
        status="enrolled",
        samples_used=len(audio_arrays),
        message=f"Successfully enrolled {len(audio_arrays)} sample(s) for student {student_id}",
    )


@router.get("/{student_id}/status", response_model=EnrollmentStatusResponse)
async def get_enrollment_status(student_id: str):
    """Check if a student has an enrollment."""
    p = pipeline_module.pipeline
    if p is None or p.verifier is None:  # type: ignore[union-attr]
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Speaker verification model not loaded",
        )
    verifier = p.verifier  # type: ignore[union-attr]

    enrolled = verifier.is_enrolled(student_id)
    enrolled_at = ""

    if enrolled:
        path = verifier.enrollment_dir / f"{student_id}.npy"
        mtime = path.stat().st_mtime
        enrolled_at = datetime.fromtimestamp(mtime).isoformat()

    return EnrollmentStatusResponse(
        student_id=student_id,
        enrolled=enrolled,
        enrolled_at=enrolled_at,
    )


@router.delete("/{student_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_enrollment(student_id: str):
    """Delete a student's enrollment."""
    p = pipeline_module.pipeline
    if p is None or p.verifier is None:  # type: ignore[union-attr]
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Speaker verification model not loaded",
        )
    verifier = p.verifier  # type: ignore[union-attr]

    deleted = verifier.delete_enrollment(student_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No enrollment found for student {student_id}",
        )
