"""WebSocket audio streaming handler."""

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

from src.core.models import (
    AckResponse,
    AudioChunkMessage,
    ErrorResponse,
    SessionStatus,
    StatusUpdateResponse,
)
from src.core.session import session_manager
from src.processing.pipeline import get_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/audio/{session_id}")
async def audio_stream_handler(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for audio streaming.

    Args:
        websocket: WebSocket connection
        session_id: Session identifier
    """
    # Verify session exists
    session = session_manager.get_session(session_id)
    if not session:
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Invalid session ID",
        )
        return

    # Get audio queue
    audio_queue = session_manager.get_audio_queue(session_id)
    if not audio_queue:
        await websocket.close(
            code=status.WS_1011_INTERNAL_ERROR,
            reason="Audio queue not found",
        )
        return

    # Accept connection
    await websocket.accept()
    logger.info(f"WebSocket connected for session {session_id}")

    # Start processing task
    pipeline = get_pipeline()
    processing_task = asyncio.create_task(pipeline.process_session(session_id))
    session_manager.set_processing_task(session_id, processing_task)

    chunk_count = 0

    try:
        while True:
            # Stop receiving when session is no longer active
            if session.status != SessionStatus.ACTIVE:
                logger.info(
                    "Stop receiving audio for non-active session %s", session_id
                )
                break

            # Receive message
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            # Parse message
            try:
                message = AudioChunkMessage(**data)
            except Exception as e:
                logger.error(f"Invalid message format: {e}")
                error_response = ErrorResponse(
                    code="INVALID_AUDIO",
                    message=f"Invalid message format: {str(e)}",
                )
                await websocket.send_json(error_response.model_dump())
                continue

            # Handle different message types
            if message.type == "audio_chunk":
                chunk_count += 1

                try:
                    await asyncio.wait_for(
                        audio_queue.put(
                            {
                                "data": message.data,
                                "timestamp": message.timestamp,
                                "sample_rate": message.sample_rate,
                            }
                        ),
                        timeout=0.5,
                    )

                    # Send acknowledgment
                    ack = AckResponse(chunk_id=chunk_count)
                    await websocket.send_json(ack.model_dump())

                    # Send status update every 5 chunks
                    if chunk_count % 5 == 0:
                        status_update = StatusUpdateResponse(
                            cheating_flag=session.cheating_flag,
                        )
                        await websocket.send_json(status_update.model_dump())

                except asyncio.TimeoutError:
                    logger.warning(
                        "Audio queue put timeout",
                        extra={
                            "session_id": session_id,
                            "queue_size": audio_queue.qsize(),
                        },
                    )
                    error_response = ErrorResponse(
                        code="QUEUE_TIMEOUT",
                        message="Audio processing queue is congested",
                    )
                    await websocket.send_json(error_response.model_dump())
                except asyncio.QueueFull:
                    logger.warning(
                        "Audio queue full",
                        extra={
                            "session_id": session_id,
                            "queue_size": audio_queue.qsize(),
                        },
                    )
                    error_response = ErrorResponse(
                        code="QUEUE_OVERFLOW",
                        message="Audio processing queue is full",
                    )
                    await websocket.send_json(error_response.model_dump())

            elif message.type == "ping":
                await websocket.send_json({"type": "pong"})

            else:
                logger.warning(f"Unknown message type: {message.type}")

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")

    except Exception as e:
        logger.error(f"Error in WebSocket handler: {e}", exc_info=True)
        try:
            error_response = ErrorResponse(
                code="PROCESSING_ERROR",
                message=f"Internal error: {str(e)}",
            )
            await websocket.send_json(error_response.model_dump())
        except Exception:
            pass

    finally:
        # Stop processing
        pipeline.stop_processing(session_id)

        # Wait for processing task to complete
        try:
            await asyncio.wait_for(processing_task, timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning(f"Processing task timeout for session {session_id}")
            processing_task.cancel()

        # Clean up session resources (queues, tasks) to prevent memory leak
        session_manager.cleanup_session(session_id)

        logger.info(
            f"WebSocket closed for session {session_id}, processed {chunk_count} chunks"
        )
