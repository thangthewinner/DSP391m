"""Audio processing pipeline orchestration."""

import asyncio
import base64
import logging
from typing import Optional

import numpy as np

from src.core.config import settings
from src.core.models import TranscriptSegment
from src.core.session import session_manager
from src.processing.buffer import AudioBuffer
from src.processing.stt import STTProcessor
from src.processing.vad import VADProcessor
from src.storage.transcript_store import TranscriptStore

logger = logging.getLogger(__name__)


class AudioPipeline:
    """Audio processing pipeline coordinating VAD, Buffer, and STT."""

    def __init__(
        self,
        vad_processor: VADProcessor,
        stt_processor: STTProcessor,
        transcript_store: TranscriptStore,
    ):
        """
        Initialize audio pipeline.

        Args:
            vad_processor: VAD processor instance
            stt_processor: STT processor instance
            transcript_store: Transcript storage instance
        """
        self.vad = vad_processor
        self.stt = stt_processor
        self.transcript_store = transcript_store

        # Processing state
        self._processing_tasks = {}
        self._should_stop = {}

    async def process_session(self, session_id: str) -> None:
        """
        Process audio stream for a session.

        Args:
            session_id: Session identifier
        """
        logger.info(f"Starting audio processing for session {session_id}")

        # Get session and queue
        session = session_manager.get_session(session_id)
        audio_queue = session_manager.get_audio_queue(session_id)

        if not session or not audio_queue:
            logger.error(f"Session or queue not found: {session_id}")
            return

        # Initialize buffer
        buffer = AudioBuffer(
            buffer_duration=settings.buffer_size,
            sample_rate=settings.audio_sample_rate,
        )

        # Processing state
        self._should_stop[session_id] = False
        chunk_count = 0
        last_transcription_time = 0.0

        try:
            while not self._should_stop.get(session_id, False):
                try:
                    # Get audio chunk from queue with timeout
                    audio_data = await asyncio.wait_for(
                        audio_queue.get(), timeout=1.0
                    )

                    chunk_count += 1
                    timestamp = audio_data.get("timestamp", 0.0)
                    audio_bytes = base64.b64decode(audio_data["data"])

                    # Convert to numpy array
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                    audio_float = audio_array.astype(np.float32) / 32768.0

                    # VAD check
                    is_speech = self.vad.process_chunk(audio_float)

                    if not is_speech:
                        logger.debug(f"Chunk {chunk_count}: No speech detected, skipping")
                        continue

                    # Add to buffer
                    buffer.add_chunk(audio_float, timestamp)
                    logger.debug(
                        f"Chunk {chunk_count}: Speech detected, added to buffer "
                        f"(duration: {buffer.duration:.2f}s)"
                    )

                    # Check if buffer is ready for transcription
                    if buffer.is_ready() and (timestamp - last_transcription_time) >= 5.0:
                        # Get buffer audio
                        buffer_audio = buffer.get_buffer_array()

                        # Transcribe in background
                        asyncio.create_task(
                            self._transcribe_and_save(
                                session_id, buffer_audio, timestamp
                            )
                        )

                        last_transcription_time = timestamp

                except asyncio.TimeoutError:
                    # No data in queue, continue waiting
                    continue

                except Exception as e:
                    logger.error(f"Error processing audio chunk: {e}", exc_info=True)
                    continue

        except asyncio.CancelledError:
            logger.info(f"Processing cancelled for session {session_id}")

        finally:
            # Final transcription if buffer has data
            if buffer.is_ready():
                buffer_audio = buffer.get_buffer_array()
                await self._transcribe_and_save(
                    session_id, buffer_audio, timestamp=None
                )

            logger.info(
                f"Audio processing stopped for session {session_id}, "
                f"processed {chunk_count} chunks"
            )

    async def _transcribe_and_save(
        self,
        session_id: str,
        audio: np.ndarray,
        timestamp: Optional[float],
    ) -> None:
        """
        Transcribe audio and save to storage.

        Args:
            session_id: Session identifier
            audio: Audio array to transcribe
            timestamp: Timestamp of the audio
        """
        try:
            # Transcribe
            result = self.stt.transcribe(audio, settings.audio_sample_rate)

            if not result.get("text"):
                logger.debug("Empty transcription, skipping save")
                return

            # Create transcript segment
            duration = len(audio) / settings.audio_sample_rate
            segment = TranscriptSegment(
                timestamp_start=timestamp if timestamp else 0.0,
                timestamp_end=(timestamp + duration) if timestamp else duration,
                text=result["text"],
                confidence=result["confidence"],
            )

            # Save to storage
            await self.transcript_store.save_segment(session_id, segment)

            # Update session state
            session = session_manager.get_session(session_id)
            if session:
                session.transcript_segments.append(segment)

            logger.info(
                f"Transcribed and saved segment for session {session_id}: "
                f"'{result['text'][:50]}...' (confidence: {result['confidence']:.3f})"
            )

        except Exception as e:
            logger.error(f"Error in transcription: {e}", exc_info=True)

    def stop_processing(self, session_id: str) -> None:
        """
        Stop processing for a session.

        Args:
            session_id: Session identifier
        """
        self._should_stop[session_id] = True
        logger.info(f"Stop signal sent for session {session_id}")


# Global pipeline instance (will be initialized in main.py)
pipeline: Optional[AudioPipeline] = None


def get_pipeline() -> AudioPipeline:
    """Get global pipeline instance."""
    if pipeline is None:
        raise RuntimeError("Pipeline not initialized")
    return pipeline
