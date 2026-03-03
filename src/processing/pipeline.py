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
from src.processing.decision_engine import decision_engine
from src.processing.embedding import EmbeddingProcessor
from src.processing.slm import SLMProcessor
from src.processing.stt import STTProcessor
from src.processing.vad import VADProcessor
from src.storage.transcript_store import TranscriptStore

logger = logging.getLogger(__name__)

MIN_CONFIDENCE = 0.5


class AudioPipeline:
    """Audio processing pipeline: VAD → Buffer → STT → Embedding → SLM → Decision Engine."""

    def __init__(
        self,
        vad_processor: VADProcessor,
        stt_processor: STTProcessor,
        embedding_processor: EmbeddingProcessor,
        transcript_store: TranscriptStore,
        slm_processor: Optional[SLMProcessor] = None,
    ):
        self.vad = vad_processor
        self.stt = stt_processor
        self.embedding = embedding_processor
        self.slm = slm_processor
        self.transcript_store = transcript_store

        self._should_stop: dict[str, bool] = {}

    async def process_session(self, session_id: str) -> None:
        """Process audio stream for a session."""
        logger.info(f"Starting audio processing for session {session_id}")

        session = session_manager.get_session(session_id)
        audio_queue = session_manager.get_audio_queue(session_id)

        if not session or not audio_queue:
            logger.error(f"Session or queue not found: {session_id}")
            return

        buffer = AudioBuffer(
            buffer_duration=settings.buffer_size,
            sample_rate=settings.audio_sample_rate,
        )

        self._should_stop[session_id] = False
        chunk_count = 0
        last_transcription_time = 0.0
        timestamp = 0.0

        try:
            while not self._should_stop.get(session_id, False):
                try:
                    audio_data = await asyncio.wait_for(audio_queue.get(), timeout=1.0)

                    chunk_count += 1
                    timestamp = audio_data.get("timestamp", 0.0)
                    audio_bytes = base64.b64decode(audio_data["data"])

                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                    audio_float = audio_array.astype(np.float32) / 32768.0

                    # VAD
                    if not self.vad.process_chunk(audio_float):
                        logger.debug(f"Chunk {chunk_count}: silence, skipping")
                        continue

                    buffer.add_chunk(audio_float, timestamp)

                    # Transcribe every 5s once buffer has enough data
                    if buffer.is_ready() and (timestamp - last_transcription_time) >= 5.0:
                        asyncio.create_task(
                            self._process_buffer(session_id, buffer.get_buffer_array(), timestamp)
                        )
                        last_transcription_time = timestamp

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error processing chunk: {e}", exc_info=True)
                    continue

        except asyncio.CancelledError:
            logger.info(f"Processing cancelled for session {session_id}")

        finally:
            # Final flush
            if buffer.is_ready():
                await self._process_buffer(session_id, buffer.get_buffer_array(), timestamp)

            logger.info(f"Processing stopped for session {session_id} ({chunk_count} chunks)")

    async def _process_buffer(
        self,
        session_id: str,
        audio: np.ndarray,
        timestamp: float,
    ) -> None:
        """STT → Embedding → Decision Engine → Update session state."""
        try:
            # 1. Transcribe
            result = self.stt.transcribe(audio, settings.audio_sample_rate)
            text = result.get("text", "").strip()
            confidence = result.get("confidence", 0.0)

            if not text:
                return

            if confidence < MIN_CONFIDENCE:
                logger.debug(f"Low confidence ({confidence:.2f}), skipping: '{text[:40]}'")
                return

            duration = len(audio) / settings.audio_sample_rate
            segment = TranscriptSegment(
                timestamp_start=timestamp,
                timestamp_end=timestamp + duration,
                text=text,
                confidence=confidence,
            )

            # Save transcript
            await self.transcript_store.save_segment(session_id, segment)

            session = session_manager.get_session(session_id)
            if not session:
                return

            session.transcript_segments.append(segment)

            logger.info(
                f"[{session_id[:8]}] STT: '{text[:60]}' (conf={confidence:.2f})"
            )

            # 2. Embedding similarity (only if exam_question is set)
            similarity = 0.0
            if session.question_embedding is not None and text:
                loop = asyncio.get_event_loop()
                similarity = await loop.run_in_executor(
                    None,
                    self.embedding.similarity_to_question,
                    text,
                    session.question_embedding,
                )
                logger.info(
                    f"[{session_id[:8]}] Similarity: {similarity:.3f} — '{text[:40]}'"
                )

            # 3. SLM reasoning (only when similarity is high enough to be worth checking)
            slm_verdict = False
            if (
                self.slm is not None
                and similarity >= settings.similarity_threshold_low
                and session.exam_question
            ):
                loop = asyncio.get_event_loop()
                slm_verdict = await loop.run_in_executor(
                    None,
                    self.slm.predict,
                    session.exam_question,
                    text,
                )
                logger.info(
                    f"[{session_id[:8]}] SLM: {'YES ⚠️' if slm_verdict else 'NO'} "
                    f"(similarity={similarity:.2f})"
                )

            # 4. Decision Engine
            state = decision_engine.process(
                session_id=session_id,
                text=text,
                similarity_score=similarity,
                timestamp=timestamp,
                slm_verdict=slm_verdict,
            )

            # 5. Sync back to session state
            session.suspicion_score = state["suspicion_score"]
            session.cheating_flag = state["cheating_flag"]
            session.flagged_segments_count = state["flagged_count"]

        except Exception as e:
            logger.error(f"Error in _process_buffer: {e}", exc_info=True)

    def stop_processing(self, session_id: str) -> None:
        """Signal processing to stop."""
        self._should_stop[session_id] = True
        logger.info(f"Stop signal sent for session {session_id}")


# Global pipeline instance (initialized in main.py)
pipeline: Optional[AudioPipeline] = None


def get_pipeline() -> AudioPipeline:
    if pipeline is None:
        raise RuntimeError("Pipeline not initialized")
    return pipeline
