"""Audio processing pipeline orchestration."""

import asyncio
import base64
import logging
from collections import deque
from datetime import datetime
from typing import Optional

import numpy as np

from src.core.config import settings
from src.core.models import TranscriptSegment
from src.core.session import session_manager
from src.processing.buffer import AudioBuffer
from src.processing.decision_engine import decision_engine
from src.processing.embedding import EmbeddingProcessor
from src.processing.overlap_detector import OverlapDetector
from src.processing.slm import SLMProcessor
from src.processing.speaker_verification import SpeakerVerifier
from src.processing.stt import STTProcessor
from src.processing.vad import VADProcessor
from src.storage.transcript_store import TranscriptStore

logger = logging.getLogger(__name__)

MIN_CONFIDENCE = 0.5
# Keep a rolling buffer of recent audio for verification (max 30s)
VERIFICATION_AUDIO_BUFFER_SECONDS = 30


class AudioPipeline:
    """Audio processing pipeline: VAD → Buffer → STT → Embedding → SLM → Diarization → Decision Engine."""

    def __init__(
        self,
        vad_processor: VADProcessor,
        stt_processor: STTProcessor,
        embedding_processor: EmbeddingProcessor,
        transcript_store: TranscriptStore,
        slm_processor: Optional[SLMProcessor] = None,
        speaker_verifier: Optional[SpeakerVerifier] = None,
        overlap_detector: Optional[OverlapDetector] = None,
    ):
        self.vad = vad_processor
        self.stt = stt_processor
        self.embedding = embedding_processor
        self.slm = slm_processor
        self.verifier = speaker_verifier
        self.overlap_detector = overlap_detector
        self.transcript_store = transcript_store

        self._should_stop: dict[str, bool] = {}
        # Rolling audio buffer per session for verification
        self._recent_audio: dict[str, deque] = {}

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

        # Init rolling audio buffer for verification
        max_samples = int(VERIFICATION_AUDIO_BUFFER_SECONDS * settings.audio_sample_rate)
        self._recent_audio[session_id] = deque(maxlen=max_samples)

        self._should_stop[session_id] = False
        chunk_count = 0
        last_transcription_time = 0.0
        timestamp = 0.0

        # Start verification loop if verifier is available
        verification_task = None
        if self.verifier is not None and settings.speaker_verification_enabled:
            verification_task = asyncio.create_task(
                self._verification_loop(session_id)
            )

        try:
            while not self._should_stop.get(session_id, False):
                try:
                    audio_data = await asyncio.wait_for(audio_queue.get(), timeout=1.0)

                    chunk_count += 1
                    timestamp = audio_data.get("timestamp", 0.0)
                    audio_bytes = base64.b64decode(audio_data["data"])

                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                    audio_float = audio_array.astype(np.float32) / 32768.0

                    # Accumulate audio for verification (all chunks, not just speech)
                    if session_id in self._recent_audio:
                        self._recent_audio[session_id].extend(audio_float.tolist())

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
            # Stop verification loop
            if verification_task and not verification_task.done():
                verification_task.cancel()

            # Final flush
            if buffer.is_ready():
                await self._process_buffer(session_id, buffer.get_buffer_array(), timestamp)

            # Cleanup rolling buffer
            self._recent_audio.pop(session_id, None)

            logger.info(f"Processing stopped for session {session_id} ({chunk_count} chunks)")

    async def _verification_loop(self, session_id: str) -> None:
        """Periodic speaker verification running in background."""
        logger.info(
            f"[{session_id[:8]}] Verification loop started "
            f"(interval={settings.verification_interval}s)"
        )

        # Wait for first interval before first check
        await asyncio.sleep(settings.verification_interval)

        while not self._should_stop.get(session_id, False):
            try:
                session = session_manager.get_session(session_id)
                if not session:
                    break

                # Get recent audio for verification
                audio_deque = self._recent_audio.get(session_id)
                if audio_deque is None:
                    await asyncio.sleep(settings.verification_interval)
                    continue

                min_samples = int(
                    settings.min_verification_audio_seconds * settings.audio_sample_rate
                )
                if len(audio_deque) < min_samples:
                    logger.debug(
                        f"[{session_id[:8]}] Not enough audio for verification "
                        f"({len(audio_deque)} < {min_samples} samples)"
                    )
                    await asyncio.sleep(settings.verification_interval)
                    continue

                audio = np.array(list(audio_deque), dtype=np.float32)

                # Run verification in executor (blocking)
                loop = asyncio.get_event_loop()
                passed, similarity = await loop.run_in_executor(
                    None,
                    self.verifier.verify,
                    session.student_id,
                    audio,
                    settings.audio_sample_rate,
                )

                # Update session state
                session.last_verification_time = datetime.now()
                session.last_verification_similarity = similarity

                if not passed:
                    session.last_verification_failed = True
                    session.verification_failures_count += 1

                    # Feed into Decision Engine
                    state = decision_engine.process(
                        session_id=session_id,
                        text="[speaker verification failed]",
                        similarity_score=0.0,
                        verification_failed=True,
                    )
                    session.suspicion_score = state["suspicion_score"]
                    session.cheating_flag = state["cheating_flag"]
                    session.flagged_segments_count = state["flagged_count"]

                    logger.warning(
                        f"[{session_id[:8]}] Verification FAILED "
                        f"(similarity={similarity:.3f}, failures={session.verification_failures_count})"
                    )
                else:
                    session.last_verification_failed = False
                    logger.info(
                        f"[{session_id[:8]}] Verification passed (similarity={similarity:.3f})"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in verification loop: {e}", exc_info=True)

            await asyncio.sleep(settings.verification_interval)

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

            # Push STT result to frontend via WebSocket
            session.last_transcript = {
                "text": text,
                "confidence": confidence,
                "timestamp": timestamp,
                "similarity": 0.0,  # updated below after embedding
            }

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
                # Update similarity in transcript push
                if session.last_transcript:
                    session.last_transcript["similarity"] = similarity

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

            # 4. Overlap detection (runs on rolling audio buffer for context)
            overlap_detected = False
            if self.overlap_detector is not None and settings.diarization_enabled:
                audio_deque = self._recent_audio.get(session_id)
                if audio_deque is not None:
                    min_samples = int(
                        settings.min_diarization_audio_seconds * settings.audio_sample_rate
                    )
                    if len(audio_deque) >= min_samples:
                        audio_buffer = np.array(list(audio_deque), dtype=np.float32)
                        loop = asyncio.get_event_loop()
                        overlap_detected, overlap_conf, diar_segments = await loop.run_in_executor(
                            None,
                            self.overlap_detector.detect,
                            audio_buffer,
                            settings.audio_sample_rate,
                        )
                        # Always store latest diarization result for UI display
                        unique_spk = list({s["speaker"] for s in diar_segments})
                        session.last_diarization_result = {
                            "num_speakers": len(unique_spk),
                            "speakers": unique_spk,
                            "segments": diar_segments,
                            "confidence": overlap_conf,
                            "overlap": overlap_detected,
                            "audio_duration": len(audio_buffer) / settings.audio_sample_rate,
                        }
                        if overlap_detected:
                            session.last_overlap_detected = True
                            session.overlap_count += 1
                            logger.info(
                                f"[{session_id[:8]}] Overlap detected "
                                f"(conf={overlap_conf:.2f}, count={session.overlap_count})"
                            )

            # 5. Decision Engine
            state = decision_engine.process(
                session_id=session_id,
                text=text,
                similarity_score=similarity,
                timestamp=timestamp,
                slm_verdict=slm_verdict,
                overlap_detected=overlap_detected,
            )

            # 6. Sync back to session state
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
