"""
Audio processing pipeline orchestration.

Flow (correct order):
    Audio chunks → VAD → rolling buffer (_recent_audio)
        ↓
    _diarization_loop (sliding window 15s, step 7.5s):
        Diarization → identify exam taker (verification embedding or dominant)
        → extract audio for ALL speakers
        → STT each speaker → Embedding → SLM
        → Jaccard dedup (50% window overlap → prevent duplicate alerts)
        → flag cheating if SLM detects exam-related content
"""

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
    """
    Audio processing pipeline.

    When diarization is enabled (default):
        Audio → VAD → rolling buffer → Diarization (window=15s, step=7.5s)
            → identify exam taker (enrollment embedding or dominant-by-time)
            → STT all speakers → Embedding → SLM → alert if related

    When diarization is disabled (fallback):
        Audio → VAD → buffer → STT (every 5s) → Embedding → SLM → alert
    """

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
        # Rolling audio buffer per session for verification + diarization
        self._recent_audio: dict[str, deque] = {}
        # Dedup cache: session_id -> {speaker -> (last_text, last_timestamp)}
        self._stt_dedup: dict[str, dict[str, tuple[str, float]]] = {}

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

        # Diarization loop handles STT when diarization is enabled.
        # When disabled, fall back to the legacy STT-only loop.
        diarization_task = None
        if self.overlap_detector is not None and settings.diarization_enabled:
            diarization_task = asyncio.create_task(
                self._diarization_loop(session_id)
            )
            logger.info("[%s] Mode: Diarization → STT pipeline", session_id[:8])
        else:
            logger.info("[%s] Mode: STT-only pipeline (diarization disabled)", session_id[:8])

        try:
            while not self._should_stop.get(session_id, False):
                try:
                    audio_data = await asyncio.wait_for(audio_queue.get(), timeout=1.0)

                    chunk_count += 1
                    timestamp = audio_data.get("timestamp", 0.0)
                    audio_bytes = base64.b64decode(audio_data["data"])

                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                    audio_float = audio_array.astype(np.float32) / 32768.0

                    # Accumulate rolling audio for diarization + verification
                    if session_id in self._recent_audio:
                        self._recent_audio[session_id].extend(audio_float.tolist())

                    # VAD
                    if not self.vad.process_chunk(audio_float):
                        logger.debug(f"Chunk {chunk_count}: silence, skipping")
                        continue

                    buffer.add_chunk(audio_float, timestamp)

                    # Fallback STT — CHỈ chạy khi DIARIZATION_ENABLED=false.
                    # Nếu diarization bật mà model lỗi, KHÔNG fallback về STT-only.
                    if diarization_task is None and not settings.diarization_enabled:
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
            # Stop background loops
            if verification_task and not verification_task.done():
                verification_task.cancel()
            if diarization_task and not diarization_task.done():
                diarization_task.cancel()

            # Cleanup rolling buffer + dedup cache
            self._recent_audio.pop(session_id, None)
            self._stt_dedup.pop(session_id, None)

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

                # Phase 8: skip verification if student has not enrolled
                if not self.verifier.is_enrolled(session.student_id):
                    logger.debug(
                        "[%s] Student %s chưa enroll — bỏ qua verification",
                        session_id[:8], session.student_id,
                    )
                    await asyncio.sleep(settings.verification_interval)
                    continue

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
                    logger.warning(
                        f"[{session_id[:8]}] Verification FAILED "
                        f"(similarity={similarity:.3f}, failures={session.verification_failures_count})"
                    )
                    # Phase 8: fail ≥ 3 → flag cheating trực tiếp
                    if session.verification_failures_count >= 3:
                        session.cheating_flag = True
                        logger.warning(
                            f"[{session_id[:8]}] ⚠️ 3+ verification failures — cheating_flag set"
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


    @staticmethod
    def _extract_speaker_audio(
        audio: np.ndarray,
        segments: list[dict],
        speaker: str,
        sample_rate: int,
    ) -> np.ndarray:
        """Concatenate audio slices belonging to the given speaker."""
        chunks = []
        for seg in segments:
            if seg.get("speaker") != speaker:
                continue
            lo = int(seg["start"] * sample_rate)
            hi = int(seg["end"] * sample_rate)
            lo = max(0, min(lo, len(audio)))
            hi = max(lo, min(hi, len(audio)))
            if hi > lo:
                chunks.append(audio[lo:hi])
        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks)

    @staticmethod
    def _dominant_speaker(segments: list[dict]) -> str:
        """Return the speaker with the most total speaking time."""
        duration: dict[str, float] = {}
        for seg in segments:
            spk = seg.get("speaker", "")
            if spk:
                duration[spk] = duration.get(spk, 0.0) + (seg["end"] - seg["start"])
        return max(duration, key=lambda k: duration[k]) if duration else ""

    def _identify_exam_taker(
        self,
        segments: list[dict],
        audio_buffer: np.ndarray,
        student_id: str,
        sample_rate: int,
    ) -> tuple[str, bool]:
        """
        Xác định speaker nào là thí sinh:
        - Nếu sinh viên đã enroll: so sánh embedding từng speaker với enrollment
          → speaker có cosine sim cao nhất = thí sinh
        - Nếu chưa enroll: fallback về dominant-by-time

        Returns: (speaker_id, is_verified)
        """
        if not segments:
            return "", False

        if (
            self.verifier is not None
            and self.verifier.is_enrolled(student_id)
        ):
            enrollment = self.verifier.load_enrollment(student_id)
            if enrollment is not None:
                unique_spk = list({s["speaker"] for s in segments})
                best_spk = ""
                best_sim = -1.0

                for spk in unique_spk:
                    spk_audio = self._extract_speaker_audio(
                        audio_buffer, segments, spk, sample_rate
                    )
                    # Need at least 1.5s to get a reliable embedding
                    if len(spk_audio) < int(1.5 * sample_rate):
                        continue
                    try:
                        emb = self.verifier.extract_embedding(spk_audio, sample_rate)
                        sim = max(0.0, min(1.0, float(np.dot(enrollment, emb))))
                        if sim > best_sim:
                            best_sim = sim
                            best_spk = spk
                    except Exception as exc:
                        logger.debug("Embedding failed for %s: %s", spk, exc)

                if best_spk:
                    verified = best_sim >= settings.speaker_verification_threshold
                    logger.info(
                        "[%s] Exam taker: %s (sim=%.3f, verified=%s)",
                        student_id, best_spk, best_sim, verified,
                    )
                    return best_spk, verified

        # Fallback: dominant by speaking time (no enrollment)
        dominant = self._dominant_speaker(segments)
        logger.debug("No enrollment — exam taker by dominant time: %s", dominant)
        return dominant, False

    async def _diarization_loop(self, session_id: str) -> None:
        """
        Sliding-window diarization loop:
          - Window: 15s of audio analysed per run
          - Step:   7.5s (50% overlap → max 7.5s conversation loss instead of 15s)
          1. Diarize → N speakers
          2. Identify exam taker via verification embedding (fallback: dominant-by-time)
          3. STT ALL speakers with role label (thí sinh / người lạ)
          4. Embedding → SLM on every transcript
          5. Jaccard dedup prevents duplicate alerts from overlapping windows
        """
        DIARIZATION_WINDOW = 15.0   # seconds of audio analysed per run
        DIARIZATION_STEP   = 7.5    # run every N seconds (50% overlap)

        logger.info(
            "[%s] Diarization loop started (window=%.0fs, step=%.0fs)",
            session_id[:8], DIARIZATION_WINDOW, DIARIZATION_STEP,
        )

        # Wait for first full window
        await asyncio.sleep(DIARIZATION_WINDOW + 2.0)

        while not self._should_stop.get(session_id, False):
            try:
                session = session_manager.get_session(session_id)
                if not session:
                    break

                audio_deque = self._recent_audio.get(session_id)
                if audio_deque is None:
                    await asyncio.sleep(DIARIZATION_STEP)
                    continue

                window_samples = int(DIARIZATION_WINDOW * settings.audio_sample_rate)
                if len(audio_deque) < window_samples:
                    await asyncio.sleep(DIARIZATION_STEP)
                    continue

                # ── Take last DIARIZATION_WINDOW seconds ─────────────────────
                audio_buffer = np.array(
                    list(audio_deque)[-window_samples:], dtype=np.float32
                )
                buf_timestamp = (datetime.now() - session.started_at).total_seconds()

                logger.info(
                    "[%s] Diarization: running on %.1fs audio",
                    session_id[:8], DIARIZATION_WINDOW,
                )

                # ── Step 1: Diarize ───────────────────────────────────────────
                loop = asyncio.get_event_loop()
                overlap_detected, overlap_conf, diar_segments = await loop.run_in_executor(
                    None,
                    self.overlap_detector.detect,
                    audio_buffer,
                    settings.audio_sample_rate,
                )

                unique_spk = list({s["speaker"] for s in diar_segments})

                # ── Step 2: Identify exam taker ───────────────────────────────
                exam_taker, exam_taker_verified = await loop.run_in_executor(
                    None,
                    self._identify_exam_taker,
                    diar_segments,
                    audio_buffer,
                    session.student_id,
                    settings.audio_sample_rate,
                )

                # Store diarization result for UI
                session.last_diarization_result = {
                    "num_speakers": len(unique_spk),
                    "speakers": unique_spk,
                    "segments": diar_segments,
                    "confidence": overlap_conf,
                    "overlap": overlap_detected,
                    "audio_duration": DIARIZATION_WINDOW,
                    "dominant_speaker": exam_taker,
                }

                if overlap_detected:
                    session.last_overlap_detected = True
                    session.overlap_count += 1
                    logger.warning(
                        "[%s] Overlap: %d speakers (conf=%.2f, total=%d)",
                        session_id[:8], len(unique_spk), overlap_conf, session.overlap_count,
                    )
                else:
                    logger.info(
                        "[%s] Speakers: %s | exam_taker=%s (verified=%s)",
                        session_id[:8],
                        ", ".join(sorted(unique_spk)) if unique_spk else "none",
                        exam_taker, exam_taker_verified,
                    )

                if not diar_segments:
                    await asyncio.sleep(DIARIZATION_STEP)
                    continue

                # ── Step 3: STT ALL speakers with role labels ─────────────────
                for spk in unique_spk:
                    spk_audio = self._extract_speaker_audio(
                        audio_buffer, diar_segments, spk, settings.audio_sample_rate
                    )
                    if len(spk_audio) < int(0.5 * settings.audio_sample_rate):
                        logger.debug(
                            "[%s] Speaker %s audio too short (%.2fs), skipping STT",
                            session_id[:8], spk, len(spk_audio) / settings.audio_sample_rate,
                        )
                        continue

                    role = "thí sinh" if spk == exam_taker else "người lạ"
                    logger.info("[%s] STT speaker %s (%s)", session_id[:8], spk, role)

                    await self._process_buffer(
                        session_id, spk_audio, buf_timestamp,
                        speaker=spk, speaker_role=role,
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in diarization loop: {e}", exc_info=True)

            await asyncio.sleep(DIARIZATION_STEP)


    async def _process_buffer(
        self,
        session_id: str,
        audio: np.ndarray,
        timestamp: float,
        speaker: str = "",
        speaker_role: str = "",
    ) -> None:
        """STT → Dedup → Embedding → SLM → Update session state."""
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

            # 2. Jaccard dedup — tránh duplicate do sliding window 50% overlap
            if speaker:
                session_dedup = self._stt_dedup.setdefault(session_id, {})
                last_text, last_ts = session_dedup.get(speaker, ("", 0.0))
                if last_text and (timestamp - last_ts) < 20.0:
                    words_new = set(text.lower().split())
                    words_old = set(last_text.lower().split())
                    if words_new and words_old:
                        jaccard = len(words_new & words_old) / len(words_new | words_old)
                        if jaccard > 0.70:
                            logger.debug(
                                "[%s] Dedup skip %s (jaccard=%.2f)",
                                session_id[:8], speaker, jaccard,
                            )
                            return
                session_dedup[speaker] = (text, timestamp)

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
                "[%s] STT [%s•%s]: '%s' (conf=%.2f)",
                session_id[:8], speaker, speaker_role, text[:60], confidence,
            )

            # Push STT result to frontend
            session.last_transcript = {
                "text": text,
                "confidence": confidence,
                "timestamp": timestamp,
                "similarity": 0.0,  # updated below after embedding
                "speaker": speaker,
                "speaker_role": speaker_role,
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

            # 4. SLM warning — nếu SLM xác nhận liên quan đến câu hỏi thi → hiện warning
            if slm_verdict:
                session.cheating_flag = True
                session.last_slm_alert = {
                    "text": text,
                    "timestamp": timestamp,
                    "similarity": similarity,
                }
                logger.warning(
                    f"[{session_id[:8]}] ⚠️ SLM ALERT: related to exam question "
                    f"(sim={similarity:.2f}) — '{text[:60]}'"
                )

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
