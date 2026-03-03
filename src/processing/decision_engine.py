"""
Decision Engine — stateful suspicion scoring with sliding window decay.

Scoring weights:
  similarity LOW  (≥0.60) → +1.0 pt
  similarity HIGH (≥0.75) → +2.0 pt
  SLM YES (Phase 3)        → +2.0 pt
  speaker overlap          → +1.0 pt  (Phase 6)
  verification failure     → +3.0 pt  (Phase 5)

Cheating threshold: 10.0 pts
Decay: 0.9 per second (exponential)
Window: 30 seconds
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# --- Thresholds (tunable via config) ---
SIMILARITY_LOW = 0.60
SIMILARITY_HIGH = 0.75
CHEATING_THRESHOLD = 10.0
DECAY_FACTOR = 0.9       # per second
WINDOW_SIZE = 30.0       # seconds
MAX_EXPECTED_SCORE = 20.0


@dataclass
class SuspicionEvent:
    timestamp: float
    text: str
    similarity_score: float
    points_added: float
    slm_verdict: bool = False
    overlap_detected: bool = False
    verification_failed: bool = False


@dataclass
class DecisionEngineState:
    suspicion_score: float = 0.0
    max_suspicion_score: float = 0.0
    cheating_flag: bool = False
    last_update_time: float = field(default_factory=time.time)
    events: deque = field(default_factory=lambda: deque(maxlen=100))
    flagged_events: list = field(default_factory=list)


class DecisionEngine:
    """Stateful decision engine that accumulates suspicion score over time."""

    def __init__(
        self,
        similarity_low: float = SIMILARITY_LOW,
        similarity_high: float = SIMILARITY_HIGH,
        cheating_threshold: float = CHEATING_THRESHOLD,
        decay_factor: float = DECAY_FACTOR,
        window_size: float = WINDOW_SIZE,
    ):
        self.similarity_low = similarity_low
        self.similarity_high = similarity_high
        self.cheating_threshold = cheating_threshold
        self.decay_factor = decay_factor
        self.window_size = window_size

        self._states: dict[str, DecisionEngineState] = {}

    def get_or_create_state(self, session_id: str) -> DecisionEngineState:
        if session_id not in self._states:
            self._states[session_id] = DecisionEngineState()
        return self._states[session_id]

    def _apply_decay(self, state: DecisionEngineState, current_time: float) -> None:
        """Apply exponential decay based on elapsed time since last update."""
        elapsed = current_time - state.last_update_time
        if elapsed > 0:
            state.suspicion_score *= (self.decay_factor ** elapsed)
            state.last_update_time = current_time

    def _prune_window(self, state: DecisionEngineState, current_time: float) -> None:
        """Remove events older than window_size."""
        cutoff = current_time - self.window_size
        while state.events and state.events[0].timestamp < cutoff:
            state.events.popleft()

    def _score_similarity(self, similarity: float) -> float:
        if similarity >= self.similarity_high:
            return 2.0
        elif similarity >= self.similarity_low:
            return 1.0
        return 0.0

    def _score_slm(self, slm_verdict: bool) -> float:
        return 2.0 if slm_verdict else 0.0

    def process(
        self,
        session_id: str,
        text: str,
        similarity_score: float,
        timestamp: Optional[float] = None,
        slm_verdict: bool = False,
        overlap_detected: bool = False,
        verification_failed: bool = False,
    ) -> dict:
        """
        Process a new signal and update suspicion score.

        Args:
            session_id: Session identifier
            text: Transcribed text
            similarity_score: Cosine similarity to exam question (0-1)
            timestamp: Event timestamp (defaults to now)
            slm_verdict: SLM reasoning result — True = related (Phase 3)
            overlap_detected: Multiple speakers detected (Phase 6)
            verification_failed: Speaker identity mismatch (Phase 5)

        Returns:
            dict with current state snapshot
        """
        current_time = timestamp or time.time()
        state = self.get_or_create_state(session_id)

        # Apply decay since last update
        self._apply_decay(state, current_time)

        # Prune old events
        self._prune_window(state, current_time)

        # Calculate points
        points = self._score_similarity(similarity_score)
        points += self._score_slm(slm_verdict)
        points += 1.0 if overlap_detected else 0.0
        points += 3.0 if verification_failed else 0.0

        # Update score
        if points > 0:
            state.suspicion_score += points
            state.max_suspicion_score = max(state.max_suspicion_score, state.suspicion_score)

            event = SuspicionEvent(
                timestamp=current_time,
                text=text,
                similarity_score=similarity_score,
                points_added=points,
                slm_verdict=slm_verdict,
                overlap_detected=overlap_detected,
                verification_failed=verification_failed,
            )
            state.events.append(event)
            state.flagged_events.append(event)

            logger.info(
                f"[{session_id[:8]}] suspicion +{points:.1f} → {state.suspicion_score:.2f} "
                f"(similarity={similarity_score:.2f}, slm={slm_verdict}, text='{text[:40]}')"
            )

        # Check cheating threshold
        if not state.cheating_flag and state.suspicion_score >= self.cheating_threshold:
            state.cheating_flag = True
            logger.warning(
                f"[{session_id[:8]}] ⚠️  CHEATING DETECTED! "
                f"score={state.suspicion_score:.2f} ≥ threshold={self.cheating_threshold}"
            )

        return self.snapshot(session_id)

    def snapshot(self, session_id: str) -> dict:
        """Return current state as a dict."""
        state = self.get_or_create_state(session_id)
        confidence = min(1.0, state.max_suspicion_score / MAX_EXPECTED_SCORE)

        return {
            "suspicion_score": round(state.suspicion_score, 2),
            "max_suspicion_score": round(state.max_suspicion_score, 2),
            "cheating_flag": state.cheating_flag,
            "confidence": round(confidence, 2),
            "flagged_count": len(state.flagged_events),
        }

    def reset(self, session_id: str) -> None:
        """Reset state for a session."""
        if session_id in self._states:
            del self._states[session_id]

    def get_report(self, session_id: str) -> dict:
        """Generate full decision report for a session."""
        state = self.get_or_create_state(session_id)
        confidence = min(1.0, state.max_suspicion_score / MAX_EXPECTED_SCORE)

        # Build rationale
        rationale_parts = []
        high_sim = sum(1 for e in state.flagged_events if e.similarity_score >= self.similarity_high)
        low_sim = sum(1 for e in state.flagged_events if self.similarity_low <= e.similarity_score < self.similarity_high)
        slm_confirmed = sum(1 for e in state.flagged_events if e.slm_verdict)
        overlaps = sum(1 for e in state.flagged_events if e.overlap_detected)
        verif_fails = sum(1 for e in state.flagged_events if e.verification_failed)

        if high_sim:
            rationale_parts.append(f"{high_sim} đoạn có độ tương đồng cao với câu hỏi")
        if low_sim:
            rationale_parts.append(f"{low_sim} đoạn có độ tương đồng trung bình")
        if slm_confirmed:
            rationale_parts.append(f"{slm_confirmed} đoạn được AI xác nhận liên quan đến câu hỏi thi")
        if overlaps:
            rationale_parts.append(f"{overlaps} lần phát hiện nhiều người nói")
        if verif_fails:
            rationale_parts.append(f"{verif_fails} lần xác minh danh tính thất bại")

        if state.cheating_flag:
            rationale = "Phát hiện gian lận: " + "; ".join(rationale_parts) + "."
        elif rationale_parts:
            rationale = "Nghi ngờ nhưng chưa đủ bằng chứng: " + "; ".join(rationale_parts) + "."
        else:
            rationale = "Không phát hiện hoạt động đáng ngờ."

        return {
            "cheating_detected": state.cheating_flag,
            "suspicion_score": round(state.suspicion_score, 2),
            "max_suspicion_score": round(state.max_suspicion_score, 2),
            "confidence": round(confidence, 2),
            "flagged_segments_count": len(state.flagged_events),
            "rationale": rationale,
            "flagged_segments": [
                {
                    "timestamp": e.timestamp,
                    "text": e.text,
                    "similarity_score": round(e.similarity_score, 3),
                    "slm_verdict": e.slm_verdict,
                    "points_added": e.points_added,
                }
                for e in state.flagged_events
            ],
        }


# Global instance
decision_engine = DecisionEngine()
