"""Session-centric transcript storage and retrieval."""

import asyncio
import json
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import aiofiles  # type: ignore[import-untyped]

from src.core.config import settings
from src.core.models import TranscriptSegment

logger = logging.getLogger(__name__)


class TranscriptStore:
    """Store and retrieve transcript segments with session metadata."""

    def __init__(self, storage_dir: Optional[Path] = None):
        """Initialize storage under `storage/sessions`."""
        self.storage_dir = storage_dir or settings.sessions_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.storage_dir / "index.jsonl"
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._session_dir_cache: dict[str, Path] = {}

    def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        if session_id not in self._session_locks:
            self._session_locks[session_id] = asyncio.Lock()
        return self._session_locks[session_id]

    @staticmethod
    def _today_bucket() -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def _legacy_transcript_path(self, session_id: str) -> Path:
        """Fallback for old storage layout: storage/transcripts/{session_id}.json."""
        return settings.transcripts_dir / f"{session_id}.json"

    def _resolve_session_dir(
        self, session_id: str, create_if_missing: bool = False
    ) -> Optional[Path]:
        cached = self._session_dir_cache.get(session_id)
        if cached is not None:
            return cached

        # New layout lookup: storage/sessions/YYYY-MM-DD/{session_id}
        for date_dir in self.storage_dir.glob("*"):
            if not date_dir.is_dir():
                continue
            candidate = date_dir / session_id
            if candidate.exists() and candidate.is_dir():
                self._session_dir_cache[session_id] = candidate
                return candidate

        if not create_if_missing:
            return None

        session_dir = self.storage_dir / self._today_bucket() / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        self._session_dir_cache[session_id] = session_dir
        return session_dir

    async def _append_index_record(self, record: dict) -> None:
        """Append index record for discoverability/audit."""
        async with aiofiles.open(self.index_path, "a", encoding="utf-8") as f:
            await f.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def _rewrite_index(self, records: list[dict]) -> None:
        """Rewrite index file atomically from records."""
        tmp_path = self.index_path.with_suffix(".tmp")
        async with aiofiles.open(tmp_path, "w", encoding="utf-8") as f:
            for rec in records:
                await f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp_path.replace(self.index_path)

    async def init_session(
        self,
        *,
        session_id: str,
        student_id: str,
        exam_id: str,
        exam_question: str,
        started_at: datetime,
    ) -> None:
        """Create session folder and metadata for quick lookup."""
        lock = self._get_session_lock(session_id)
        async with lock:
            session_dir = self._resolve_session_dir(session_id, create_if_missing=True)
            if session_dir is None:
                return

            meta = {
                "session_id": session_id,
                "student_id": student_id,
                "exam_id": exam_id,
                "exam_question": exam_question,
                "status": "active",
                "started_at": started_at.isoformat(),
                "ended_at": None,
                "paths": {
                    "transcript": str(session_dir / "transcript.jsonl"),
                    "events": str(session_dir / "events.jsonl"),
                    "report": str(session_dir / "report.json"),
                },
            }
            async with aiofiles.open(session_dir / "meta.json", "w", encoding="utf-8") as f:
                await f.write(json.dumps(meta, ensure_ascii=False, indent=2))

            await self._append_index_record(
                {
                    "event": "session_started",
                    "timestamp": datetime.now().isoformat(),
                    "session_id": session_id,
                    "student_id": student_id,
                    "exam_id": exam_id,
                    "session_dir": str(session_dir),
                }
            )

    async def finalize_session(
        self,
        *,
        session_id: str,
        status: str,
        ended_at: datetime,
        cheating_detected: bool,
    ) -> None:
        """Update meta and index when a session is stopped."""
        lock = self._get_session_lock(session_id)
        async with lock:
            session_dir = self._resolve_session_dir(session_id, create_if_missing=False)
            if session_dir is None:
                return

            meta_path = session_dir / "meta.json"
            meta: dict = {}
            if meta_path.exists():
                async with aiofiles.open(meta_path, "r", encoding="utf-8") as f:
                    raw = (await f.read()).strip()
                if raw:
                    meta = json.loads(raw)

            meta["status"] = status
            meta["ended_at"] = ended_at.isoformat()
            meta["cheating_detected"] = cheating_detected
            async with aiofiles.open(meta_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(meta, ensure_ascii=False, indent=2))

            await self._append_index_record(
                {
                    "event": "session_completed",
                    "timestamp": datetime.now().isoformat(),
                    "session_id": session_id,
                    "status": status,
                    "cheating_detected": cheating_detected,
                    "session_dir": str(session_dir),
                }
            )

    async def save_report(self, session_id: str, report: dict) -> None:
        """Persist latest report snapshot for the session."""
        lock = self._get_session_lock(session_id)
        async with lock:
            session_dir = self._resolve_session_dir(session_id, create_if_missing=True)
            if session_dir is None:
                raise RuntimeError(f"Failed to resolve session dir for {session_id}")

            report_path = session_dir / "report.json"
            async with aiofiles.open(report_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(report, ensure_ascii=False, indent=2))

    async def save_segment(self, session_id: str, segment: TranscriptSegment) -> None:
        """Append transcript segment as JSONL record."""
        lock = self._get_session_lock(session_id)
        async with lock:
            session_dir = self._resolve_session_dir(session_id, create_if_missing=True)
            if session_dir is None:
                raise RuntimeError(f"Failed to resolve session dir for {session_id}")

            transcript_path = session_dir / "transcript.jsonl"
            async with aiofiles.open(transcript_path, "a", encoding="utf-8") as f:
                await f.write(json.dumps(segment.model_dump(), ensure_ascii=False) + "\n")

    async def _load_new_segments_unlocked(self, session_id: str) -> list[TranscriptSegment]:
        session_dir = self._resolve_session_dir(session_id, create_if_missing=False)
        if session_dir is None:
            return []

        transcript_path = session_dir / "transcript.jsonl"
        if not transcript_path.exists():
            return []

        async with aiofiles.open(transcript_path, "r", encoding="utf-8") as f:
            content = await f.read()

        segments: list[TranscriptSegment] = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                segments.append(TranscriptSegment(**json.loads(line)))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skip invalid transcript record for %s: %s", session_id, exc)
        return segments

    async def _load_legacy_segments_unlocked(
        self, session_id: str
    ) -> list[TranscriptSegment]:
        file_path = self._legacy_transcript_path(session_id)
        if not file_path.exists():
            return []

        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            content = (await f.read()).strip()
        if not content:
            return []

        data = json.loads(content)
        return [TranscriptSegment(**seg) for seg in data.get("segments", [])]

    async def load_segments(self, session_id: str) -> list[TranscriptSegment]:
        """Load transcript segments for session (new layout first, legacy fallback)."""
        lock = self._get_session_lock(session_id)
        async with lock:
            segments = await self._load_new_segments_unlocked(session_id)
            if segments:
                return segments
            return await self._load_legacy_segments_unlocked(session_id)

    async def get_full_transcript(self, session_id: str) -> str:
        segments = await self.load_segments(session_id)
        return " ".join(seg.text for seg in segments)

    async def delete_transcript(self, session_id: str) -> bool:
        """Delete session transcript storage (new + legacy)."""
        lock = self._get_session_lock(session_id)
        async with lock:
            deleted = False

            session_dir = self._resolve_session_dir(session_id, create_if_missing=False)
            if session_dir and session_dir.exists():
                shutil.rmtree(session_dir)
                deleted = True

            legacy = self._legacy_transcript_path(session_id)
            if legacy.exists():
                legacy.unlink()
                deleted = True

            self._session_locks.pop(session_id, None)
            self._session_dir_cache.pop(session_id, None)

            if deleted:
                await self._append_index_record(
                    {
                        "event": "session_deleted",
                        "timestamp": datetime.now().isoformat(),
                        "session_id": session_id,
                    }
                )
            return deleted

    async def query_sessions(
        self,
        *,
        student_id: Optional[str] = None,
        exam_id: Optional[str] = None,
        status: Optional[str] = None,
        started_from: Optional[datetime] = None,
        started_to: Optional[datetime] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        """Query stored sessions by student/exam/time/status with pagination."""
        rows: list[dict] = []
        for date_dir in self.storage_dir.glob("*"):
            if not date_dir.is_dir():
                continue
            for session_dir in date_dir.glob("*"):
                if not session_dir.is_dir():
                    continue
                meta_path = session_dir / "meta.json"
                if not meta_path.exists():
                    continue
                try:
                    async with aiofiles.open(meta_path, "r", encoding="utf-8") as f:
                        raw = (await f.read()).strip()
                    if not raw:
                        continue
                    meta = json.loads(raw)
                    row = {
                        "session_id": meta.get("session_id", session_dir.name),
                        "student_id": meta.get("student_id", ""),
                        "exam_id": meta.get("exam_id", ""),
                        "status": meta.get("status", ""),
                        "started_at": meta.get("started_at"),
                        "ended_at": meta.get("ended_at"),
                        "cheating_detected": meta.get("cheating_detected", False),
                        "session_dir": str(session_dir),
                    }
                    rows.append(row)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Skip invalid meta file %s: %s", meta_path, exc)

        def _in_time_range(started_at_raw: Optional[str]) -> bool:
            if started_at_raw is None:
                return False
            try:
                started_at = datetime.fromisoformat(started_at_raw)
            except ValueError:
                return False
            if started_from and started_at < started_from:
                return False
            if started_to and started_at > started_to:
                return False
            return True

        filtered = []
        for row in rows:
            if student_id and row["student_id"] != student_id:
                continue
            if exam_id and row["exam_id"] != exam_id:
                continue
            if status and row["status"] != status:
                continue
            if started_from or started_to:
                if not _in_time_range(row["started_at"]):
                    continue
            filtered.append(row)

        filtered.sort(key=lambda r: (r.get("started_at") or ""), reverse=True)
        total = len(filtered)
        items = filtered[offset : offset + limit]
        return {"items": items, "total": total, "limit": limit, "offset": offset}

    async def apply_retention(self, retention_days: Optional[int] = None) -> dict:
        """Delete expired session folders and prune index records."""
        days = settings.storage_retention_days if retention_days is None else retention_days
        if days <= 0:
            return {
                "enabled": False,
                "retention_days": days,
                "deleted_session_dirs": 0,
                "kept_index_records": 0,
                "removed_index_records": 0,
            }

        cutoff = datetime.now() - timedelta(days=days)
        deleted_dirs = 0

        for date_dir in self.storage_dir.glob("*"):
            if not date_dir.is_dir():
                continue
            try:
                date_bucket = datetime.strptime(date_dir.name, "%Y-%m-%d")
            except ValueError:
                continue
            if date_bucket.date() < cutoff.date():
                shutil.rmtree(date_dir, ignore_errors=True)
                deleted_dirs += 1

        kept_records: list[dict] = []
        removed_records = 0
        if self.index_path.exists():
            async with aiofiles.open(self.index_path, "r", encoding="utf-8") as f:
                lines = (await f.read()).splitlines()

            for line in lines:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    removed_records += 1
                    continue

                ts_raw = rec.get("timestamp")
                session_dir = rec.get("session_dir")
                keep = True
                if ts_raw:
                    try:
                        ts = datetime.fromisoformat(ts_raw)
                        if ts < cutoff:
                            keep = False
                    except ValueError:
                        keep = False
                if session_dir and not Path(session_dir).exists():
                    keep = False

                if keep:
                    kept_records.append(rec)
                else:
                    removed_records += 1

            await self._rewrite_index(kept_records)

        return {
            "enabled": True,
            "retention_days": days,
            "deleted_session_dirs": deleted_dirs,
            "kept_index_records": len(kept_records),
            "removed_index_records": removed_records,
        }

    def list_sessions(self) -> list[str]:
        """List all session IDs from new layout."""
        session_ids: list[str] = []
        try:
            for date_dir in self.storage_dir.glob("*"):
                if not date_dir.is_dir():
                    continue
                for session_dir in date_dir.glob("*"):
                    if session_dir.is_dir():
                        session_ids.append(session_dir.name)
            return sorted(set(session_ids))
        except Exception as e:  # noqa: BLE001
            logger.error("Error listing sessions: %s", e, exc_info=True)
            return []
