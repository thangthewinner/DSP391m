# DSP391m - AI Exam Proctoring (MVP)

## Overview
DSP391m is a local, real-time exam proctoring system for spoken Vietnamese exams.

The system monitors microphone audio during an exam session and applies these core rules:
- Voice enrollment is required before starting an exam.
- Candidate identity is re-verified periodically during the session.
- If verification fails repeatedly (default: 3 times), the session is terminated.
- Multi-speaker overlap is monitored, but overlap alone is not cheating.
- If exam-related content is detected from any speaker, it is flagged as cheating.

### Main components
- **Backend**: FastAPI (`src/api`)
- **Frontend**: Streamlit (`frontend/app.py`)
- **Audio pipeline**: VAD -> STT -> similarity -> optional SLM reasoning (`src/processing/pipeline.py`)
- **Storage**: session-centric logs/reports under `storage/sessions/`

## Setup

### 1. Prerequisites
- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv)

### 2. Install dependencies
```bash
uv sync
```

### 3. Configure environment
```bash
cp .env.example .env
```
Edit `.env` as needed (device, model paths, feature toggles). Important keys include:
- `TORCH_DEVICE`
- `SPEAKER_VERIFICATION_ENABLED`
- `VERIFICATION_INTERVAL`
- `VERIFICATION_MAX_FAILURES`
- `DIARIZATION_ENABLED`
- `SLM_ENABLED`
- `EVENT_SCHEMA_VERSION`
- `REPORT_SCHEMA_VERSION`

### 4. Download models
```bash
uv run python scripts/download_models.py --all
```
Or download selectively (for example `--stt`, `--slm`, `--diarization`).

By default, STT uses Whisper Large v3 (`large-v3`) via faster-whisper.

If you want to override STT model path, set in `.env`:
```bash
STT_MODEL_PATH=./models/stt/whisper-large-v3-ct2
```

## Run

### Terminal 1: start backend
```bash
uv run uvicorn src.api.main:app --reload
```

### Terminal 2: start frontend
```bash
uv run streamlit run frontend/app.py
```

Open `http://localhost:8501` and run the full flow:
1. Enroll voice
2. Start session
3. Speak/send audio
4. Stop session
5. Review report

## Useful API endpoints
- `POST /api/enroll/{student_id}`
- `POST /api/exam/start`
- `POST /api/exam/stop/{session_id}`
- `GET /api/exam/status/{session_id}`
- `GET /api/exam/events/{session_id}`
- `GET /api/exam/report/{session_id}`
- `GET /api/exam/history`
- `WS /ws/audio/{session_id}`
