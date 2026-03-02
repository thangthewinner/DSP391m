# DSP391m — AI Exam Proctoring System

Real-time verbal cheating detection for online exams using Vietnamese speech recognition.

## Architecture

```
Microphone → WebSocket → VAD → Buffer → STT → Decision Engine → Alert
```

**Phase 1 (done):** Audio pipeline — VAD + Buffer + STT  
**Phase 2:** Text embedding similarity (Vietnamese SBERT)  
**Phase 3:** SLM reasoning (Qwen2.5-7B)  
**Phase 4:** Decision engine with suspicion scoring  
**Phase 5:** Speaker verification (ECAPA-TDNN)  
**Phase 6:** Speaker diarization (pyannote)

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- ~3GB disk space for models

## Setup

### 1. Install dependencies

```bash
uv sync
```

For GPU (CUDA 12.6):
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env if needed (default: CPU mode)
```

### 3. Download & prepare models

```bash
# Download PhoWhisper (926MB)
uv run python scripts/download_models.py

# Convert to CTranslate2 format (required by faster-whisper)
uv run ct2-transformers-converter \
  --model models/stt/phowhisper-small \
  --output_dir models/stt/phowhisper-small-ct2 \
  --quantization int8 \
  --copy_files tokenizer.json preprocessor_config.json
```

> Silero VAD is cached automatically by `torch.hub` — no extra step needed.

### 4. Start server

```bash
uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Server ready when you see:
```
✓ Silero VAD model loaded successfully
✓ PhoWhisper model loaded successfully
Application startup complete.
```

## API

Interactive docs: **http://localhost:8000/docs**

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/exam/start` | Start exam session |
| `POST` | `/api/exam/stop/{session_id}` | Stop session |
| `GET` | `/api/exam/status/{session_id}` | Session status |
| `GET` | `/api/exam/report/{session_id}` | Full report + transcript |
| `GET` | `/api/exam/sessions` | List active sessions |
| `DELETE` | `/api/exam/session/{session_id}` | Delete session |
| `WS` | `/ws/audio/{session_id}` | Audio stream |

### Quick test

```bash
# Start a session
curl -X POST http://localhost:8000/api/exam/start \
  -H "Content-Type: application/json" \
  -d '{"student_id": "student_001", "exam_id": "exam_001"}'

# WebSocket audio format
{
  "type": "audio_chunk",
  "data": "<base64 PCM int16>",
  "sample_rate": 16000,
  "timestamp": 1.0
}
```

## Project Structure

```
src/
├── api/
│   ├── main.py              # FastAPI app, model loading
│   ├── routes/
│   │   ├── exam.py          # Session management endpoints
│   │   ├── health.py        # Health check
│   │   └── enrollment.py    # Speaker enrollment (Phase 5 stub)
│   └── websocket/
│       └── audio_handler.py # WebSocket audio stream
├── core/
│   ├── config.py            # Settings (pydantic-settings)
│   ├── models.py            # Pydantic data models
│   └── session.py           # In-memory session manager
├── processing/
│   ├── vad.py               # Silero VAD
│   ├── stt.py               # PhoWhisper (faster-whisper)
│   ├── buffer.py            # Sliding window audio buffer
│   └── pipeline.py          # VAD → Buffer → STT orchestration
└── storage/
    └── transcript_store.py  # Async JSON transcript storage

models/
├── stt/phowhisper-small/        # PyTorch source model
└── stt/phowhisper-small-ct2/    # CTranslate2 model (used at runtime)

storage/transcripts/             # Session transcripts (JSON)
```

## Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `TORCH_DEVICE` | `cpu` | `cpu` or `cuda` |
| `STT_MODEL_PATH` | *(auto)* | Override CT2 model path |
| `VAD_THRESHOLD` | `0.5` | Speech detection sensitivity |
| `AUDIO_SAMPLE_RATE` | `16000` | Hz |
| `BUFFER_SIZE` | `10.0` | Seconds of audio to buffer |
| `STORAGE_ROOT` | `./storage` | Transcript storage directory |

## GPU Setup

If you have an NVIDIA GPU:

```bash
# 1. Install PyTorch with CUDA
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 2. Convert model with float16
uv run ct2-transformers-converter \
  --model models/stt/phowhisper-small \
  --output_dir models/stt/phowhisper-small-ct2-gpu \
  --quantization float16

# 3. Update .env
TORCH_DEVICE=cuda
STT_MODEL_PATH=./models/stt/phowhisper-small-ct2-gpu
```

## Running Tests

```bash
uv run pytest tests/ -v
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API | FastAPI + WebSocket |
| VAD | Silero VAD v4 |
| STT | PhoWhisper-small (faster-whisper) |
| Embedding | Vietnamese SBERT *(Phase 2)* |
| SLM | Qwen2.5-7B-Instruct *(Phase 3)* |
| Speaker ID | ECAPA-TDNN *(Phase 5)* |
