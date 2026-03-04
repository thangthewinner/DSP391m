# AI Exam Proctoring System

Real-time verbal cheating detection for online exams. Captures microphone audio, transcribes Vietnamese speech, and flags semantically suspicious answers using a multi-layer AI pipeline.

**Stack:** FastAPI · Streamlit · Silero VAD · PhoWhisper · Vietnamese SBERT · Qwen2.5-3B · ECAPA-TDNN · NeMo Sortformer

---

## Overview

| Layer | Model | Purpose |
|---|---|---|
| VAD | Silero VAD v4 | Filter silence |
| STT | PhoWhisper-small (CTranslate2) | Vietnamese speech → text |
| Embedding | Vietnamese SBERT | Semantic similarity to exam content |
| SLM | Qwen2.5-3B-Instruct (GGUF) | Confirm cheating via reasoning |
| Speaker ID | SpeechBrain ECAPA-TDNN | Verify student identity |
| Diarization | NeMo Streaming Sortformer | Detect multiple speakers |
| Decision | Suspicion score engine | Decay + threshold scoring |

Audio flows: `Browser mic → WebSocket → VAD → STT → Embedding → SLM → Decision Engine`

---

## Setup & Run

**Requirements:** Python 3.11, `uv`

```bash
# 1. Install dependencies
uv sync

# 2. Configure
cp .env.example .env   # edit TORCH_DEVICE, enable/disable modules

# 3. Download models
uv run python scripts/download_models.py --all
# or selectively: --stt --slm --diarization

# 4. Start backend
uv run uvicorn src.api.main:app --reload

# 5. Start frontend (new terminal)
uv run streamlit run frontend/app.py
```

Open `http://localhost:8501` → select exam → click **Bắt đầu giám sát** → allow mic.

### Key `.env` options

```env
TORCH_DEVICE=cpu              # or cuda
SLM_ENABLED=true
SPEAKER_VERIFICATION_ENABLED=true
DIARIZATION_ENABLED=true
DIARIZATION_MODEL_PATH=./models/diarization/diar_streaming_sortformer_4spk-v2.nemo
```

---

## Architecture

```
Browser
  └─ JS getUserMedia → PCM int16 → base64 JSON
       └─ WebSocket /ws/audio/{session_id}
            └─ FastAPI AudioPipeline
                 ├─ VAD (Silero)          — drop silence
                 ├─ AudioBuffer (10s)     — sliding window
                 ├─ STT (PhoWhisper)      — transcribe
                 ├─ Embedding (SBERT)     — similarity score
                 ├─ SLM (Qwen2.5)        — reasoning verdict
                 ├─ Speaker Verifier      — identity check (async loop)
                 ├─ Diarization (NeMo)    — overlap detection
                 └─ Decision Engine       — suspicion score + decay
                      └─ WebSocket push → Streamlit log panel
```

**Suspicion scoring:**
- Similarity ≥ 0.75 + SLM=YES → +3 pts
- Similarity ≥ 0.60 → +1 pt
- Speaker overlap → +1 pt
- Verification fail → +2 pts
- Score decays 10%/chunk; threshold = 10 → cheating flag

---

## Project Structure

```
.
├── frontend/
│   └── app.py              # Streamlit UI (mic capture, log panel, report)
├── src/
│   ├── api/
│   │   ├── main.py         # FastAPI app + lifespan (model loading)
│   │   ├── routes/         # REST endpoints (start/stop/status/report)
│   │   └── websocket/
│   │       └── audio_handler.py  # WebSocket audio stream handler
│   ├── core/
│   │   ├── config.py       # Pydantic settings (from .env)
│   │   ├── models.py       # Pydantic data models
│   │   └── session.py      # In-memory session manager
│   └── processing/
│       ├── pipeline.py     # Orchestrates full audio pipeline
│       ├── vad.py          # Silero VAD (frame-by-frame)
│       ├── buffer.py       # Sliding audio buffer
│       ├── stt.py          # PhoWhisper transcription
│       ├── embedding.py    # Vietnamese SBERT similarity
│       ├── slm.py          # Qwen2.5 GGUF reasoning
│       ├── speaker_verification.py  # ECAPA-TDNN identity
│       ├── overlap_detector.py      # NeMo Sortformer diarization
│       └── decision_engine.py       # Scoring + decay logic
├── exams/                  # Exam content (.txt) for semantic matching
├── models/                 # Downloaded model weights (gitignored)
├── storage/                # Transcripts + enrollments (gitignored)
├── scripts/
│   ├── download_models.py  # Download all models from HuggingFace
│   └── convert_phowhisper.py  # Convert to CTranslate2 format
└── .env                    # Runtime configuration
```
