# Speech to Text + SLM Verification Feature

## 📖 Tổng Quan

Feature này kết hợp **Speech-to-Text (STT)** và **Small Language Model (SLM)** để phát hiện gian lận trong thi trực tuyến thông qua phân tích nội dung giọng nói.

### Pipeline Xử Lý

```
Audio Input → VAD → STT → Embedding → SLM → Cheating Detection
```

1. **VAD (Voice Activity Detection)** - Silero VAD
   - Lọc bỏ đoạn câm
   - Chỉ xử lý phần có tiếng nói

2. **STT (Speech to Text)** - PhoWhisper
   - Chuyển giọng nói tiếng Việt → văn bản
   - Model: vinai/PhoWhisper-small (CTranslate2)

3. **Embedding** - Vietnamese SBERT
   - Tính độ tương đồng ngữ nghĩa với câu hỏi thi
   - Cosine similarity score [0-1]

4. **SLM (Small Language Model)** - Qwen2.5-3B-Instruct
   - Xác nhận xem nội dung có liên quan đến đề thi không
   - Output: YES/NO

5. **Decision Logic**
   - Similarity ≥ 0.60 → Chạy SLM
   - SLM = YES → 🚨 **CHEATING DETECTED**

## 🚀 Cài Đặt

### 1. Cài Dependencies

```bash
uv sync
```

### 2. Download Models

```bash
# Download tất cả models
uv run python scripts/download_models.py --all

# Hoặc download riêng lẻ:
uv run python scripts/download_models.py --stt    # PhoWhisper
uv run python scripts/download_models.py --slm    # Qwen2.5-3B
```

### 3. Cấu Hình .env

```bash
# Copy template
cp .env.example .env

# Edit .env
TORCH_DEVICE=cpu                    # hoặc cuda nếu có GPU
SLM_ENABLED=true
SLM_MODEL_PATH=./models/slm/qwen2.5-3b-instruct-q4_k_m.gguf
SLM_N_GPU_LAYERS=0                  # 0=CPU, -1=all layers on GPU
```

## 📝 Sử Dụng

### Demo Script: STT + SLM

#### 1. Test với file audio

```bash
uv run python scripts/demo_stt_slm.py \
    --audio test_audio.wav \
    --question "Giải thích khái niệm đệ quy trong Python"
```

#### 2. Test với microphone (record live)

```bash
uv run python scripts/demo_stt_slm.py \
    --record 10 \
    --question "Viết câu lệnh SQL JOIN hai bảng"
```

#### 3. Test với text (không cần audio)

```bash
uv run python scripts/demo_stt_slm.py \
    --text "Đệ quy là hàm tự gọi chính nó" \
    --question "Giải thích đệ quy"
```

### Quick Test: Text-only (Không Cần Audio)

Test nhanh chỉ với Embedding + SLM:

```bash
uv run python scripts/test_slm_only.py
```

Script này chạy 4 test cases:
- ✓ Test 1: Clear cheating (expected: YES)
- ✓ Test 2: Not related (expected: NO)
- ✓ Test 3: Topic related (expected: NO)
- ✓ Test 4: Discussing exam (expected: YES)

## 🎯 Ví Dụ Thực Tế

### Ví Dụ 1: Phát Hiện Gian Lận

**Câu hỏi thi:**
```
"Viết hàm đệ quy tính giai thừa trong Python"
```

**Thí sinh nói:**
```
"Hàm factorial đệ quy thì mình định nghĩa def factorial(n),
nếu n bằng 0 thì return 1, còn không thì return n nhân factorial(n-1)"
```

**Kết quả:**
```
✓ STT Confidence: 0.923
✓ Similarity:     0.847
✓ SLM Verdict:    YES
🚨 CHEATING DETECTED
```

### Ví Dụ 2: Không Phát Hiện Gian Lận

**Câu hỏi thi:**
```
"Giải thích OOP trong Python"
```

**Thí sinh nói:**
```
"Hôm nay trời đẹp quá, tôi muốn đi chơi công viên"
```

**Kết quả:**
```
✓ STT Confidence: 0.891
✓ Similarity:     0.124
✓ SLM Verdict:    SKIPPED (low similarity)
✓ Not cheating
```

## 📊 Output Format

Demo script trả về dictionary:

```python
{
    "transcript": "Văn bản đã chuyển đổi",
    "confidence": 0.923,        # STT confidence [0-1]
    "similarity": 0.847,        # Embedding similarity [0-1]
    "slm_verdict": True,        # SLM: YES=True, NO=False
    "cheating_detected": True   # Final decision
}
```

## 🔧 Tùy Chỉnh Threshold

Điều chỉnh trong `.env`:

```env
# Similarity thresholds
SIMILARITY_THRESHOLD_LOW=0.60    # Ngưỡng để chạy SLM
SIMILARITY_THRESHOLD_HIGH=0.75   # Ngưỡng cảnh báo cao

# VAD threshold
VAD_THRESHOLD=0.5                # 0-1, càng cao càng strict

# SLM settings
SLM_MAX_TOKENS=4                 # Output tokens (YES/NO = 3-4 tokens)
SLM_CONTEXT_LENGTH=512           # Context window
```

## 🧪 Testing

### Unit Tests

```bash
# Test STT
pytest tests/test_stt.py -v

# Test VAD
pytest tests/test_vad.py -v

# Test integration
pytest tests/test_integration.py -v
```

### Manual Testing trong Code

```python
from src.processing.stt import STTProcessor
from src.processing.embedding import EmbeddingProcessor
from src.processing.slm import SLMProcessor

# Load models
stt = STTProcessor(device="cpu")
stt.load_model()

embedding = EmbeddingProcessor(device="cpu")
embedding.load_model()

slm = SLMProcessor(model_path=Path("models/slm/qwen2.5-3b-instruct-q4_k_m.gguf"))
slm.load_model()

# Transcribe audio
result = stt.transcribe(audio_array, sample_rate=16000)
transcript = result["text"]

# Check similarity
question_emb = embedding.embed(exam_question)
transcript_emb = embedding.embed(transcript)
similarity = embedding.similarity(question_emb, transcript_emb)

# SLM verification
if similarity >= 0.60:
    verdict = slm.predict(exam_question, transcript)
    if verdict:
        print("🚨 CHEATING DETECTED")
```

## 📈 Performance

**Thời gian xử lý (CPU - Intel i7):**
- VAD: ~0.1s per chunk
- STT: ~2-3s per 10s audio
- Embedding: ~0.2s per text
- SLM: ~1-2s per inference

**Thời gian xử lý (GPU - CUDA):**
- STT: ~0.5-1s per 10s audio
- Embedding: ~0.1s per text
- SLM: ~0.3-0.5s per inference

## 🐛 Troubleshooting

### Error: "SLM model not found"

```bash
uv run python scripts/download_models.py --slm
```

### Error: "STT model not found"

```bash
uv run python scripts/download_models.py --stt
```

### Error: "CUDA out of memory"

Giảm GPU layers trong `.env`:

```env
SLM_N_GPU_LAYERS=0    # Chạy full CPU
```

### Error: "No speech detected"

Điều chỉnh VAD threshold:

```env
VAD_THRESHOLD=0.3     # Thấp hơn = sensitive hơn
```

## 📚 API Integration

Sử dụng trong FastAPI backend:

```python
from src.processing.pipeline import AudioPipeline

# Pipeline được khởi tạo trong src/api/main.py
# Tự động chạy STT → Embedding → SLM cho mỗi session

# WebSocket endpoint: /ws/audio/{session_id}
# Events được push về frontend qua polling: /api/exam/events/{session_id}
```

## 🎓 Model Details

| Component | Model | Size | Language |
|-----------|-------|------|----------|
| STT | vinai/PhoWhisper-small | ~244MB | Vietnamese |
| Embedding | keepitreal/vietnamese-sbert | ~400MB | Vietnamese |
| SLM | Qwen2.5-3B-Instruct-Q4_K_M | ~1.9GB | Multilingual |
| VAD | Silero VAD v4 | ~2MB | Language-agnostic |

## 📝 Notes

- **CPU-friendly:** Toàn bộ pipeline có thể chạy trên CPU với performance chấp nhận được
- **GPU acceleration:** Set `TORCH_DEVICE=cuda` để tăng tốc STT và Embedding
- **Quantization:** SLM đã được quantize (Q4_K_M) để giảm kích thước và tăng tốc độ
- **Vietnamese-optimized:** STT và Embedding được train riêng cho tiếng Việt

## 🚀 Next Steps

1. Fine-tune SLM trên dữ liệu thi cử thật
2. Thu thập dataset câu hỏi + câu trả lời để improve embedding threshold
3. Implement caching cho embeddings của câu hỏi thi
4. Add confidence calibration cho SLM output

## 📞 Support

Nếu gặp vấn đề, check logs:

```bash
# Backend logs
tail -f logs/app.log

# Hoặc chạy với debug mode
LOG_LEVEL=DEBUG uv run uvicorn src.api.main:app --reload
```
