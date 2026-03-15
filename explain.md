# EXPLAIN - Tổng Quan Chi Tiết Codebase DSP391m

Tài liệu này dùng để bạn trình bày đồ án: mục tiêu, kiến trúc, luồng dữ liệu, chính sách chống gian lận, storage, API, test, và các điểm mạnh/yếu hiện tại.

## 1. Mục tiêu hệ thống

Hệ thống là một POC/MVP giám sát thi nói online theo thời gian thực, xử lý local trên máy chạy backend.

Mục tiêu chính:
- Xác minh danh tính thí sinh bằng voice enrollment + verify định kỳ.
- Cho phép thi trong môi trường nhiều người nói (không phạt vì overlap đơn thuần).
- Phát hiện nội dung nói liên quan đề thi bằng pipeline STT + embedding + SLM.
- Lưu vết đầy đủ để truy xuất báo cáo và lịch sử phiên thi.

## 2. Kiến trúc tổng thể

### 2.1 Thành phần
- `frontend/app.py`: Streamlit UI cho enrollment, start/stop, log realtime, report.
- `src/api/main.py`: khởi tạo FastAPI app, load model, setup pipeline.
- `src/api/routes/`: REST endpoints (`exam.py`, `enrollment.py`, `health.py`).
- `src/api/websocket/audio_handler.py`: nhận audio chunk qua WebSocket.
- `src/processing/pipeline.py`: trung tâm xử lý audio và policy.
- `src/core/session.py`: session manager in-memory.
- `src/core/models.py`: schema/pydantic models cho session và API.
- `src/storage/transcript_store.py`: lưu transcript/report/index theo session-centric layout.

### 2.2 Luồng runtime
1. User enroll giọng nói qua `/api/enroll/{student_id}`.
2. User start exam qua `/api/exam/start`.
3. Frontend mở WS `/ws/audio/{session_id}` và gửi chunk PCM int16 base64.
4. Backend xử lý chunk trong `AudioPipeline.process_session`.
5. Frontend polling `/api/exam/events/{session_id}` để nhận log/alert realtime.
6. Stop phiên qua `/api/exam/stop/{session_id}`.
7. Lấy báo cáo cuối qua `/api/exam/report/{session_id}`.

## 3. Chính sách nghiệp vụ hiện tại (quan trọng khi báo cáo)

### 3.1 Điều kiện bắt đầu phiên thi
`POST /api/exam/start` chỉ thành công khi:
- verifier khả dụng (không thì `503`),
- student đã enroll (không thì `403`).

### 3.2 Xác minh trong lúc thi
- Verification loop chạy định kỳ theo `VERIFICATION_INTERVAL`.
- Mỗi lần fail tăng `verification_failures_count` và phát `verification_alert`.
- Khi fail đạt `VERIFICATION_MAX_FAILURES` (mặc định 3):
  - session chuyển trạng thái `cancelled`,
  - phát event `session_terminated`,
  - `cheating_flag = true`,
  - finalize metadata session.

### 3.3 Quy tắc cheating
- Overlap/multi-speaker **không** tự động là cheating.
- Nếu SLM xác nhận transcript liên quan đề thi (`slm_verdict = true`), dù speaker là candidate hay other, đều set cheating.

## 4. Audio pipeline chi tiết

File chính: `src/processing/pipeline.py`.

### 4.1 Các vòng lặp song song
- `process_session`: vòng lặp nhận chunk audio từ queue.
- `_verification_loop`: loop verify giọng nói nền.
- `_diarization_loop` (nếu enable): phân cụm speaker theo cửa sổ trượt.

### 4.2 Các bước xử lý trong phiên
- Decode base64 -> `np.float32` audio.
- VAD lọc silence.
- Ghi rolling buffer (`_recent_audio`) cho verify + diarization.
- Nếu diarization bật:
  - phân đoạn speaker,
  - xác định exam taker,
  - chạy STT cho từng speaker,
  - chạy embedding similarity,
  - chạy SLM nếu similarity đủ cao.
- Nếu diarization tắt/fallback:
  - chạy STT theo buffer chunk.

### 4.3 Lưu transcript enriched
Mỗi segment lưu thêm:
- `speaker_id`, `speaker_role` (`candidate|other|unknown`),
- `source` (`diarization|stt_only`),
- `similarity`, `slm_verdict`,
- `is_exam_related`, `is_candidate_speech`.

## 5. Realtime event contract

Endpoint: `GET /api/exam/events/{session_id}`

Đặc điểm:
- FIFO queue per session,
- poll-based (frontend không cần WS thứ 2 để nhận event),
- response có `schema_version` để chống lệch contract frontend/backend.

Event thường gặp:
- `transcript_log`
- `diarization_log`
- `verification_alert`
- `slm_alert`
- `cheating_alert`
- `session_terminated`

## 6. Report contract

Endpoint: `GET /api/exam/report/{session_id}`

Report gồm:
- metadata phiên (`session_id`, `student_id`, `exam_id`, `status`, `started_at`, `ended_at`),
- kết quả (`cheating_detected`, `verification_failures`, `overlap_count`),
- transcript đầy đủ enriched,
- `schema_version` để frontend kiểm tra tương thích.

## 7. Storage design

### 7.1 Cấu trúc thư mục
```text
storage/
  enrollment/{student_id}.npy
  sessions/
    index.jsonl
    YYYY-MM-DD/{session_id}/
      meta.json
      transcript.jsonl
      report.json
```

### 7.2 Query lịch sử
`GET /api/exam/history` hỗ trợ filter:
- `student_id`
- `exam_id`
- `status_filter`
- `started_from`, `started_to`
- `limit`, `offset`

### 7.3 Retention
- Dọn session/index hết hạn theo `STORAGE_RETENTION_DAYS`.
- Cleanup chạy ở startup backend.
- Nếu `<= 0` thì disable retention.

## 8. Frontend behavior

Frontend là Streamlit + JS mic capture:
- JS lấy audio từ browser, encode PCM int16 base64, gửi WS.
- Python side poll `/events` để render detection log.
- Frontend check `schema_version` cho events/report:
  - nếu mismatch với expected (`1.0.0`) thì hiển thị warning rõ ràng,
  - đồng thời ghi log error để dễ debug.

## 9. Mô hình/processor đang dùng

Cấu hình thực tế phụ thuộc `.env` và model files trên máy.
Hệ thống hỗ trợ:
- VAD: Silero
- STT: PhoWhisper
- Speaker Verification: ECAPA-based
- Diarization: NeMo Sortformer
- SLM: GGUF via llama-cpp (optional)

Nếu model nào không load được, hệ thống có fallback tương ứng (ví dụ chạy không SLM hoặc không diarization).

## 10. Testing strategy hiện tại

### 10.1 Unit/Component
- Nhiều test cho buffer, vad, stt, api, policy.

### 10.2 E2E (đã có)
- Happy path:
  `enroll -> start -> websocket audio -> poll events -> stop -> report -> history`
- Policy fail path:
  verify fail 3 lần -> phát `session_terminated` -> status `cancelled`.

### 10.3 Lệnh chạy
```bash
uv run ruff check .
uv run mypy src
uv run pytest -q
```

## 11. Các điểm mạnh khi trình bày

- Policy rõ ràng, dễ giải thích, bám use-case thực tế môi trường nhiều người nói.
- Contract API tách bạch REST + WS + poll events.
- Dữ liệu lưu vết tốt để audit sau thi.
- Có schema version để giảm rủi ro lệch frontend/backend.
- Có retention và query lịch sử, không chỉ demo runtime.

## 12. Hạn chế hiện tại (nên nêu minh bạch)

- Session manager còn in-memory (chưa distributed).
- Chưa có auth/authorization production-grade.
- Độ chính xác phụ thuộc chất lượng model + dữ liệu môi trường thực.
- E2E hiện dùng fake processors cho độ ổn định CI; test với model thật cần chạy manual benchmark riêng.

## 13. Đề xuất bước tiếp theo sau báo cáo

1. Thêm export báo cáo/histories (`csv/json`) cho nghiệp vụ vận hành.
2. Thêm auth + role-based API.
3. Viết benchmark latency/accuracy theo CPU vs GPU.
4. Chuẩn hóa migration path cho schema version 1.1/2.0.
5. Đóng gói triển khai pilot bằng Docker profile theo phần cứng.

---

Nếu bạn dùng tài liệu này để thuyết trình, nên trình bày theo thứ tự:
1) Problem + Policy,
2) Architecture,
3) Runtime flow,
4) API & storage contracts,
5) Test/E2E evidence,
6) Risk & next steps.
