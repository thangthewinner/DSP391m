"""
AI Exam Proctoring System — Streamlit Frontend (Phase 7)

Audio approach:
  Browser JS captures mic → encodes PCM int16 → sends via WebSocket directly
  to FastAPI backend. No Python thread needed for audio.
  
  A separate Python thread only handles RECEIVING WebSocket messages (alerts).
"""

import json
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import requests
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BACKEND_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000"
EXAMS_DIR = Path(__file__).parent.parent / "exams"
CHEATING_THRESHOLD = 10.0

# ---------------------------------------------------------------------------
# Module-level thread-safe queues
# ---------------------------------------------------------------------------
_log_queue: queue.Queue = queue.Queue(maxsize=500)
_state_queue: queue.Queue = queue.Queue(maxsize=200)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Exam Proctoring",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_state() -> None:
    defaults: dict = {
        "session_id": None,
        "recording": False,
        "suspicion_score": 0.0,
        "cheating_flag": False,
        "flagged_count": 0,
        "logs": [],
        "report": None,
        "ws_stop_event": threading.Event(),
        "overlap_count": 0,
        "verification_failures": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ---------------------------------------------------------------------------
# Drain queues into session_state (called each rerun)
# ---------------------------------------------------------------------------
def drain_queues() -> None:
    while not _log_queue.empty():
        try:
            st.session_state.logs.append(_log_queue.get_nowait())
        except queue.Empty:
            break
    if len(st.session_state.logs) > 200:
        st.session_state.logs = st.session_state.logs[-200:]

    while not _state_queue.empty():
        try:
            patch = _state_queue.get_nowait()
            for k, v in patch.items():
                st.session_state[k] = v
        except queue.Empty:
            break

# ---------------------------------------------------------------------------
# Thread-safe helpers
# ---------------------------------------------------------------------------
def add_log(level: str, message: str) -> None:
    ts = time.strftime("%H:%M:%S")
    try:
        _log_queue.put_nowait((ts, level, message))
    except queue.Full:
        pass


def patch_state(updates: dict) -> None:
    try:
        _state_queue.put_nowait(updates)
    except queue.Full:
        pass

# ---------------------------------------------------------------------------
# Receiver thread — only RECEIVES WS messages (alerts/status)
# ---------------------------------------------------------------------------
def ws_receiver(session_id: str, stop_event: threading.Event) -> None:
    """Connects to backend WS and listens for alert messages only."""
    import websocket

    ws_url = f"{WS_URL}/ws/audio/{session_id}"
    add_log("INFO", f"Receiver kết nối → {ws_url}")

    def on_open(ws):  # noqa: ANN001
        add_log("INFO", "✅ WebSocket receiver connected")

    def on_message(ws, raw):  # noqa: ANN001
        try:
            data = json.loads(raw)
            t = data.get("type", "")
            if t == "status_update":
                patch_state({
                    "suspicion_score": data.get("suspicion_score", 0.0),
                    "cheating_flag": data.get("cheating_flag", False),
                })
            elif t == "cheating_alert":
                score = data.get("suspicion_score", 0.0)
                add_log("ALERT", f"🚨 GIAN LẬN! Score={score:.1f}")
                patch_state({"cheating_flag": True})
            elif t == "overlap_alert":
                count = data.get("overlap_count", 0)
                patch_state({"overlap_count": count})
                add_log("WARN", f"⚠️ Nhiều người nói! (lần {count})")
            elif t == "diarization_log":
                num_spk = data.get("num_speakers", 0)
                speakers = data.get("speakers", [])
                segments = data.get("segments", [])
                overlap = data.get("overlap", False)
                duration = data.get("audio_duration", 0.0)
                spk_str = ", ".join(sorted(speakers)) if speakers else "none"
                level = "WARN" if overlap else "DIAR"
                add_log(level, f"[Diarization] {duration:.1f}s → {num_spk} speaker(s): {spk_str}")
                for seg in segments:
                    spk = seg.get("speaker", "?")
                    s = seg.get("start", 0.0)
                    e = seg.get("end", 0.0)
                    add_log("DIAR", f"  └ [{spk}] {s:.2f}s → {e:.2f}s ({e-s:.1f}s)")
            elif t == "verification_alert":
                sim = data.get("similarity", 0.0)
                fails = data.get("failures_count", 0)
                patch_state({"verification_failures": fails})
                add_log("WARN", f"⚠️ ID thất bại! (sim={sim:.2f}, lần {fails})")
            elif t == "transcript_log":
                text = data.get("text", "")
                conf = data.get("confidence", 0.0)
                sim = data.get("similarity", 0.0)
                ts = data.get("timestamp", 0.0)
                m, s = divmod(int(ts), 60)
                sim_str = f" | sim={sim:.2f}" if sim > 0 else ""
                add_log("STT", f"[{m:02d}:{s:02d}] \"{text}\" (conf={conf:.2f}{sim_str})")
            elif t == "ack":
                pass
            elif t == "error":
                add_log("ERROR", f"Backend: {data.get('message', '')}")
        except Exception as e:
            add_log("ERROR", f"WS parse: {e}")

    def on_error(ws, error):  # noqa: ANN001
        add_log("ERROR", f"WS error: {error}")

    def on_close(ws, code, msg):  # noqa: ANN001
        add_log("INFO", "WS receiver closed")

    ws_app = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    ws_app.run_forever()

# ---------------------------------------------------------------------------
# JavaScript mic component — captures audio and sends WS directly
# ---------------------------------------------------------------------------
def mic_js_component(session_id: str, ws_url: str) -> None:
    """
    Inject a JS component that:
    1. Opens mic via getUserMedia (16kHz, mono)
    2. Opens WebSocket to backend
    3. Encodes audio as PCM int16, base64, sends JSON chunks
    """
    html = f"""
<div id="mic-container" style="font-family:monospace;font-size:13px;">
  <div id="mic-status" style="padding:8px;border-radius:4px;background:#1a3a1a;color:#4CAF50;margin-bottom:8px;">
    ⏳ Đang khởi tạo mic...
  </div>
  <div id="mic-stats" style="color:#888;font-size:11px;"></div>
</div>

<script>
(function() {{
  const SESSION_ID = "{session_id}";
  const WS_URL = "{ws_url}/ws/audio/" + SESSION_ID;
  const SAMPLE_RATE = 16000;
  const CHUNK_DURATION = 2.0; // seconds per chunk

  let ws = null;
  let audioCtx = null;
  let chunkTs = 0.0;
  let chunkCount = 0;
  let pcmBuffer = [];
  let samplesPerChunk = Math.floor(SAMPLE_RATE * CHUNK_DURATION);

  const statusEl = document.getElementById('mic-status');
  const statsEl = document.getElementById('mic-stats');

  function setStatus(msg, color) {{
    statusEl.textContent = msg;
    statusEl.style.color = color || '#4CAF50';
  }}

  function base64Encode(int16Array) {{
    const bytes = new Uint8Array(int16Array.buffer);
    let binary = '';
    for (let i = 0; i < bytes.length; i++) {{
      binary += String.fromCharCode(bytes[i]);
    }}
    return btoa(binary);
  }}

  function sendChunk(samples) {{
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const int16 = new Int16Array(samples.length);
    for (let i = 0; i < samples.length; i++) {{
      int16[i] = Math.max(-32768, Math.min(32767, Math.round(samples[i] * 32767)));
    }}
    chunkTs += CHUNK_DURATION;
    chunkCount++;
    const payload = {{
      type: "audio_chunk",
      data: base64Encode(int16),
      timestamp: parseFloat(chunkTs.toFixed(3)),
      sample_rate: SAMPLE_RATE,
      channels: 1
    }};
    ws.send(JSON.stringify(payload));
    statsEl.textContent = "Chunks sent: " + chunkCount + " | Time: " + chunkTs.toFixed(1) + "s";
  }}

  // Open WebSocket
  function connectWS() {{
    ws = new WebSocket(WS_URL);
    ws.onopen = () => {{
      setStatus('🔴 Mic đang ghi âm — WebSocket connected', '#4CAF50');
    }};
    ws.onerror = (e) => {{
      setStatus('❌ WebSocket error', '#F44336');
      console.error('WS error', e);
    }};
    ws.onclose = () => {{
      setStatus('⏹ WebSocket closed', '#888');
    }};
    ws.onmessage = (e) => {{
      // Responses handled by Python receiver thread
    }};
  }}

  // Start mic
  async function startMic() {{
    try {{
      const stream = await navigator.mediaDevices.getUserMedia({{
        audio: {{
          sampleRate: SAMPLE_RATE,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        }},
        video: false
      }});

      setStatus('🎤 Mic granted — connecting WebSocket...', '#FF9800');
      connectWS();

      audioCtx = new (window.AudioContext || window.webkitAudioContext)({{ sampleRate: SAMPLE_RATE }});
      const source = audioCtx.createMediaStreamSource(stream);
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);

      processor.onaudioprocess = (e) => {{
        const inputData = e.inputBuffer.getChannelData(0);
        for (let i = 0; i < inputData.length; i++) {{
          pcmBuffer.push(inputData[i]);
        }}
        while (pcmBuffer.length >= samplesPerChunk) {{
          const chunk = pcmBuffer.splice(0, samplesPerChunk);
          sendChunk(chunk);
        }}
      }};

      source.connect(processor);
      processor.connect(audioCtx.destination);

    }} catch(err) {{
      setStatus('❌ Mic error: ' + err.message, '#F44336');
      console.error(err);
    }}
  }}

  startMic();
}})();
</script>
"""
    components.html(html, height=80)

# ---------------------------------------------------------------------------
# Exam file loader
# ---------------------------------------------------------------------------
def load_exam_files() -> dict[str, dict]:
    exams: dict[str, dict] = {}
    if not EXAMS_DIR.exists():
        return exams
    for f in sorted(EXAMS_DIR.glob("*.txt")):
        meta = {"filename": f.name, "path": f, "name": f.stem, "code": "", "duration": "", "content": ""}
        lines = f.read_text(encoding="utf-8").splitlines()
        content_lines = []
        for line in lines:
            if line.startswith("# Tên môn:"):
                meta["name"] = line.split(":", 1)[1].strip()
            elif line.startswith("# Mã đề:"):
                meta["code"] = line.split(":", 1)[1].strip()
            elif line.startswith("# Thời gian:"):
                meta["duration"] = line.split(":", 1)[1].strip()
            elif not line.startswith("#"):
                content_lines.append(line)
        meta["content"] = "\n".join(content_lines).strip()
        exams[meta["name"]] = meta
    return exams

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def api_start_session(student_id: str, exam_id: str, exam_question: str) -> Optional[str]:
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/exam/start",
            json={"student_id": student_id, "exam_id": exam_id, "exam_question": exam_question},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()["session_id"]
    except Exception as e:
        add_log("ERROR", f"Không thể tạo session: {e}")
        return None


def api_stop_session(session_id: str) -> None:
    try:
        requests.post(f"{BACKEND_URL}/api/exam/stop/{session_id}", timeout=10)
    except Exception as e:
        add_log("ERROR", f"Lỗi dừng session: {e}")


def api_get_status(session_id: str) -> Optional[dict]:
    try:
        resp = requests.get(f"{BACKEND_URL}/api/exam/status/{session_id}", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def api_get_report(session_id: str) -> Optional[dict]:
    try:
        resp = requests.get(f"{BACKEND_URL}/api/exam/report/{session_id}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        add_log("ERROR", f"Lỗi lấy report: {e}")
        return None


def check_backend() -> bool:
    try:
        return requests.get(f"{BACKEND_URL}/health", timeout=3).status_code == 200
    except Exception:
        return False

# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------
def render_sidebar(exams: dict) -> tuple[str, str, str, str]:
    st.sidebar.title("🎓 Cài đặt phiên thi")
    st.sidebar.markdown("---")

    if check_backend():
        st.sidebar.success("✅ Backend: Online")
    else:
        st.sidebar.error("❌ Backend: Offline\n\n`uv run uvicorn src.api.main:app --reload`")

    st.sidebar.markdown("---")
    student_id = st.sidebar.text_input("Student ID", value="student_001", key="student_id_input")

    exam_names = list(exams.keys())
    if not exam_names:
        st.sidebar.warning(f"Không tìm thấy đề thi trong `{EXAMS_DIR}`")
        return student_id, "", "", ""

    selected_name = st.sidebar.selectbox("Chọn đề thi", exam_names, key="exam_select")
    exam = exams[selected_name]

    with st.sidebar.expander("📄 Xem đề thi", expanded=False):
        st.text(f"Mã đề: {exam['code']}")
        st.text(f"Thời gian: {exam['duration']}")
        st.markdown("---")
        st.text(exam["content"][:800] + ("..." if len(exam["content"]) > 800 else ""))

    return student_id, selected_name, exam["code"], exam["content"]


def render_controls(student_id: str, exam_name: str, exam_id: str, exam_content: str) -> None:
    col1, col2 = st.columns(2)

    with col1:
        start_disabled = st.session_state.recording or not student_id or not exam_name
        if st.button("🎤 Bắt đầu giám sát", disabled=start_disabled, use_container_width=True, type="primary"):
            session_id = api_start_session(student_id, exam_id, exam_content)
            if session_id:
                st.session_state.session_id = session_id
                st.session_state.recording = True
                st.session_state.suspicion_score = 0.0
                st.session_state.cheating_flag = False
                st.session_state.logs = []
                st.session_state.report = None
                st.session_state.overlap_count = 0
                st.session_state.verification_failures = 0
                st.session_state.flagged_count = 0
                st.session_state.ws_stop_event.clear()

                # Start receiver thread
                t = threading.Thread(
                    target=ws_receiver,
                    args=(session_id, st.session_state.ws_stop_event),
                    daemon=True,
                )
                t.start()
                add_log("INFO", f"Session: {session_id[:8]}...")
                st.rerun()

    with col2:
        stop_disabled = not st.session_state.recording
        if st.button("⏹ Kết thúc", disabled=stop_disabled, use_container_width=True):
            if st.session_state.session_id:
                st.session_state.ws_stop_event.set()
                api_stop_session(st.session_state.session_id)
                time.sleep(0.5)
                report = api_get_report(st.session_state.session_id)
                st.session_state.report = report
                st.session_state.recording = False
                add_log("INFO", "Session kết thúc")
                st.rerun()


def render_status_bar() -> None:
    if st.session_state.recording:
        sid = st.session_state.session_id or ""
        st.info(f"🔴 **Đang ghi âm** — Session: `{sid[:8]}...`")
    elif st.session_state.report:
        st.success("✅ Phiên thi đã kết thúc — Xem report bên dưới")
    else:
        st.info("⏸ Chưa bắt đầu — Cấu hình và nhấn **Bắt đầu giám sát**")


def render_suspicion_panel() -> None:
    score = st.session_state.suspicion_score
    flag = st.session_state.cheating_flag
    pct = min(score / CHEATING_THRESHOLD, 1.0)

    col_score, col_badge = st.columns([3, 1])
    with col_score:
        if flag:
            color, label = "🔴", "GIAN LẬN"
        elif pct > 0.6:
            color, label = "🟠", "NGHI NGỜ CAO"
        elif pct > 0.3:
            color, label = "🟡", "NGHI NGỜ"
        else:
            color, label = "🟢", "BÌNH THƯỜNG"
        st.markdown(f"**Suspicion Score: {score:.1f} / {CHEATING_THRESHOLD:.0f}** — {color} {label}")
        st.progress(pct)

    with col_badge:
        if st.session_state.overlap_count > 0:
            st.metric("Overlap", st.session_state.overlap_count, delta="⚠️")
        if st.session_state.verification_failures > 0:
            st.metric("ID Fail", st.session_state.verification_failures, delta="⚠️")


def render_log_panel() -> None:
    logs = st.session_state.logs
    level_colors = {
        "INFO":  "#4CAF50",
        "WARN":  "#FF9800",
        "ALERT": "#F44336",
        "ERROR": "#9C27B0",
        "DIAR":  "#29B6F6",
        "STT":   "#E0E0E0",
    }
    parts = []
    for ts, level, msg in logs[-80:]:
        color = level_colors.get(level, "#888")
        safe_msg = msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        parts.append(
            f'<div style="font-family:monospace;font-size:12px;margin:1px 0;">'
            f'<span style="color:#666">[{ts}]</span> '
            f'<span style="color:{color};font-weight:bold">[{level}]</span> '
            f'<span style="color:#ddd">{safe_msg}</span>'
            f'</div>'
        )
    body = "\n".join(parts) if parts else '<div style="color:#555;font-style:italic;padding:8px">Chưa có log...</div>'
    st.markdown(
        f'<div style="background:#1a1a1a;border:1px solid #333;border-radius:6px;'
        f'padding:10px;height:320px;overflow-y:auto;">{body}</div>',
        unsafe_allow_html=True,
    )


def render_report_panel(report: dict) -> None:
    st.markdown("---")
    st.subheader("📊 Báo cáo phiên thi")

    cheating = report.get("cheating_detected", False)
    max_score = report.get("max_suspicion_score", 0.0)
    confidence = report.get("confidence", 0.0)
    rationale = report.get("rationale", "")
    flagged = report.get("flagged_segments", [])
    transcript = report.get("transcript", [])
    elapsed = report.get("elapsed_seconds", 0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Kết quả", "🚨 GIAN LẬN" if cheating else "✅ BÌNH THƯỜNG")
    col2.metric("Max Score", f"{max_score:.1f}")
    col3.metric("Confidence", f"{confidence:.0%}")
    col4.metric("Thời gian", f"{elapsed:.0f}s")

    if rationale:
        st.info(f"**Phân tích:** {rationale}")

    if flagged:
        st.markdown(f"**⚠️ {len(flagged)} đoạn bị đánh dấu:**")
        for seg in flagged:
            ts = seg.get("timestamp", 0)
            text = seg.get("text", "")
            sim = seg.get("similarity_score", 0)
            slm = seg.get("slm_verdict", False)
            m, s = divmod(int(ts), 60)
            flags = []
            if sim >= 0.75:
                flags.append(f"Sim={sim:.2f}")
            if slm:
                flags.append("SLM=YES")
            st.markdown(
                f'<div style="background:#2a1a1a;border-left:3px solid #F44336;'
                f'padding:6px 10px;margin:3px 0;border-radius:3px;font-family:monospace;font-size:13px;">'
                f'[{m:02d}:{s:02d}] "{text}" '
                f'<span style="color:#F44336;font-size:11px">{" | ".join(flags)}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    if transcript:
        with st.expander(f"📝 Transcript ({len(transcript)} đoạn)"):
            for seg in transcript:
                ts = seg.get("start", 0)
                text = seg.get("text", "")
                conf = seg.get("confidence", 0)
                m, s = divmod(int(ts), 60)
                st.markdown(f"`[{m:02d}:{s:02d}]` {text} *(conf={conf:.2f})*")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    drain_queues()

    st.title("🎓 AI Exam Proctoring System")
    st.caption("Hệ thống giám sát thi cử thời gian thực — Phase 7")

    exams = load_exam_files()
    student_id, exam_name, exam_id, exam_content = render_sidebar(exams)

    render_controls(student_id, exam_name, exam_id, exam_content)
    render_status_bar()
    st.markdown("---")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("🎤 Microphone")
        if st.session_state.recording and st.session_state.session_id:
            # JS component handles mic capture + WebSocket sending
            mic_js_component(st.session_state.session_id, WS_URL)

            # Poll status
            status = api_get_status(st.session_state.session_id)
            if status:
                st.session_state.suspicion_score = status.get("current_suspicion_score", 0.0)
                st.session_state.cheating_flag = status.get("cheating_flag", False)
                st.session_state.flagged_count = status.get("flagged_segments_count", 0)
                elapsed = status.get("elapsed_time_seconds", 0)
                m, s = divmod(int(elapsed), 60)
                st.metric("Thời gian", f"{m:02d}:{s:02d}")
                st.metric("Flagged", st.session_state.flagged_count)
        else:
            st.info("Nhấn **Bắt đầu giám sát** để bật mic")

    with col_right:
        st.subheader("📊 Giám sát thời gian thực")
        render_suspicion_panel()
        st.markdown("**🔍 Detection Log**")
        render_log_panel()

        if st.session_state.recording:
            time.sleep(1)
            st.rerun()

    if st.session_state.report:
        render_report_panel(st.session_state.report)


if __name__ == "__main__":
    main()
