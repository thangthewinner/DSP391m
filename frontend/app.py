"""
AI Exam Proctoring System — Streamlit Frontend (Phase 7/8)

Audio approach:
  Browser JS captures mic → encodes PCM int16 → sends via WebSocket directly
  to FastAPI backend. No Python thread needed for audio.

Event delivery:
  Frontend polls GET /api/exam/events/{session_id} every ~1.5s to receive
  transcript, diarization, and alert events. This avoids the dual-WebSocket
  problem where a Python receiver thread would open a second connection to the
  same session endpoint and conflict with the JS audio sender.
"""

import queue
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


# ---------------------------------------------------------------------------
# Module-level queues (kept for compatibility but no longer fed by threads)
# ---------------------------------------------------------------------------
_log_queue: queue.Queue = queue.Queue(maxsize=500)
_state_queue: queue.Queue = queue.Queue(maxsize=200)
_diar_queue: queue.Queue = queue.Queue(maxsize=50)

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
        "cheating_flag": False,
        "logs": [],
        "diar_events": [],
        "report": None,
        "overlap_count": 0,
        "verification_failures": 0,
        # Phase 8: enrollment
        "enrolled": False,
        "enrolled_at": None,
        "show_reverify": False,
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

    while not _diar_queue.empty():
        try:
            st.session_state.diar_events.append(_diar_queue.get_nowait())
        except queue.Empty:
            break
    if len(st.session_state.diar_events) > 30:
        st.session_state.diar_events = st.session_state.diar_events[-30:]

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
# Event polling — replaces ws_receiver thread
# ---------------------------------------------------------------------------
def process_events(events: list[dict]) -> None:
    """Process events returned by GET /api/exam/events/{session_id}."""
    for data in events:
        t = data.get("type", "")
        if t == "transcript_log":
            text = data.get("text", "")
            conf = data.get("confidence", 0.0)
            sim = data.get("similarity", 0.0)
            ts = data.get("timestamp", 0.0)
            spk = data.get("speaker", "")
            role = data.get("speaker_role", "")
            m, s = divmod(int(ts), 60)
            sim_str = f" | sim={sim:.2f}" if sim > 0 else ""
            if spk and role:
                spk_str = f" [{spk}•{role}]"
            elif spk:
                spk_str = f" [{spk}]"
            else:
                spk_str = ""
            add_log("STT", f'[{m:02d}:{s:02d}]{spk_str} "{text}" (conf={conf:.2f}{sim_str})')

        elif t == "diarization_log":
            num_spk = data.get("num_speakers", 0)
            speakers = data.get("speakers", [])
            segments = data.get("segments", [])
            overlap = data.get("overlap", False)
            duration = data.get("audio_duration", 0.0)
            dominant = data.get("dominant_speaker", "")  # Phase 8
            recv_ts = time.strftime("%H:%M:%S")

            # Push to visual diarization panel
            try:
                _diar_queue.put_nowait({
                    "ts": recv_ts,
                    "num_speakers": num_spk,
                    "speakers": speakers,
                    "segments": segments,
                    "overlap": overlap,
                    "duration": duration,
                    "dominant_speaker": dominant,  # Phase 8
                })
            except queue.Full:
                pass

            # Text log
            if overlap:
                add_log("WARN", f"[DIAR] ⚠️ {num_spk} người nói — {duration:.1f}s")
            else:
                spk_label = speakers[0] if speakers else "?"
                add_log("DIAR", f"[DIAR] ✅ 1 người ({spk_label}) — {duration:.1f}s")
            for seg in segments:
                spk = seg.get("speaker", "?")
                s = seg.get("start", 0.0)
                e = seg.get("end", 0.0)
                add_log("DIAR", f"  └ [{spk}] {s:.2f}s–{e:.2f}s ({e-s:.1f}s)")

        elif t == "overlap_alert":
            count = data.get("overlap_count", 0)
            st.session_state.overlap_count = count
            add_log("WARN", f"⚠️ Nhiều người nói! (lần {count})")

        elif t == "verification_alert":
            sim = data.get("similarity", 0.0)
            fails = data.get("failures_count", 0)
            st.session_state.verification_failures = fails
            st.session_state.show_reverify = True  # Phase 8: trigger re-verify popup
            add_log("WARN", f"⚠️ ID thất bại! (sim={sim:.2f}, lần {fails})")

        elif t == "slm_alert":
            text = data.get("text", "")
            sim = data.get("similarity", 0.0)
            add_log("ALERT", f"🚨 [SLM] Nội dung liên quan bài thi! \"{text[:60]}\" (sim={sim:.2f})")

        elif t == "cheating_alert":
            st.session_state.cheating_flag = True
            add_log("ALERT", "🚨 GIAN LẬN phát hiện!")

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


def api_poll_events(session_id: str) -> list[dict]:
    try:
        resp = requests.get(f"{BACKEND_URL}/api/exam/events/{session_id}", timeout=5)
        resp.raise_for_status()
        return resp.json().get("events", [])
    except Exception:
        return []


def check_backend() -> bool:
    try:
        return requests.get(f"{BACKEND_URL}/health", timeout=3).status_code == 200
    except Exception:
        return False


def api_enroll(student_id: str, audio_b64: str) -> bool:
    """Call POST /api/enroll/{student_id} with one base64-encoded PCM int16 audio sample."""
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/enroll/{student_id}",
            json={"audio_samples": [audio_b64], "sample_rate": 16000},
            timeout=15,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        add_log("ERROR", f"Enrollment thất bại: {e}")
        return False


def api_check_enrollment(student_id: str) -> bool:
    """Return True if student already has an enrollment on the backend."""
    try:
        resp = requests.get(f"{BACKEND_URL}/api/enroll/{student_id}/status", timeout=5)
        if resp.status_code == 200:
            return resp.json().get("enrolled", False)
    except Exception:
        pass
    return False


def enrollment_recorder_js(student_id: str, key_suffix: str = "") -> None:
    """
    JS component: 5s countdown → thu âm PCM int16 → POST /api/enroll/{student_id}.
    Sau khi enroll xong, reload page để cập nhật trạng thái.
    """
    html = f"""
<div id="enroll-container-{key_suffix}" style="font-family:monospace;font-size:13px;padding:8px;background:#111;border-radius:6px;">
  <div id="enroll-status-{key_suffix}" style="color:#FF9800;margin-bottom:6px;">⏳ Nhấn nút bên dưới để bắt đầu đăng ký</div>
  <div style="color:#888;font-size:11px;margin-bottom:8px;">
    Vui lòng đọc to câu sau trong 5 giây:<br>
    <strong style="color:#fff;">&ldquo;Tôi là {student_id}, tôi đang tham gia kỳ thi hôm nay&rdquo;</strong>
  </div>
  <button id="enroll-btn-{key_suffix}"
    onclick="startEnrollment_{key_suffix}()"
    style="background:#1565C0;color:#fff;border:none;padding:6px 14px;border-radius:4px;cursor:pointer;font-size:13px;">
    🎤 Bắt đầu đăng ký (5s)
  </button>
  <div id="enroll-progress-{key_suffix}" style="margin-top:6px;color:#aaa;font-size:11px;"></div>
</div>

<script>
window.startEnrollment_{key_suffix} = async function() {{
  const statusEl = document.getElementById('enroll-status-{key_suffix}');
  const progressEl = document.getElementById('enroll-progress-{key_suffix}');
  const btn = document.getElementById('enroll-btn-{key_suffix}');
  btn.disabled = true;
  btn.style.opacity = '0.5';

  const SAMPLE_RATE = 16000;
  const DURATION = 5;
  const BACKEND = '{BACKEND_URL}';
  const STUDENT_ID = '{student_id}';

  try {{
    const stream = await navigator.mediaDevices.getUserMedia({{ audio: {{ sampleRate: SAMPLE_RATE, channelCount: 1 }}, video: false }});
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)({{ sampleRate: SAMPLE_RATE }});
    const source = audioCtx.createMediaStreamSource(stream);
    const processor = audioCtx.createScriptProcessor(4096, 1, 1);

    let allSamples = [];
    let countdown = DURATION;

    statusEl.textContent = '🔴 Đang thu âm... ' + countdown + 's còn lại';
    statusEl.style.color = '#F44336';

    const timer = setInterval(() => {{
      countdown--;
      if (countdown > 0) {{
        statusEl.textContent = '🔴 Đang thu âm... ' + countdown + 's còn lại';
      }}
    }}, 1000);

    processor.onaudioprocess = (e) => {{
      const data = e.inputBuffer.getChannelData(0);
      allSamples.push(...data);
    }};
    source.connect(processor);
    processor.connect(audioCtx.destination);

    await new Promise(r => setTimeout(r, DURATION * 1000));
    clearInterval(timer);

    // Stop
    processor.disconnect();
    source.disconnect();
    stream.getTracks().forEach(t => t.stop());
    audioCtx.close();

    statusEl.textContent = '⏳ Đang gửi lên server...';
    statusEl.style.color = '#FF9800';

    // Encode to PCM int16 base64
    const int16 = new Int16Array(allSamples.length);
    for (let i = 0; i < allSamples.length; i++) {{
      int16[i] = Math.max(-32768, Math.min(32767, Math.round(allSamples[i] * 32767)));
    }}
    const bytes = new Uint8Array(int16.buffer);
    let binary = '';
    for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
    const b64 = btoa(binary);

    // POST to backend
    const resp = await fetch(BACKEND + '/api/enroll/' + STUDENT_ID, {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ audio_samples: [b64], sample_rate: SAMPLE_RATE }})
    }});

    if (resp.ok) {{
      statusEl.textContent = '✅ Đăng ký thành công! Trang sẽ tự reload...';
      statusEl.style.color = '#4CAF50';
      progressEl.textContent = '';
      setTimeout(() => window.location.reload(), 1500);
    }} else {{
      const err = await resp.json().catch(() => ({{ detail: resp.statusText }}));
      statusEl.textContent = '❌ Thất bại: ' + (err.detail || resp.status);
      statusEl.style.color = '#F44336';
      btn.disabled = false;
      btn.style.opacity = '1';
    }}
  }} catch(err) {{
    statusEl.textContent = '❌ Lỗi mic: ' + err.message;
    statusEl.style.color = '#F44336';
    btn.disabled = false;
    btn.style.opacity = '1';
  }}
}};
</script>
"""
    components.html(html, height=200)

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

    # Phase 8: Enrollment section
    st.sidebar.markdown("**🎤 Đăng ký giọng nói**")
    if student_id:
        enrolled = api_check_enrollment(student_id)
        st.session_state.enrolled = enrolled
        if enrolled:
            st.sidebar.success("✅ Đã đăng ký giọng nói")
            if st.sidebar.button("🔄 Đăng ký lại", key="re_enroll_btn", use_container_width=True):
                st.session_state["show_enroll_recorder"] = True
        else:
            st.sidebar.warning("⚠️ Chưa có giọng đăng ký")
            if st.sidebar.button("🎤 Đăng ký giọng nói", key="enroll_btn", use_container_width=True, type="primary"):
                st.session_state["show_enroll_recorder"] = True

        if st.session_state.get("show_enroll_recorder", False):
            with st.sidebar.expander("🎙 Thu âm đăng ký", expanded=True):
                enrollment_recorder_js(student_id, key_suffix="sidebar")
    else:
        st.sidebar.info("Nhập Student ID để đăng ký giọng")

    st.sidebar.markdown("---")

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
        # Phase 8: disable if not enrolled (with bypass checkbox for testing)
        bypass_enroll = st.checkbox("🔓 Bỏ qua kiểm tra enrollment (test)", key="bypass_enroll_check")
        enrolled = st.session_state.get("enrolled", False) or bypass_enroll
        start_disabled = st.session_state.recording or not student_id or not exam_name or not enrolled
        if not enrolled and not bypass_enroll:
            st.caption("⚠️ Vui lòng đăng ký giọng nói trong sidebar trước khi bắt đầu")
        if st.button("🎤 Bắt đầu giám sát", disabled=start_disabled, use_container_width=True, type="primary"):
            session_id = api_start_session(student_id, exam_id, exam_content)
            if session_id:
                st.session_state.session_id = session_id
                st.session_state.recording = True
                st.session_state.cheating_flag = False
                st.session_state.logs = []
                st.session_state.diar_events = []
                st.session_state.report = None
                st.session_state.overlap_count = 0
                st.session_state.verification_failures = 0
                add_log("INFO", f"Session: {session_id[:8]}...")
                st.rerun()

    with col2:
        stop_disabled = not st.session_state.recording
        if st.button("⏹ Kết thúc", disabled=stop_disabled, use_container_width=True):
            if st.session_state.session_id:
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
    flag = st.session_state.cheating_flag

    col_status, col_badge = st.columns([3, 1])
    with col_status:
        if flag:
            st.markdown("**🔴 CẢNH BÁO GIAN LẬN**")
        else:
            st.markdown("**🟢 BÌNH THƯỜNG**")

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


def _diar_timeline(segments: list[dict], duration: float, dominant: str = "", width: int = 32) -> str:
    """Render ASCII timeline bar for each speaker. dominant speaker is always blue."""
    if not segments or duration <= 0:
        return ""

    speakers_sorted = sorted({s.get("speaker", "") for s in segments})

    def spk_color(spk: str) -> str:
        if dominant and spk == dominant:
            return "#29B6F6"  # blue = thí sinh
        fallback = ["#FF9800", "#AB47BC", "#66BB6A"]
        non_dom = [s for s in speakers_sorted if s != dominant]
        idx = non_dom.index(spk) % len(fallback) if spk in non_dom else 0
        return fallback[idx]

    lines = []
    for spk in speakers_sorted:
        bar = ["░"] * width
        for seg in segments:
            if seg.get("speaker") != spk:
                continue
            s = seg.get("start", 0.0)
            e = seg.get("end", 0.0)
            lo = int(s / duration * width)
            hi = int(e / duration * width)
            for i in range(max(0, lo), min(width, hi + 1)):
                bar[i] = "█"
        bar_str = "".join(bar)
        color = spk_color(spk)
        role = " •thí sinh" if (dominant and spk == dominant) else ""
        lines.append(
            f'<div style="font-family:monospace;font-size:12px;margin:2px 0;">'
            f'<span style="color:{color};min-width:110px;display:inline-block">[{spk}{role}]</span> '
            f'<span style="color:{color}">{bar_str}</span>'
            f'</div>'
        )
    return "\n".join(lines)


def render_diarization_panel() -> None:
    """Visual diarization history panel — last N events with timeline bars."""
    events = st.session_state.get("diar_events", [])

    st.markdown("**🎙 Speaker Diarization Log**")

    if not events:
        st.markdown(
            '<div style="background:#111;border:1px solid #333;border-radius:6px;'
            'padding:10px;color:#555;font-style:italic;font-size:13px;">Chưa có dữ liệu diarization...</div>',
            unsafe_allow_html=True,
        )
        return

    rows = []
    for ev in reversed(events[-10:]):   # newest first, max 10
        ts = ev["ts"]
        num_spk = ev["num_speakers"]
        duration = ev["duration"]
        overlap = ev["overlap"]
        segments = ev["segments"]
        dominant = ev.get("dominant_speaker", "")  # Phase 8

        if overlap:
            header_color = "#FF9800"
            icon = "⚠️"
            label = f"{num_spk} người nói"
        else:
            header_color = "#29B6F6"
            icon = "✅"
            label = "1 người nói"

        timeline_html = _diar_timeline(segments, duration, dominant)

        # Phase 8: dominant speaker info line
        dominant_html = ""
        if dominant:
            other_spk = [s for s in {seg.get("speaker") for seg in segments} if s != dominant]
            if other_spk:
                stranger_label = ", ".join(sorted(other_spk))
                dominant_html = (
                    f'<div style="font-size:11px;color:#aaa;margin-bottom:3px;">'
                    f'👤 <span style="color:#29B6F6">{dominant}</span> (thí sinh) | '
                    f'⚠️ <span style="color:#FF9800">{stranger_label}</span> (người lạ)</div>'
                )
            else:
                dominant_html = (
                    f'<div style="font-size:11px;color:#aaa;margin-bottom:3px;">'
                    f'👤 <span style="color:#29B6F6">{dominant}</span> (thí sinh)</div>'
                )

        # Segment detail lines
        seg_lines = []
        for seg in segments:
            spk = seg.get("speaker", "?")
            s = seg.get("start", 0.0)
            e = seg.get("end", 0.0)
            is_dominant = spk == dominant
            spk_color = "#29B6F6" if is_dominant else "#FF9800"
            role = " (thí sinh)" if is_dominant else " (người lạ)"
            seg_lines.append(
                f'<span style="color:{spk_color};font-size:11px;margin-left:12px;">'
                f'[{spk}]{role} {s:.1f}s–{e:.1f}s ({e-s:.1f}s)</span><br>'
            )
        seg_detail = "".join(seg_lines)

        rows.append(
            f'<div style="background:#1a1a1a;border:1px solid #333;border-left:3px solid {header_color};'
            f'border-radius:4px;padding:8px 10px;margin-bottom:6px;">'
            f'<div style="color:{header_color};font-weight:bold;font-size:12px;margin-bottom:4px;">'
            f'{icon} [{ts}] {label} — {duration:.1f}s audio</div>'
            f'{dominant_html}'
            f'{timeline_html}'
            f'<div style="margin-top:4px">{seg_detail}</div>'
            f'</div>'
        )

    body = "\n".join(rows)
    st.markdown(
        f'<div style="max-height:340px;overflow-y:auto;">{body}</div>',
        unsafe_allow_html=True,
    )


def render_reverify_popup(student_id: str) -> None:
    """Phase 8: show re-verification prompt when verification fails."""
    if not st.session_state.get("show_reverify", False):
        return

    fails = st.session_state.get("verification_failures", 0)
    st.warning(
        f"⚠️ **Không nhận ra giọng nói (lần {fails})**\n\n"
        "Vui lòng đọc câu sau để xác nhận danh tính:"
    )
    st.info(f'🗣 *"Tôi là {student_id}, tôi đang tham gia kỳ thi hôm nay"*')

    col_rec, col_dismiss = st.columns([2, 1])
    with col_rec:
        enrollment_recorder_js(student_id, key_suffix="reverify")
    with col_dismiss:
        if st.button("✖ Bỏ qua", key="dismiss_reverify"):
            st.session_state.show_reverify = False
            st.rerun()


def render_report_panel(report: dict) -> None:
    st.markdown("---")
    st.subheader("📊 Báo cáo phiên thi")

    cheating = report.get("cheating_detected", False)
    verif_fails = report.get("verification_failures", 0)
    overlap_count = report.get("overlap_count", 0)
    transcript = report.get("transcript", [])
    elapsed = report.get("elapsed_seconds", 0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Kết quả", "🚨 CẢNH BÁO" if cheating else "✅ BÌNH THƯỜNG")
    col2.metric("Thời gian", f"{elapsed:.0f}s")
    col3.metric("ID Fail", verif_fails)
    col4.metric("Overlap", overlap_count)

    if cheating:
        st.error("🚨 Phát hiện nội dung liên quan đến câu hỏi thi hoặc xác minh danh tính thất bại nhiều lần.")
    else:
        st.success("✅ Không phát hiện hành vi đáng ngờ.")

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
    # Always drain queues first so session_state is up to date before rendering
    drain_queues()

    st.title("🎓 AI Exam Proctoring System")
    st.caption("Hệ thống giám sát thi cử thời gian thực — Phase 8")

    exams = load_exam_files()
    student_id, exam_name, exam_id, exam_content = render_sidebar(exams)

    # Phase 8: re-verify popup (shown above controls when triggered)
    if st.session_state.get("show_reverify", False):
        render_reverify_popup(student_id)
        st.markdown("---")

    render_controls(student_id, exam_name, exam_id, exam_content)
    render_status_bar()
    st.markdown("---")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("🎤 Microphone")
        if st.session_state.recording and st.session_state.session_id:
            # JS component handles mic capture + WebSocket sending
            mic_js_component(st.session_state.session_id, WS_URL)

            # Poll events (transcript, diarization, alerts) — always drain queues
            events = api_poll_events(st.session_state.session_id)
            if events:
                process_events(events)
            drain_queues()  # always flush _log_queue/_diar_queue into session_state

            # Poll status
            status = api_get_status(st.session_state.session_id)
            if status:
                st.session_state.cheating_flag = status.get("cheating_flag", False)
                elapsed = status.get("elapsed_time_seconds", 0)
                m, s = divmod(int(elapsed), 60)
                st.metric("Thời gian", f"{m:02d}:{s:02d}")
        else:
            st.info("Nhấn **Bắt đầu giám sát** để bật mic")

    with col_right:
        st.subheader("📊 Giám sát thời gian thực")
        render_suspicion_panel()

        tab_log, tab_diar = st.tabs(["🔍 Detection Log", "🎙 Speaker Diarization"])
        with tab_log:
            render_log_panel()
        with tab_diar:
            render_diarization_panel()

        if st.session_state.recording:
            time.sleep(1)
            st.rerun()

    if st.session_state.report:
        render_report_panel(st.session_state.report)


if __name__ == "__main__":
    main()
