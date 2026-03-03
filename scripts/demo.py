"""
Demo script - test toàn bộ flow Phase 1.

Mô phỏng: Start session → Stream audio từ microphone → Nhận transcript → Stop → Xem report

Usage:
    uv run python scripts/demo.py              # Dùng microphone thật (mặc định)
    uv run python scripts/demo.py --fake       # Dùng sine wave giả (test pipeline)
"""

import argparse
import asyncio
import base64
import json
import time

import numpy as np
import requests
import websockets

BASE_URL = "http://localhost:8000"
WS_BASE = "ws://localhost:8000"


def check_server():
    print("  Waiting for server to be ready...", end="", flush=True)
    for _ in range(15):
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            data = r.json()
            if data["models_loaded"]:
                print(f"\n  ✅ Server: {data['status']} (v{data['version']})")
                return True
            print(".", end="", flush=True)
            time.sleep(2)
        except Exception:
            print(".", end="", flush=True)
            time.sleep(2)
    print("\n  ❌ Server not ready after 30s")
    print("  → Run: uv run uvicorn src.api.main:app --reload")
    return False


def start_session(student_id: str, exam_question: str) -> str:
    r = requests.post(f"{BASE_URL}/api/exam/start", json={
        "student_id": student_id,
        "exam_id": "demo_exam",
        "exam_question": exam_question,
    })
    data = r.json()
    session_id = data["session_id"]
    ws_url = data["websocket_url"]
    print(f"  Session ID: {session_id[:16]}...")
    print(f"  WebSocket:  {ws_url}")
    return session_id


def make_fake_audio_chunk(duration: float = 2.0, sample_rate: int = 16000) -> str:
    """Tạo sine wave giả để test pipeline (không phải tiếng nói thật)."""
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples, endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.1 * 32767).astype(np.int16)
    return base64.b64encode(audio.tobytes()).decode()


def record_mic_chunk(duration: float = 2.0, sample_rate: int = 16000) -> str:
    """Record audio từ microphone thật."""
    try:
        import sounddevice as sd
    except ImportError:
        print("  ⚠️  sounddevice not installed. Run: uv pip install sounddevice")
        raise

    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype=np.int16,
        blocking=True,
    )
    return base64.b64encode(audio.tobytes()).decode()


async def stream_audio(session_id: str, use_mic: bool = True, num_chunks: int = 5):
    ws_url = f"{WS_BASE}/ws/audio/{session_id}"
    sample_rate = 16000
    chunk_duration = 2.0

    print(f"\n  Connecting to {ws_url}...")

    async with websockets.connect(ws_url) as ws:
        print("  ✅ WebSocket connected")

        if use_mic:
            print(f"  🎤 Recording from microphone — {num_chunks} chunks × {chunk_duration}s")
            print(f"  ⏱  Total recording time: ~{num_chunks * chunk_duration:.0f}s")
            print("  Speak now!\n")
        else:
            print(f"  🔊 Sending fake audio (sine wave) — {num_chunks} chunks")
            print("  Note: Transcript sẽ bị filter vì confidence thấp\n")

        for i in range(num_chunks):
            if use_mic:
                print(f"  🔴 Recording chunk {i+1}/{num_chunks}...", end=" ", flush=True)
                loop = asyncio.get_event_loop()
                audio_b64 = await loop.run_in_executor(
                    None, record_mic_chunk, chunk_duration, sample_rate
                )
                print("done")
            else:
                audio_b64 = make_fake_audio_chunk(chunk_duration, sample_rate)

            msg = {
                "type": "audio_chunk",
                "data": audio_b64,
                "sample_rate": sample_rate,
                "timestamp": float(i * chunk_duration),
            }
            await ws.send(json.dumps(msg))

            try:
                resp = await asyncio.wait_for(ws.recv(), timeout=3.0)
                data = json.loads(resp)
                if data["type"] == "ack":
                    if not use_mic:
                        print(f"  → Chunk {i+1}/{num_chunks} ack'd")
                elif data["type"] == "status_update":
                    print(f"  📊 suspicion={data['suspicion_score']:.2f}, cheating={data['cheating_flag']}")
                elif data["type"] == "cheating_alert":
                    print(f"  🚨 {data['message']} (score={data['suspicion_score']:.2f})")
                elif data["type"] == "error":
                    print(f"  ❌ Error: {data['message']}")
            except asyncio.TimeoutError:
                pass

        print("\n  Closing WebSocket...")


def get_status(session_id: str):
    r = requests.get(f"{BASE_URL}/api/exam/status/{session_id}")
    data = r.json()
    print(f"  Status:          {data['status']}")
    print(f"  Suspicion score: {data['current_suspicion_score']:.2f}")
    print(f"  Cheating flag:   {data['cheating_flag']}")
    print(f"  Elapsed:         {data['elapsed_time_seconds']:.1f}s")
    print(f"  Flagged segments:{data['flagged_segments_count']}")


def stop_session(session_id: str):
    r = requests.post(f"{BASE_URL}/api/exam/stop/{session_id}")
    data = r.json()
    print(f"  Status: {data['status']}")
    print(f"  Report: {data['report_url']}")


def get_report(session_id: str):
    r = requests.get(f"{BASE_URL}/api/exam/report/{session_id}")
    data = r.json()
    print(f"  Student:         {data['student_id']}")
    print(f"  Exam question:   {data.get('exam_question', '')[:60]}")
    print(f"  Duration:        {data.get('elapsed_seconds', 0):.1f}s")

    # Decision Engine results
    print(f"\n  ── Decision Engine ──────────────────────")
    cheating = data.get("cheating_detected", False)
    print(f"  Cheating:        {'🚨 YES' if cheating else '✅ NO'}")
    print(f"  Suspicion score: {data.get('suspicion_score', 0):.2f} / 10.0 (threshold)")
    print(f"  Max score:       {data.get('max_suspicion_score', 0):.2f}")
    print(f"  Confidence:      {data.get('confidence', 0):.0%}")
    print(f"  Rationale:       {data.get('rationale', '')}")

    flagged = data.get("flagged_segments", [])
    if flagged:
        print(f"\n  ── Flagged Segments ({len(flagged)}) ───────────────")
        for seg in flagged:
            print(f"    [{seg['timestamp']:.1f}s] similarity={seg['similarity_score']:.2f} +{seg['points_added']:.0f}pt")
            print(f"    \"{seg['text'][:70]}\"")

    # Transcript
    transcript = data.get("transcript", [])
    if transcript:
        print(f"\n  ── Transcript ({len(transcript)} segments) ──────────")
        for seg in transcript:
            print(f"    [{seg['start']:.1f}s → {seg['end']:.1f}s] conf={seg['confidence']:.2f}")
            print(f"    \"{seg['text']}\"")
    else:
        print("\n  Transcript: (empty)")
        print("  → VAD filtered audio as silence, or confidence too low")
        print("  → Use real microphone audio for actual transcription")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", action="store_true", help="Dùng sine wave thay vì microphone")
    parser.add_argument("--chunks", type=int, default=5, help="Số chunks (mỗi chunk 2s)")
    args = parser.parse_args()

    use_mic = not args.fake

    print("=" * 55)
    print("  DSP391m — Phase 1 Demo")
    print(f"  Mode: {'🎤 Microphone' if use_mic else '🔊 Fake audio (sine wave)'}")
    print("=" * 55)

    # 1. Check server
    print("\n[1] Health Check")
    if not check_server():
        return

    # 2. Start session
    print("\n[2] Start Exam Session")
    exam_question = "Giải thích nguyên lý hoạt động của mạng nơ-ron nhân tạo"
    session_id = start_session("demo_student", exam_question)

    # 3. Stream audio
    print("\n[3] Stream Audio (WebSocket)")
    asyncio.run(stream_audio(session_id, use_mic=use_mic, num_chunks=args.chunks))

    # Wait for pipeline to process
    print("\n  Waiting for pipeline to process...")
    time.sleep(3)

    # 4. Check status
    print("\n[4] Session Status")
    get_status(session_id)

    # 5. Stop session
    print("\n[5] Stop Session")
    stop_session(session_id)

    # 6. Get report
    print("\n[6] Final Report")
    get_report(session_id)

    print("\n" + "=" * 55)
    print("  Demo complete!")
    print("=" * 55)
    print("""
Next steps:
  • Open http://localhost:8000/docs for interactive API
  • Use real microphone audio for actual transcription
  • Phase 2: Text embedding similarity (coming next)
""")


if __name__ == "__main__":
    main()
