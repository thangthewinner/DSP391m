"""
WebSocket Audio Streaming Test.

Simulates sending audio chunks via WebSocket and receiving transcriptions.

Usage:
    uv run python scripts/test_websocket.py
    uv run python scripts/test_websocket.py --session-id my-session
"""

import argparse
import asyncio
import json
import struct
import wave
import numpy as np


async def test_websocket(base_url: str, session_id: str):
    try:
        import websockets
    except ImportError:
        print("Install websockets: uv pip install websockets")
        return

    import requests

    ws_url = f"ws://{base_url.replace('http://', '')}/ws/audio/{session_id}"
    print(f"\n{'='*55}")
    print(f"  WebSocket Test")
    print(f"  URL: {ws_url}")
    print(f"{'='*55}")

    # Generate silent audio (16-bit PCM, 16kHz, 1s)
    sample_rate = 16000
    duration = 1.0
    samples = int(sample_rate * duration)
    # Generate a simple sine wave (440Hz) to simulate speech
    t = np.linspace(0, duration, samples, endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.1 * 32767).astype(np.int16)
    audio_bytes = audio.tobytes()
    audio_b64 = __import__("base64").b64encode(audio_bytes).decode()

    print(f"\n  Connecting to WebSocket...")
    try:
        async with websockets.connect(ws_url) as ws:
            print("  ✅ Connected!")

            # Send a few audio chunks
            for i in range(3):
                msg = {
                    "type": "audio_chunk",
                    "data": audio_b64,
                    "sample_rate": sample_rate,
                    "timestamp": i * duration,
                }
                await ws.send(json.dumps(msg))
                print(f"\n  → Sent chunk {i+1}/3 ({len(audio_bytes)} bytes)")

                # Wait for response
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    data = json.loads(response)
                    print(f"  ← Received: {json.dumps(data, indent=4, ensure_ascii=False)}")
                except asyncio.TimeoutError:
                    print("  ← No response (timeout)")

                await asyncio.sleep(0.5)

            # Send stop signal
            await ws.send(json.dumps({"type": "stop"}))
            print("\n  → Sent stop signal")

            # Wait for final response
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(response)
                print(f"  ← Final response: {json.dumps(data, indent=4, ensure_ascii=False)}")
            except asyncio.TimeoutError:
                print("  ← No final response")

    except Exception as e:
        print(f"  ❌ WebSocket error: {e}")
        return

    print("\n  ✅ WebSocket test complete!")


def main():
    parser = argparse.ArgumentParser(description="WebSocket Test")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--session-id", default=None)
    args = parser.parse_args()

    # Create a session first if not provided
    if not args.session_id:
        import requests
        try:
            r = requests.post(f"{args.base_url}/api/exam/start", json={
                "student_id": "ws_test_student",
                "exam_id": "ws_test_exam",
            })
            if r.status_code == 200:
                args.session_id = r.json()["session_id"]
                print(f"Created session: {args.session_id}")
            else:
                print(f"Failed to create session: {r.status_code}")
                args.session_id = "test-session-fallback"
        except Exception as e:
            print(f"Cannot create session: {e}")
            args.session_id = "test-session-fallback"

    asyncio.run(test_websocket(args.base_url, args.session_id))


if __name__ == "__main__":
    main()
