"""
API Test Script - Run this to verify all endpoints work correctly.

Usage:
    uv run python scripts/test_api.py
    uv run python scripts/test_api.py --base-url http://localhost:8000
"""

import argparse
import json
import sys
import time

import requests

BASE_URL = "http://localhost:8000"


def show(title: str, r: requests.Response):
    status_icon = "✅" if r.status_code < 400 else "❌"
    print(f"\n{'─'*55}")
    print(f"  {status_icon}  {title}")
    print(f"{'─'*55}")
    print(f"  HTTP {r.status_code}")
    try:
        print(json.dumps(r.json(), indent=2, ensure_ascii=False))
    except Exception:
        print(r.text[:500])


def test_health():
    print("\n" + "="*55)
    print("  TEST 1: Health Check")
    print("="*55)
    r = requests.get(f"{BASE_URL}/health")
    show("GET /health", r)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert data.get("status") == "healthy", f"Expected healthy, got {data.get('status')}"
    print("\n  ✅ PASSED")
    return True


def test_exam_lifecycle():
    print("\n" + "="*55)
    print("  TEST 2: Exam Session Lifecycle")
    print("="*55)

    # Start session
    r = requests.post(f"{BASE_URL}/api/exam/start", json={
        "student_id": "test_student_001",
        "exam_id": "exam_001"
    })
    show("POST /api/exam/start", r)
    assert r.status_code == 200, f"Start failed: {r.status_code}"
    session_id = r.json().get("session_id")
    assert session_id, "No session_id in response"
    print(f"\n  Session ID: {session_id}")

    # Get status
    r = requests.get(f"{BASE_URL}/api/exam/status/{session_id}")
    show(f"GET /api/exam/status/{session_id[:8]}...", r)
    assert r.status_code == 200

    # List sessions
    r = requests.get(f"{BASE_URL}/api/exam/sessions")
    show("GET /api/exam/sessions", r)
    assert r.status_code == 200

    # Stop session
    r = requests.post(f"{BASE_URL}/api/exam/stop/{session_id}")
    show(f"POST /api/exam/stop/{session_id[:8]}...", r)
    assert r.status_code == 200

    # Get report
    r = requests.get(f"{BASE_URL}/api/exam/report/{session_id}")
    show(f"GET /api/exam/report/{session_id[:8]}...", r)
    assert r.status_code == 200

    print("\n  ✅ PASSED")
    return session_id


def test_websocket_info():
    """Print WebSocket test instructions."""
    print("\n" + "="*55)
    print("  TEST 3: WebSocket Audio Streaming")
    print("="*55)
    print("""
  WebSocket endpoint: ws://localhost:8000/ws/audio/{session_id}
  
  To test manually:
  
  Option A - Using wscat (npm):
    npm install -g wscat
    wscat -c ws://localhost:8000/ws/audio/test-session-id
  
  Option B - Using Python:
    uv run python scripts/test_websocket.py
  
  Option C - Using browser console:
    const ws = new WebSocket('ws://localhost:8000/ws/audio/test-id');
    ws.onmessage = (e) => console.log(JSON.parse(e.data));
    ws.send(JSON.stringify({type: 'audio_chunk', data: '', sample_rate: 16000}));
  """)


def test_error_handling():
    print("\n" + "="*55)
    print("  TEST 4: Error Handling")
    print("="*55)

    # Non-existent session
    r = requests.get(f"{BASE_URL}/api/exam/status/non-existent-session")
    show("GET /api/exam/status/non-existent (should be 404)", r)
    assert r.status_code == 404, f"Expected 404, got {r.status_code}"

    # Invalid endpoint
    r = requests.get(f"{BASE_URL}/api/invalid-endpoint")
    show("GET /api/invalid-endpoint (should be 404)", r)
    assert r.status_code == 404

    print("\n  ✅ PASSED")


def test_docs():
    print("\n" + "="*55)
    print("  TEST 5: API Documentation")
    print("="*55)

    r = requests.get(f"{BASE_URL}/docs")
    show("GET /docs (Swagger UI)", r)
    assert r.status_code == 200

    r = requests.get(f"{BASE_URL}/openapi.json")
    show("GET /openapi.json", r)
    assert r.status_code == 200
    endpoints = list(r.json().get("paths", {}).keys())
    print(f"\n  Available endpoints ({len(endpoints)}):")
    for ep in sorted(endpoints):
        print(f"    {ep}")

    print("\n  ✅ PASSED")


def main():
    parser = argparse.ArgumentParser(description="API Test Script")
    parser.add_argument("--base-url", default=BASE_URL, help="Base URL")
    args = parser.parse_args()

    global BASE_URL
    BASE_URL = args.base_url

    print(f"\n{'='*55}")
    print(f"  AI Proctoring API Test Suite")
    print(f"  Target: {BASE_URL}")
    print(f"{'='*55}")

    # Check server is running
    try:
        requests.get(f"{BASE_URL}/health", timeout=3)
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Cannot connect to {BASE_URL}")
        print("   Make sure server is running:")
        print("   uv run uvicorn src.api.main:app --reload")
        sys.exit(1)

    passed = 0
    failed = 0
    tests = [
        ("Health Check", test_health),
        ("Exam Lifecycle", test_exam_lifecycle),
        ("Error Handling", test_error_handling),
        ("API Docs", test_docs),
    ]

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"\n  ❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n  ❌ ERROR: {e}")
            failed += 1

    test_websocket_info()

    print(f"\n{'='*55}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*55}\n")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
