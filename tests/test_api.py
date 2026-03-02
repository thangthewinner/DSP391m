"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient

# Note: These tests require the app to be running with models loaded
# For now, they are basic structure tests


@pytest.mark.skip(reason="Requires running app with loaded models")
def test_health_endpoint():
    """Test health check endpoint."""
    from src.api.main import app

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "models_loaded" in data
    assert "version" in data


@pytest.mark.skip(reason="Requires running app with loaded models")
def test_start_exam():
    """Test starting exam session."""
    from src.api.main import app

    client = TestClient(app)
    response = client.post(
        "/api/exam/start",
        json={
            "user_id": "test_user",
            "exam_id": "test_exam",
            "exam_question": "Test question",
            "duration_minutes": 60,
        },
    )

    assert response.status_code == 201
    data = response.json()
    assert "session_id" in data
    assert "websocket_url" in data


def test_api_structure():
    """Test basic API structure without running server."""
    from src.api.main import app

    # Check routes are registered
    routes = [route.path for route in app.routes]

    assert "/" in routes
    assert "/health" in routes
    assert "/api/exam/start" in routes
    assert "/api/exam/stop" in routes
    assert "/api/exam/status/{session_id}" in routes
    assert "/ws/audio/{session_id}" in routes
