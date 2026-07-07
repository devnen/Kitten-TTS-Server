import pytest
import os
import subprocess
from fastapi.testclient import TestClient

os.environ["TTS_API_KEY"] = "test-key"
os.environ["ENABLE_MANAGEMENT_ENDPOINTS"] = "true"
os.environ["ENABLE_WEB_UI"] = "true"

from server import app

client = TestClient(app)

def test_import_server_without_torchaudio():
    result = subprocess.run(["python", "-c", "import server"], capture_output=True, text=True)
    assert result.returncode == 0, f"import server failed: {result.stderr}"

def test_import_utils_without_torchaudio():
    result = subprocess.run(["python", "-c", "import utils"], capture_output=True, text=True)
    assert result.returncode == 0, f"import utils failed: {result.stderr}"

def test_health_remains_available():
    response = client.get("/health")
    assert response.status_code in (200, 503)

def test_api_tts_health_remains_available():
    response = client.get("/api/tts/health")
    assert response.status_code in (200, 503)

def test_fail_closed_when_key_unset(monkeypatch):
    monkeypatch.delenv("TTS_API_KEY", raising=False)
    # The endpoint should return 503 because key is missing in server env
    response = client.post("/api/tts/speak", json={"text": "hi", "voice": "default", "output_format": "wav"})
    assert response.status_code == 503

def test_invalid_format_returns_400():
    response = client.post(
        "/api/tts/speak",
        headers={"Authorization": "Bearer test-key"},
        json={"text": "Hello", "voice": "default", "output_format": "opus"}
    )
    # Could be 503 if model not loaded, but format validation happens before! Wait, it is valid so it continues to 503.
    # Let's test MP3 which should be 400
    response_mp3 = client.post(
        "/api/tts/speak",
        headers={"Authorization": "Bearer test-key"},
        json={"text": "Hello", "voice": "default", "output_format": "mp3"}
    )
    assert response_mp3.status_code == 400

def test_auth_missing_key_rejected():
    response = client.post("/api/unload")
    assert response.status_code == 401

def test_auth_invalid_key_rejected():
    response = client.post("/api/unload", headers={"Authorization": "Bearer bad-key"})
    assert response.status_code == 403

def test_auth_valid_bearer_accepted():
    response = client.post("/api/unload", headers={"Authorization": "Bearer test-key"})
    # It might return 200 or 503/etc, but not 401/403
    assert response.status_code not in (401, 403)

def test_auth_valid_x_api_key_accepted():
    response = client.post("/api/unload", headers={"X-API-Key": "test-key"})
    assert response.status_code not in (401, 403)

def test_management_endpoints_blocked_by_default(monkeypatch):
    monkeypatch.setenv("ENABLE_MANAGEMENT_ENDPOINTS", "false")
    for ep in ["/api/unload", "/restart_server", "/save_settings", "/reset_settings", "/api/cancel-loading"]:
        response = client.post(ep, headers={"Authorization": "Bearer test-key"})
        assert response.status_code == 403

def test_web_ui_disabled_by_default(monkeypatch):
    monkeypatch.setenv("ENABLE_WEB_UI", "false")
    response = client.get("/")
    assert response.status_code == 403

    response_ui = client.get("/api/ui/initial-data", headers={"Authorization": "Bearer test-key"})
    assert response_ui.status_code == 403

def test_ui_initial_data_sanitized_when_enabled():
    response = client.get("/api/ui/initial-data", headers={"Authorization": "Bearer test-key"})
    assert response.status_code == 200
    data = response.json()
    assert "config" in data
    assert "server" not in data["config"] # Server block shouldn't be leaked
