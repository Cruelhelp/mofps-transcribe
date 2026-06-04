import json

import numpy as np
from fastapi.testclient import TestClient

import web_app


client = TestClient(web_app.app)


def test_health_does_not_load_model():
    web_app._models.clear()
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert web_app._models == {}


def test_index_contains_live_controls():
    response = client.get("/")

    assert response.status_code == 200
    assert "Start live transcription" in response.text


def test_upload_transcription(monkeypatch):
    monkeypatch.setattr(
        web_app,
        "transcribe_input",
        lambda audio, model_name, vad_filter=True: "mock transcript",
    )

    response = client.post(
        "/api/transcribe",
        data={"quality": "standard"},
        files={"file": ("test.wav", b"fake audio", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json() == {"text": "mock transcript"}


def test_live_websocket_accepts_silent_pcm_without_loading_model():
    web_app._models.clear()

    with client.websocket_connect("/ws/live") as websocket:
        websocket.send_text(json.dumps({"quality": "fast", "chunk_seconds": 3}))
        assert websocket.receive_json()["type"] == "ready"

        silence = np.zeros(48000, dtype=np.int16)
        websocket.send_bytes(silence.tobytes())
        assert websocket.receive_json() == {"type": "status", "message": "Listening..."}
        websocket.send_text(json.dumps({"type": "stop"}))
        assert websocket.receive_json()["type"] == "ready_to_stop"

    assert web_app._models == {}


def test_live_websocket_flushes_final_audio(monkeypatch):
    monkeypatch.setattr(
        web_app,
        "transcribe_input",
        lambda audio, model_name, vad_filter=True: "final words",
    )

    with client.websocket_connect("/ws/live") as websocket:
        websocket.send_text(json.dumps({"quality": "fast", "chunk_seconds": 4}))
        assert websocket.receive_json()["type"] == "ready"
        speech = np.full(16000, 12000, dtype=np.int16)
        websocket.send_bytes(speech.tobytes())
        websocket.send_text(json.dumps({"type": "stop"}))
        assert websocket.receive_json()["message"] == "Transcribing..."
        assert websocket.receive_json() == {"type": "transcript", "text": "final words"}
        assert websocket.receive_json()["type"] == "ready_to_stop"


def test_upload_rejects_unsupported_file():
    response = client.post(
        "/api/transcribe",
        data={"quality": "standard"},
        files={"file": ("test.txt", b"not audio", "text/plain")},
    )

    assert response.status_code == 400


def test_append_without_overlap():
    result = web_app.append_without_overlap(
        "Good morning everyone we will review",
        "we will review the report today",
    )

    assert result == "Good morning everyone we will review the report today"
