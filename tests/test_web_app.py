import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

import web_app


client = TestClient(web_app.app)


@pytest.fixture(autouse=True)
def avoid_model_downloads(monkeypatch):
    monkeypatch.setattr(web_app, "get_model", lambda model_name: object())


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
        lambda audio, model_name, vad_filter=True, glossary="": "mock transcript",
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
        assert websocket.receive_json()["message"] == "Loading base live model..."
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
        lambda audio, model_name, vad_filter=True, glossary="": "final words",
    )

    with client.websocket_connect("/ws/live") as websocket:
        websocket.send_text(json.dumps({"quality": "fast", "chunk_seconds": 4}))
        assert websocket.receive_json()["message"] == "Loading base live model..."
        assert websocket.receive_json()["type"] == "ready"
        speech = np.full(16000, 12000, dtype=np.int16)
        websocket.send_bytes(speech.tobytes())
        websocket.send_text(json.dumps({"type": "stop"}))
        assert websocket.receive_json()["message"] == "Transcribing..."
        assert websocket.receive_json() == {"type": "transcript", "text": "Final words."}
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


def test_refine_transcript_applies_corrections_and_removes_repetition():
    result = web_app.refine_transcript(
        "m o f p s met today. m o f p s met today. very very very useful",
        "m o f p s => MOFPS",
    )

    assert result == "MOFPS met today. Very useful."


def test_refine_endpoint():
    response = client.post(
        "/api/refine",
        json={"text": "good morning everyone", "glossary": ""},
    )

    assert response.status_code == 200
    assert response.json() == {"text": "Good morning everyone."}


def test_parse_glossary_separates_hotwords_and_corrections():
    terms, corrections = web_app.parse_glossary("MOFPS\nmof ps => MOFPS")

    assert terms == ["MOFPS", "MOFPS"]
    assert corrections == [("mof ps", "MOFPS")]
