# MOFPS Transcribe

Mobile-focused local transcription using Faster-Whisper.

## Features

- Upload audio or video and transcribe it.
- Standard and Advanced Accuracy upload modes.
- Near-live microphone transcription over secure WebSockets.
- Voice-level and pitch feedback in the browser.
- Copy and download complete transcripts.

## Local Setup

```powershell
.\.python313\python.exe -m pip install -r requirements.txt
.\.python313\python.exe -m uvicorn web_app:app --host 0.0.0.0 --port 8502
```

Open:

```text
http://localhost:8502
```

## How Live Mode Works

The browser captures microphone audio with an AudioWorklet, converts it to mono
16 kHz PCM, and sends it over a same-origin WebSocket. The server transcribes
short overlapping chunks with Faster-Whisper and returns updated text.

This does not use WebRTC, STUN, or TURN.

## Railway Deployment

Railway runs the app using `start.sh` and checks `/health`.

Recommended variables:

```text
WHISPER_CACHE_DIR=/data/whisper-models
OMP_NUM_THREADS=2
MAX_LIVE_SESSIONS=1
```

Attach a Railway volume at `/data` to preserve downloaded model files between
deployments. Use one replica because the Faster-Whisper model and live sessions
are held in process memory.

## Models

- **Fast Live**: `tiny`
- **Clear Live**: `base`
- **Standard Upload**: `small`
- **Advanced Accuracy Upload**: `medium`

Live audio is buffered in a bounded queue while Whisper transcribes. If the
server falls behind, old queued audio is discarded so the transcript stays
near-live instead of accumulating an ever-growing delay. Pressing Stop flushes
the final partial audio chunk before closing the microphone connection.
