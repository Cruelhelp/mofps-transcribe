---
title: MOFPS Transcribe
sdk: docker
app_file: app.py
app_port: 7860
---

# Audio Transcription

This project uses OpenAI Whisper for local audio transcription.

Hosted app:

```text
https://ruelmcneil-mofps-transcribe.hf.space
```

## Setup

```powershell
.\.python313\python.exe -m pip install -r requirements.txt
```

The app uses `imageio-ffmpeg` so Whisper can find a local `ffmpeg` executable.

## Browser UI

```powershell
.\.python313\python.exe -m streamlit run .\streamlit_app.py --server.port 8502
```

If uploads fail behind a hosted proxy, run Streamlit with:

```powershell
.\.python313\python.exe -m streamlit run .\streamlit_app.py --server.port 8502 --server.enableCORS false --server.enableXsrfProtection false
```

Then choose either **Upload Recording** or **Live Microphone**.

Upload recording options:

- **Standard**: faster transcription for clear recordings and everyday use.
- **Higher Accuracy**: better for accents, noisy rooms, and longer meetings, but slower.

Live microphone options:

- **Fast Live**: fastest response for quick notes, with more transcription mistakes.
- **Clear Live**: still quick, with better wording than Fast Live.

Live microphone mode streams audio from the browser and transcribes it in short chunks when running locally. On Hugging Face free Spaces, microphone streaming may fail because WebRTC needs network support that the hosted proxy does not reliably provide, so the hosted app falls back to transcribing a recorded microphone clip.

## Railway Deployment

Railway can run this app as a long-lived Streamlit service.

Before deploying, make sure these files are committed:

- `streamlit_app.py`
- `transcribe_audio.py`
- `requirements.txt`
- `railway.json`
- `.python-version`

Do not commit the local `.python313` folder or generated log files.

For faster cold starts after redeploys, attach a Railway volume and set:

```text
WHISPER_CACHE_DIR=/data/whisper-cache
```

## Command-Line Usage

```powershell
.\.python313\python.exe .\transcribe_audio.py .\audio-file.mp3
```

Save the transcript to a text file:

```powershell
.\.python313\python.exe .\transcribe_audio.py .\audio-file.mp3 -o .\transcript.txt
```

Use a different model:

```powershell
.\.python313\python.exe .\transcribe_audio.py .\audio-file.mp3 --model medium
```
