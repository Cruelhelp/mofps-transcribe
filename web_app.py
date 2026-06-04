import asyncio
import json
import os
import re
import tempfile
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from pydantic import BaseModel


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
SAMPLE_RATE = 16000
LIVE_MODELS = {"fast": "base", "clear": "medium"}
UPLOAD_MODELS = {"standard": "large", "advanced": "large-v2"}
MODEL_CACHE_DIR = os.environ.get("WHISPER_CACHE_DIR")
DEFAULT_WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))
MAX_LIVE_SECONDS = int(os.environ.get("MAX_LIVE_SECONDS", str(45 * 60)))
MAX_LIVE_MESSAGE_BYTES = 256 * 1024
MAX_LIVE_SESSIONS = int(os.environ.get("MAX_LIVE_SESSIONS", "1"))
ALLOWED_UPLOAD_SUFFIXES = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}

app = FastAPI(title="MOFPS Transcribe")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_models: dict[str, WhisperModel] = {}
_model_lock = threading.Lock()
_inference_lock = threading.Lock()
_activity_lock = threading.Lock()
_active_live_sessions = 0
_active_uploads = 0


class RefineRequest(BaseModel):
    text: str
    glossary: str = ""


def parse_glossary(glossary: str) -> tuple[list[str], list[tuple[str, str]]]:
    terms = []
    corrections = []
    for raw_line in glossary.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "=>" in line:
            wrong, correct = (part.strip() for part in line.split("=>", 1))
            if wrong and correct:
                corrections.append((wrong, correct))
                terms.append(correct)
        else:
            terms.append(line)
    return terms[:100], corrections[:100]


def refine_transcript(text: str, glossary: str = "") -> str:
    """Conservatively clean transcript structure without inventing new wording."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""

    _, corrections = parse_glossary(glossary)
    for wrong, correct in corrections:
        text = re.sub(rf"\b{re.escape(wrong)}\b", correct, text, flags=re.IGNORECASE)

    # Collapse accidental repeated words and repeated adjacent sentences.
    text = re.sub(r"\b(\w+)(?:\s+\1\b){2,}", r"\1", text, flags=re.IGNORECASE)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    cleaned = []
    previous = ""
    for sentence in sentences:
        sentence = sentence.strip(" ,")
        if not sentence:
            continue
        normalized = re.sub(r"[^a-z0-9]+", " ", sentence.lower()).strip()
        if normalized and normalized == previous:
            continue
        previous = normalized
        sentence = sentence[0].upper() + sentence[1:]
        if sentence[-1] not in ".!?":
            sentence += "."
        cleaned.append(sentence)
    return " ".join(cleaned)


def get_model(model_name: str) -> WhisperModel:
    with _model_lock:
        if model_name not in _models:
            _models.clear()
            options = {
                "device": DEFAULT_WHISPER_DEVICE,
                "compute_type": "int8",
                "cpu_threads": max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))),
            }
            if MODEL_CACHE_DIR:
                options["download_root"] = MODEL_CACHE_DIR
            _models[model_name] = WhisperModel(model_name, **options)
        return _models[model_name]


def transcribe_input(
    audio,
    model_name: str,
    vad_filter: bool = True,
    glossary: str = "",
) -> str:
    # Model loading and inference share one lock to prevent memory spikes on small hosts.
    with _inference_lock:
        model = get_model(model_name)
        terms, _ = parse_glossary(glossary)
        hotwords = ", ".join(terms)
        segments, _ = model.transcribe(
            audio,
            language="en",
            task="transcribe",
            beam_size=5,
            vad_filter=vad_filter,
            condition_on_previous_text=False,
            initial_prompt=(
                f"Use these names and terms when spoken: {hotwords}." if hotwords else None
            ),
            hotwords=hotwords or None,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            hallucination_silence_threshold=1.0,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return refine_transcript(text, glossary)


def append_without_overlap(existing: str, new_text: str) -> str:
    old_words = existing.split()
    new_words = new_text.split()
    max_overlap = min(20, len(old_words), len(new_words))

    for size in range(max_overlap, 0, -1):
        old_tail = [word.lower().strip(".,!?") for word in old_words[-size:]]
        new_head = [word.lower().strip(".,!?") for word in new_words[:size]]
        if old_tail == new_head:
            new_words = new_words[size:]
            break

    return " ".join(old_words + new_words).strip()


def reserve_upload() -> bool:
    global _active_uploads
    with _activity_lock:
        if _active_live_sessions:
            return False
        _active_uploads += 1
        return True


def release_upload() -> None:
    global _active_uploads
    with _activity_lock:
        _active_uploads = max(0, _active_uploads - 1)


def reserve_live_session() -> bool:
    global _active_live_sessions
    with _activity_lock:
        if _active_uploads or _active_live_sessions >= MAX_LIVE_SESSIONS:
            return False
        _active_live_sessions += 1
        return True


def release_live_session() -> None:
    global _active_live_sessions
    with _activity_lock:
        _active_live_sessions = max(0, _active_live_sessions - 1)


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/refine")
async def refine(request: RefineRequest):
    if len(request.text) > 500_000 or len(request.glossary) > 10_000:
        return JSONResponse({"error": "Transcript or glossary is too large."}, status_code=413)
    return {"text": refine_transcript(request.text, request.glossary)}


@app.post("/api/transcribe")
async def transcribe_upload(
    file: UploadFile = File(...),
    quality: str = Form("standard"),
    glossary: str = Form(""),
):
    model_name = UPLOAD_MODELS.get(quality)
    suffix = Path(file.filename or "").suffix.lower()
    if model_name is None:
        return JSONResponse({"error": "Unsupported transcription quality."}, status_code=400)
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        return JSONResponse({"error": "Please upload a supported audio or video file."}, status_code=400)
    if not reserve_upload():
        return JSONResponse(
            {"error": "Stop live transcription before starting an upload."},
            status_code=409,
        )

    temp_path = None
    total_bytes = 0
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = Path(temp_file.name)
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    return JSONResponse(
                        {"error": "This file is too large to transcribe."},
                        status_code=413,
                    )
                temp_file.write(chunk)

        text = await asyncio.to_thread(
            transcribe_input, str(temp_path), model_name, True, glossary
        )
        return {"text": text}
    except Exception:
        return JSONResponse(
            {"error": "Transcription failed. Please try another file or quality."},
            status_code=500,
        )
    finally:
        release_upload()
        if temp_path:
            temp_path.unlink(missing_ok=True)


async def receive_live_audio(
    websocket: WebSocket,
    audio_queue: asyncio.Queue,
    stop_event: asyncio.Event,
) -> None:
    received_samples = 0
    max_samples = SAMPLE_RATE * MAX_LIVE_SECONDS
    try:
        while not stop_event.is_set():
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                stop_event.set()
                break

            text = message.get("text")
            if text:
                try:
                    if json.loads(text).get("type") == "stop":
                        stop_event.set()
                        break
                except json.JSONDecodeError:
                    continue

            raw_audio = message.get("bytes")
            if not raw_audio:
                continue
            if len(raw_audio) > MAX_LIVE_MESSAGE_BYTES or len(raw_audio) % 2:
                await websocket.send_json(
                    {"type": "error", "message": "Invalid live audio packet."}
                )
                stop_event.set()
                break

            samples = np.frombuffer(raw_audio, dtype="<i2").astype(np.float32) / 32768.0
            received_samples += samples.size
            if received_samples > max_samples:
                await websocket.send_json(
                    {"type": "error", "message": "Maximum live session length reached."}
                )
                stop_event.set()
                break

            if audio_queue.full():
                try:
                    audio_queue.get_nowait()
                    audio_queue.task_done()
                except asyncio.QueueEmpty:
                    pass
            await audio_queue.put(samples)
    except WebSocketDisconnect:
        stop_event.set()


async def process_live_audio(
    websocket: WebSocket,
    audio_queue: asyncio.Queue,
    stop_event: asyncio.Event,
    model_name: str,
    chunk_samples: int,
    glossary: str,
) -> None:
    transcript = ""
    audio_buffer = np.empty(0, dtype=np.float32)
    overlap_samples = int(SAMPLE_RATE * 0.75)

    async def transcribe_chunk(chunk: np.ndarray, final: bool = False) -> None:
        nonlocal transcript
        rms = float(np.sqrt(np.mean(chunk**2))) if chunk.size else 0
        if chunk.size < SAMPLE_RATE // 2 or rms < 0.008:
            if not final:
                await websocket.send_json({"type": "status", "message": "Listening..."})
            return
        await websocket.send_json({"type": "status", "message": "Transcribing..."})
        text = await asyncio.to_thread(transcribe_input, chunk, model_name, False, glossary)
        if text:
            transcript = append_without_overlap(transcript, text)
            transcript = refine_transcript(transcript, glossary)
            await websocket.send_json({"type": "transcript", "text": transcript})
        if not final:
            await websocket.send_json({"type": "status", "message": "Listening..."})

    while not stop_event.is_set() or not audio_queue.empty():
        try:
            samples = await asyncio.wait_for(audio_queue.get(), timeout=0.2)
        except asyncio.TimeoutError:
            continue
        audio_buffer = np.concatenate((audio_buffer, samples))
        audio_queue.task_done()

        if audio_buffer.size >= chunk_samples:
            chunk = audio_buffer[:chunk_samples].copy()
            audio_buffer = audio_buffer[chunk_samples - overlap_samples :]
            await transcribe_chunk(chunk)

    if audio_buffer.size:
        await transcribe_chunk(audio_buffer.copy(), final=True)
    await websocket.send_json({"type": "ready_to_stop", "message": "Live transcription stopped"})


@app.websocket("/ws/live")
async def live_transcription(websocket: WebSocket):
    await websocket.accept()
    if not reserve_live_session():
        await websocket.send_json(
            {
                "type": "error",
                "message": "The transcriber is busy. Stop the upload or other live session first.",
            }
        )
        await websocket.close(code=1013)
        return

    stop_event = asyncio.Event()
    audio_queue: asyncio.Queue = asyncio.Queue(maxsize=8)
    try:
        config_message = await asyncio.wait_for(websocket.receive_text(), timeout=10)
        config = json.loads(config_message)
        model_name = LIVE_MODELS.get(config.get("quality"), LIVE_MODELS["fast"])
        chunk_seconds = min(8, max(3, int(config.get("chunk_seconds", 4))))
        glossary = str(config.get("glossary", ""))[:10_000]

        await websocket.send_json(
            {"type": "status", "message": f"Loading {model_name} live model..."}
        )
        await asyncio.to_thread(get_model, model_name)
        await websocket.send_json(
            {"type": "ready", "message": f"Live model ready: {model_name}"}
        )
        receiver = asyncio.create_task(
            receive_live_audio(websocket, audio_queue, stop_event)
        )
        processor = asyncio.create_task(
            process_live_audio(
                websocket,
                audio_queue,
                stop_event,
                model_name,
                SAMPLE_RATE * chunk_seconds,
                glossary,
            )
        )
        await receiver
        await processor
    except (WebSocketDisconnect, asyncio.TimeoutError, json.JSONDecodeError):
        stop_event.set()
    except Exception:
        stop_event.set()
        try:
            await websocket.send_json(
                {"type": "error", "message": "Live transcription stopped unexpectedly."}
            )
        except Exception:
            pass
    finally:
        release_live_session()




