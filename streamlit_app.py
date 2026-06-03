from pathlib import Path
from tempfile import NamedTemporaryFile
import os
import queue
import time
import wave

import numpy as np
import streamlit as st

from transcribe_audio import load_whisper_model, transcribe_audio_with_model


SUPPORTED_AUDIO_TYPES = ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]
UPLOAD_MODEL_OPTIONS = {
    "Standard": {
        "model": "small",
        "description": "Faster transcription for clear recordings and everyday use.",
    },
    "Higher Accuracy": {
        "model": "medium",
        "description": "Better for accents, noisy rooms, and longer meetings, but slower.",
    },
}
HOSTED_UPLOAD_MODEL_OPTIONS = {
    "Standard": UPLOAD_MODEL_OPTIONS["Standard"],
    "Fast Upload": {
        "model": "base",
        "description": "Lighter option for free hosting. Faster, with slightly rougher wording.",
    },
}
LIVE_MODEL_OPTIONS = {
    "Fast Live": {
        "model": "tiny",
        "description": "Fastest response for quick notes, with more transcription mistakes.",
    },
    "Clear Live": {
        "model": "base",
        "description": "Still quick, with better wording than Fast Live.",
    },
}
RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
}


@st.cache_resource
def get_model(model_name: str):
    return load_whisper_model(model_name)


def write_uploaded_audio(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".audio"
    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return Path(temp_file.name)


def write_audio_frames(frames) -> Path:
    sample_rate = frames[0].sample_rate
    samples = []

    for frame in frames:
        audio = frame.to_ndarray()
        if audio.ndim == 2:
            audio = audio.mean(axis=0)
        if np.issubdtype(audio.dtype, np.floating):
            audio = np.clip(audio, -1.0, 1.0)
            audio = (audio * 32767).astype(np.int16)
        else:
            audio = audio.astype(np.int16)
        samples.append(audio)

    chunk = np.concatenate(samples)

    with NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        with wave.open(temp_file, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(chunk.tobytes())
        return Path(temp_file.name)


def transcribe_live_stream(model_name: str, chunk_seconds: int) -> None:
    try:
        from streamlit_webrtc import WebRtcMode, webrtc_streamer
    except ImportError:
        st.warning(
            "Near-live microphone streaming is not installed in this deployment. "
            "Use Upload Recording or record a microphone clip instead."
        )
        return

    ctx = webrtc_streamer(
        key="near-live-transcription",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": False, "audio": True},
        audio_receiver_size=1024,
        sendback_audio=False,
    )

    transcript_box = st.empty()
    status_box = st.empty()

    if "live_transcript" not in st.session_state:
        st.session_state.live_transcript = ""

    if st.button("Clear Live Transcript"):
        st.session_state.live_transcript = ""

    transcript_box.text_area("Live transcript", st.session_state.live_transcript, height=300)

    if not ctx.state.playing:
        status_box.info("Press START above and allow microphone access.")
        return

    model = get_model(model_name)
    frames = []
    chunk_started_at = time.monotonic()
    status_box.info("Listening... text updates every few seconds.")

    while ctx.state.playing:
        if ctx.audio_receiver is None:
            time.sleep(0.2)
            continue

        try:
            frames.extend(ctx.audio_receiver.get_frames(timeout=1))
        except (TimeoutError, queue.Empty):
            continue

        if time.monotonic() - chunk_started_at < chunk_seconds:
            continue

        if not frames:
            chunk_started_at = time.monotonic()
            continue

        audio_path = write_audio_frames(frames)
        frames = []
        chunk_started_at = time.monotonic()

        try:
            text = transcribe_audio_with_model(audio_path, model)
            if text:
                st.session_state.live_transcript = (
                    st.session_state.live_transcript + " " + text
                ).strip()
                transcript_box.text_area(
                    "Live transcript",
                    st.session_state.live_transcript,
                    height=300,
                )
        finally:
            audio_path.unlink(missing_ok=True)


st.set_page_config(page_title="MOFPS Transcribe", layout="centered")

st.title("MOFPS Transcribe")
st.caption("Transcribe uploaded audio or record from your microphone.")

mode = st.radio("Mode", ["Upload Recording", "Live Microphone"], horizontal=True)
is_hugging_face_space = bool(os.environ.get("SPACE_ID"))

if mode == "Upload Recording":
    upload_options = HOSTED_UPLOAD_MODEL_OPTIONS if is_hugging_face_space else UPLOAD_MODEL_OPTIONS
    model_label = st.selectbox(
        "Transcription quality",
        list(upload_options),
        help="Choose the balance between speed and accuracy.",
    )
    model_choice = upload_options[model_label]
    st.caption(model_choice["description"])

    audio_file = st.file_uploader("Upload audio", type=SUPPORTED_AUDIO_TYPES)
    button_label = "Transcribe Recording"
else:
    model_label = st.selectbox(
        "Live quality",
        list(LIVE_MODEL_OPTIONS),
        help="Choose the balance between live speed and accuracy.",
    )
    model_choice = LIVE_MODEL_OPTIONS[model_label]
    st.caption(model_choice["description"])

    if is_hugging_face_space:
        st.warning(
            "Near-live microphone streaming is not reliable on Hugging Face free Spaces. "
            "Use Upload Recording here, or run the app locally for chunked live transcription."
        )
        audio_file = st.audio_input("Record a microphone clip")
        button_label = "Transcribe Microphone Clip"
    else:
        chunk_seconds = st.slider(
            "Update every",
            3,
            8,
            4,
            help="Shorter chunks update faster but may be less accurate.",
        )
        transcribe_live_stream(model_choice["model"], chunk_seconds)
        st.stop()

if audio_file:
    st.audio(audio_file)

    if st.button(button_label, type="primary"):
        audio_path = write_uploaded_audio(audio_file)

        try:
            with st.spinner("Loading Whisper and transcribing audio..."):
                model = get_model(model_choice["model"])
                transcript = transcribe_audio_with_model(audio_path, model)

            st.subheader("Transcript")
            st.text_area("Transcript text", transcript, height=300)
            st.download_button(
                "Download transcript",
                transcript + "\n",
                file_name=f"{Path(audio_file.name).stem}-transcript.txt",
                mime="text/plain",
            )
        finally:
            audio_path.unlink(missing_ok=True)
