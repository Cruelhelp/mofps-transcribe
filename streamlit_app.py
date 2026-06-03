from pathlib import Path
from tempfile import NamedTemporaryFile
import html
import os
import queue
import time
import wave

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from transcribe_audio import load_whisper_model, transcribe_audio_with_model


SUPPORTED_AUDIO_TYPES = ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]
UPLOAD_MODEL_OPTIONS = {
    "Standard": {
        "model": "small",
        "description": "Faster transcription for clear recordings and everyday use.",
    },
    "Advanced Accuracy": {
        "model": "medium",
        "description": "Heavier model for accents, noisy rooms, and longer meetings. Slower, but more careful.",
    },
}
HOSTED_UPLOAD_MODEL_OPTIONS = {
    "Standard": UPLOAD_MODEL_OPTIONS["Standard"],
    "Fast Upload": {
        "model": "base",
        "description": "Lighter option for free hosting. Faster, with slightly rougher wording.",
    },
    "Advanced Accuracy": UPLOAD_MODEL_OPTIONS["Advanced Accuracy"],
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

@st.cache_resource
def get_model(model_name: str):
    return load_whisper_model(model_name)


def get_rtc_configuration() -> dict:
    ice_servers = [
        {"urls": ["stun:stun.l.google.com:19302", "stun:global.stun.twilio.com:3478"]},
    ]

    turn_url = os.environ.get("TURN_URL")
    turn_username = os.environ.get("TURN_USERNAME")
    turn_credential = os.environ.get("TURN_CREDENTIAL")

    if turn_url:
        turn_server = {"urls": [turn_url]}
        if turn_username and turn_credential:
            turn_server["username"] = turn_username
            turn_server["credential"] = turn_credential
        ice_servers.append(turn_server)

    return {"iceServers": ice_servers}


def write_uploaded_audio(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".audio"
    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return Path(temp_file.name)


def show_copyable_transcript(transcript: str, file_stem: str) -> None:
    st.subheader("Transcript")
    st.text_area("Copy-ready transcript", transcript, height=360)

    escaped_transcript = html.escape(transcript)
    components.html(
        f"""
        <textarea id="transcript-copy-source" style="position:absolute; left:-9999px;">{escaped_transcript}</textarea>
        <button
            onclick="
                const text = document.getElementById('transcript-copy-source').value;
                navigator.clipboard.writeText(text);
                this.innerText = 'Copied';
                setTimeout(() => this.innerText = 'Copy transcript', 1600);
            "
            style="
                width: 100%;
                min-height: 44px;
                border: 0;
                border-radius: 8px;
                background: #1f77b4;
                color: white;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
            "
        >
            Copy transcript
        </button>
        """,
        height=58,
    )

    st.download_button(
        "Download transcript",
        transcript + "\n",
        file_name=f"{file_stem}-transcript.txt",
        mime="text/plain",
        use_container_width=True,
    )


def transcribe_with_feedback(audio_file, model_choice: dict, button_label: str) -> None:
    if not st.button(button_label, type="primary", use_container_width=True):
        return

    audio_path = write_uploaded_audio(audio_file)
    file_stem = Path(audio_file.name).stem
    model_name = model_choice["model"]

    try:
        progress = st.progress(0, text="Preparing audio...")
        with st.status("Transcribing audio", expanded=True) as status:
            st.write(f"Selected quality: {model_name}")
            st.write("Saving uploaded audio to a temporary file...")
            progress.progress(20, text="Audio ready")

            st.write("Loading Whisper model. First run can take longer while the model is cached.")
            model = get_model(model_name)
            progress.progress(50, text="Model loaded")

            st.write("Running transcription. Longer recordings and advanced quality take more time.")
            started_at = time.monotonic()
            transcript = transcribe_audio_with_model(audio_path, model)
            elapsed = time.monotonic() - started_at

            progress.progress(100, text="Transcription complete")
            st.write(f"Finished in {elapsed:.1f} seconds.")
            status.update(label="Transcription complete", state="complete", expanded=False)

        show_copyable_transcript(transcript, file_stem)
    finally:
        audio_path.unlink(missing_ok=True)


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


def frame_to_mono_float(frame) -> np.ndarray:
    audio = frame.to_ndarray()
    if audio.ndim == 2:
        audio = audio.mean(axis=0)

    audio = audio.astype(np.float32)
    max_abs = np.max(np.abs(audio)) if audio.size else 0
    if max_abs > 1:
        audio = audio / max_abs

    return audio


def estimate_pitch_hz(audio: np.ndarray, sample_rate: int) -> float | None:
    if audio.size < sample_rate * 0.05:
        return None

    audio = audio - np.mean(audio)
    if np.max(np.abs(audio)) < 0.02:
        return None

    min_hz = 80
    max_hz = 350
    min_lag = int(sample_rate / max_hz)
    max_lag = int(sample_rate / min_hz)
    corr = np.correlate(audio, audio, mode="full")[audio.size - 1 :]

    if max_lag >= corr.size:
        return None

    window = corr[min_lag:max_lag]
    if window.size == 0:
        return None

    lag = int(np.argmax(window) + min_lag)
    if lag <= 0:
        return None

    return sample_rate / lag


def get_voice_feedback(frames) -> tuple[int, str]:
    samples = [frame_to_mono_float(frame) for frame in frames[-12:]]
    audio = np.concatenate(samples)
    sample_rate = frames[-1].sample_rate
    rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0
    level = min(100, int(rms * 350))
    pitch = estimate_pitch_hz(audio, sample_rate)
    pitch_label = f"{pitch:.0f} Hz" if pitch else "No clear pitch"
    return level, pitch_label


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
        rtc_configuration=get_rtc_configuration(),
        media_stream_constraints={"video": False, "audio": True},
        audio_receiver_size=1024,
        sendback_audio=False,
    )

    transcript_box = st.empty()
    status_box = st.empty()
    level_box = st.empty()
    pitch_box = st.empty()

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
    level_box.progress(0, text="Voice level: waiting for microphone audio")
    pitch_box.caption("Pitch: waiting for microphone audio")

    while ctx.state.playing:
        if ctx.audio_receiver is None:
            time.sleep(0.2)
            continue

        try:
            new_frames = ctx.audio_receiver.get_frames(timeout=1)
            frames.extend(new_frames)
        except (TimeoutError, queue.Empty):
            continue

        if frames:
            level, pitch_label = get_voice_feedback(frames)
            level_box.progress(level, text=f"Voice level: {level}%")
            pitch_box.caption(f"Pitch: {pitch_label}")

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

st.markdown(
    """
    <style>
    .block-container {
        max-width: 760px;
        padding-top: 1.25rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    div[data-testid="stRadio"] label,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stFileUploader"] label,
    div[data-testid="stAudioInput"] label {
        font-size: 1rem;
        font-weight: 650;
    }

    div[data-testid="stButton"] button,
    div[data-testid="stDownloadButton"] button {
        min-height: 46px;
        border-radius: 8px;
        font-weight: 650;
    }

    textarea {
        font-size: 1rem !important;
        line-height: 1.5 !important;
    }

    @media (max-width: 640px) {
        .block-container {
            padding-top: 0.75rem;
            padding-left: 0.75rem;
            padding-right: 0.75rem;
        }

        h1 {
            font-size: 1.8rem !important;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] {
            gap: 0.35rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
    transcribe_with_feedback(audio_file, model_choice, button_label)
