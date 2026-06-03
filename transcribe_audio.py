import argparse
import os
import shutil
from pathlib import Path

MODEL_NAMES = ["tiny", "base", "small", "medium"]


def load_whisper_model(model_name: str):
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unsupported Whisper model: {model_name}")

    import whisper

    ensure_ffmpeg_on_path()
    download_root = os.environ.get("WHISPER_CACHE_DIR")
    if download_root:
        return whisper.load_model(model_name, download_root=download_root)

    return whisper.load_model(model_name)


def ensure_ffmpeg_on_path() -> None:
    import imageio_ffmpeg

    bundled_ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
    ffmpeg_dir_path = Path(__file__).resolve().parent / ".ffmpeg-bin"
    ffmpeg_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    ffmpeg_path = ffmpeg_dir_path / ffmpeg_name

    if not ffmpeg_path.exists():
        ffmpeg_dir_path.mkdir(exist_ok=True)
        shutil.copy2(bundled_ffmpeg, ffmpeg_path)
        ffmpeg_path.chmod(0o755)

    ffmpeg_dir = str(ffmpeg_dir_path)
    current_path = os.environ.get("PATH", "")

    if ffmpeg_dir not in current_path.split(os.pathsep):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + current_path


def transcribe_audio(audio_path: Path, model_name: str) -> str:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = load_whisper_model(model_name)
    return transcribe_audio_with_model(audio_path, model)


def transcribe_audio_with_model(audio_path: Path, model) -> str:
    result = model.transcribe(str(audio_path))
    return result["text"].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe audio with OpenAI Whisper.")
    parser.add_argument("audio", type=Path, help="Path to the audio file to transcribe.")
    parser.add_argument(
        "-m",
        "--model",
        default="small",
        choices=MODEL_NAMES,
        help="Whisper model to use. Tiny/base are best for live speed; small/medium are best for uploaded recordings.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional text file where the transcript should be saved.",
    )
    args = parser.parse_args()

    transcript = transcribe_audio(args.audio, args.model)
    print(transcript)

    if args.output:
        args.output.write_text(transcript + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
