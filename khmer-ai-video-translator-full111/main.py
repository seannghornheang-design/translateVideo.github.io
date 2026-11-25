import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
OUTPUT_DIR = Path(tempfile.gettempdir()) / "khmer_ai_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Khmer AI Video Translator")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


def run_ffmpeg(args: list[str]) -> None:
    result = subprocess.run(
        ["ffmpeg", "-y", *args],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or "FFmpeg failed")[-2500:])


def srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, int(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1_000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def translate_to_khmer(client: OpenAI, text: str) -> str:
    result = client.chat.completions.create(
        model=os.getenv("OPENAI_TRANSLATION_MODEL", "gpt-4o-mini"),
        messages=[
            {
                "role": "system",
                "content": (
                    "Translate spoken dialogue into natural, clear Khmer. "
                    "Preserve meaning, names, numbers and tone. "
                    "Return only the Khmer translation."
                ),
            },
            {"role": "user", "content": text},
        ],
        temperature=0.2,
    )
    translated = result.choices[0].message.content
    if not translated:
        raise RuntimeError("The translation service returned an empty result.")
    return translated.strip()


def azure_speech(text: str, voice: str, output_path: Path) -> None:
    key = require_env("AZURE_SPEECH_KEY")
    region = require_env("AZURE_SPEECH_REGION")
    # Escape text for XML so punctuation and symbols cannot break SSML.
    from xml.sax.saxutils import escape

    safe_text = escape(text)
    ssml = (
        '<speak version="1.0" xml:lang="km-KH">'
        f'<voice name="{voice}">{safe_text}</voice>'
        "</speak>"
    )
    url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
    response = requests.post(
        url,
        headers={
            "Ocp-Apim-Subscription-Key": key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm",
        },
        data=ssml.encode("utf-8"),
        timeout=120,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Azure Speech error {response.status_code}: {response.text[:800]}")
    output_path.write_bytes(response.content)


def process_video(input_path: Path, voice: str, burn_subtitles: bool, job_id: str) -> Path:
    client = OpenAI(api_key=require_env("OPENAI_API_KEY"))
    work = OUTPUT_DIR / job_id
    work.mkdir(parents=True, exist_ok=True)

    audio_path = work / "source.wav"
    run_ffmpeg([
        "-i", str(input_path), "-vn", "-ac", "1", "-ar", "16000", str(audio_path)
    ])

    with audio_path.open("rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    segments = getattr(transcript, "segments", None) or []
    if not segments:
        raise RuntimeError("No speech segments were detected in this video.")

    # Create a silent Khmer voice track with the same duration as the source.
    duration_probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(input_path)],
        capture_output=True, text=True, timeout=60
    )
    if duration_probe.returncode != 0:
        raise RuntimeError("Could not read the video duration.")
    duration_ms = int(float(duration_probe.stdout.strip()) * 1000)

    from pydub import AudioSegment
    dubbed = AudioSegment.silent(duration=duration_ms, frame_rate=24000)
    srt_entries = []

    for index, segment in enumerate(segments, start=1):
        original_text = (getattr(segment, "text", "") or "").strip()
        if not original_text:
            continue

        start = max(0.0, float(segment.start))
        end = max(start + 0.2, float(segment.end))
        khmer_text = translate_to_khmer(client, original_text)

        speech_path = work / f"speech_{index}.wav"
        azure_speech(khmer_text, voice, speech_path)
        speech = AudioSegment.from_file(speech_path)

        # Fit long dubbed segments into the source segment window when possible.
        target_ms = max(250, int((end - start) * 1000))
        if len(speech) > target_ms:
            factor = min(2.0, len(speech) / target_ms)
            speech = speech.speedup(playback_speed=factor)

        position = int(start * 1000)
        if position < duration_ms:
            dubbed = dubbed.overlay(speech, position=position)
        srt_entries.append(
            f"{len(srt_entries) + 1}\n"
            f"{srt_timestamp(start)} --> {srt_timestamp(end)}\n"
            f"{khmer_text}\n"
        )

    dubbed_path = work / "khmer_voice.wav"
    dubbed.export(dubbed_path, format="wav")

    srt_path = work / "khmer_subtitles.srt"
    srt_path.write_text("\n".join(srt_entries), encoding="utf-8")

    output_path = OUTPUT_DIR / f"{job_id}_khmer.mp4"
    if burn_subtitles:
        # Copy the SRT to a simple path to reduce FFmpeg filter escaping issues.
        subtitle_copy = work / "subtitles.srt"
        shutil.copyfile(srt_path, subtitle_copy)
        subtitle_filter_path = str(subtitle_copy).replace("\\", "/").replace(":", r"\:")
        run_ffmpeg([
            "-i", str(input_path),
            "-i", str(dubbed_path),
            "-vf", f"subtitles='{subtitle_filter_path}'",
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "veryfast",
            "-c:a", "aac", "-shortest", str(output_path),
        ])
    else:
        run_ffmpeg([
            "-i", str(input_path),
            "-i", str(dubbed_path),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "veryfast",
            "-c:a", "aac", "-shortest", str(output_path),
        ])

    return output_path


@app.get("/", response_class=HTMLResponse)
def home():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.post("/api/translate")
async def translate_video(
    video: UploadFile = File(...),
    voice_type: str = Form("female"),
    burn_subtitles: bool = Form(True),
):
    allowed_types = {"video/mp4", "video/quicktime", "video/webm", "video/x-matroska"}
    if video.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Please upload an MP4, MOV, WEBM or MKV video.")

    voices = {
        "male": "km-KH-PisethNeural",
        "female": "km-KH-SreymomNeural",
    }
    if voice_type not in voices:
        raise HTTPException(status_code=400, detail="Choose male or female Khmer voice.")

    job_id = uuid.uuid4().hex
    suffix = Path(video.filename or "upload.mp4").suffix.lower() or ".mp4"
    input_path = OUTPUT_DIR / f"{job_id}_input{suffix}"

    try:
        content = await video.read()
        if not content:
            raise HTTPException(status_code=400, detail="The uploaded video is empty.")
        if len(content) > 300 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Video is larger than the 300 MB starter limit.")
        input_path.write_bytes(content)
        output_path = process_video(input_path, voices[voice_type], burn_subtitles, job_id)
        return {
            "message": "Translation finished",
            "download_url": f"/api/download/{output_path.name}",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        input_path.unlink(missing_ok=True)


@app.get("/api/download/{filename}")
def download_result(filename: str):
    # Only serve generated MP4 files from the output directory.
    if not filename.endswith("_khmer.mp4") or Path(filename).name != filename:
        raise HTTPException(status_code=404, detail="File not found.")
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File expired or not found.")
    return FileResponse(file_path, media_type="video/mp4", filename="khmer_translated.mp4")
