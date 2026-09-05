"""
Core pipeline for turning a long-form video URL into a 45-50s vertical short.

Steps:
  1. download_video()   - pulls the source video with yt-dlp
  2. transcribe()       - runs local Whisper for a word/segment-level transcript
  3. pick_highlight()   - asks Claude to choose the strongest 45-50s window
  4. render_short()     - ffmpeg cuts that window and crops it to 9:16

Each step is a plain function so you can test/swap pieces independently
(e.g. replace faster-whisper with a hosted transcription API later).
"""

import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import anthropic
import yt_dlp
from faster_whisper import WhisperModel

WORK_DIR = Path(os.environ.get("SHORTS_WORK_DIR", "/tmp/shorts-jobs"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")
TARGET_MIN_SEC = 45
TARGET_MAX_SEC = 50

_whisper_model = None


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _whisper_model


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


def _cookiefile_from_env() -> str | None:
    cookies_text = os.environ.get("YT_COOKIES")
    if not cookies_text:
        return None
    cookie_path = WORK_DIR / "yt_cookies.txt"
    cookie_path.write_text(cookies_text)
    return str(cookie_path)


def download_video(url: str, job_id: str) -> Path:
    out_path = WORK_DIR / f"{job_id}_source.mp4"

    base_opts = {
        "outtmpl": str(out_path),
        "merge_output_format": "mp4",
        "quiet": True,
        "noprogress": True,
        "retries": 10,
        "fragment_retries": 10,
        "socket_timeout": 30,
    }
    cookiefile = _cookiefile_from_env()
    if cookiefile:
        base_opts["cookiefile"] = cookiefile

    attempts = [
        {"format": "best[ext=mp4][height<=480]/best[height<=480]/best",
         "extractor_args": {"youtube": {"player_client": ["ios"]}}},
        {"format": "best[ext=mp4][height<=480]/best[height<=480]/best",
         "extractor_args": {"youtube": {"player_client": ["android"]}}},
        {"format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
         "extractor_args": {"youtube": {"player_client": ["web"]}}},
    ]

    last_error = None
    for i, attempt_opts in enumerate(attempts):
        if out_path.exists():
            out_path.unlink()
        ydl_opts = {**base_opts, **attempt_opts}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            if out_path.exists():
                return out_path
        except Exception as e:  # noqa: BLE001
            last_error = e
            if i < len(attempts) - 1:
                time.sleep(2)
            continue

    raise RuntimeError(
        f"Could not download this video after trying multiple methods. "
        f"YouTube may be blocking this server. Last error: {last_error}"
    )


def transcribe(video_path: Path) -> list[TranscriptSegment]:
    model = _get_whisper()
    segments, _info = model.transcribe(str(video_path), beam_size=5, vad_filter=True)
    return [TranscriptSegment(start=s.start, end=s.end, text=s.text.strip()) for s in segments]


def _get_video_duration(video_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def pick_highlight(segments: list[TranscriptSegment], video_duration: float) -> dict:
    client = anthropic.Anthropic()

    transcript_text = "\n".join(
        f"[{seg.start:.1f}-{seg.end:.1f}] {seg.text}" for seg in segments
    )

    prompt = f"""You are selecting a clip from a longer video to repurpose as a short-form
vertical video (like a Reel or TikTok). The video is {video_duration:.0f} seconds long.

Below is the timestamped transcript. Choose ONE contiguous window between
{TARGET_MIN_SEC} and {TARGET_MAX_SEC} seconds long that will perform best as a standalone short:
- Opens with a hook: a surprising claim, question, or high-stakes statement in the first 2-3 seconds
- Is self-contained (a viewer with no other context understands and stays engaged)
- Has a natural, satisfying end point (payoff, punchline, or cliffhanger)
- Avoids starting or ending mid-sentence where possible

Transcript:
{transcript_text}

Respond with ONLY a JSON object, no other text, in this exact shape:
{{"start": <seconds as float>, "end": <seconds as float>, "reason": "<one sentence why this clip works>"}}
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = "".join(block.text for block in response.content if block.type == "text").strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    choice = json.loads(raw)

    start = max(0.0, float(choice["start"]))
    end = min(video_duration, float(choice["end"]))
    if end - start < TARGET_MIN_SEC:
        end = min(video_duration, start + TARGET_MIN_SEC)
    if end - start > TARGET_MAX_SEC:
        end = start + TARGET_MAX_SEC

    return {"start": start, "end": end, "reason": choice.get("reason", "")}


def render_short(video_path: Path, job_id: str, start: float, end: float) -> Path:
    out_path = WORK_DIR / f"{job_id}_short.mp4"
    duration = end - start

    vf = (
        "scale=-2:1920,"
        "crop=1080:1920"
    )

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(video_path),
        "-t", str(duration),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k",
        str(out_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return out_path


def run_pipeline(url: str, job_id: str, progress_cb=None) -> Path:
    def report(msg):
        if progress_cb:
            progress_cb(msg)

    report("Downloading video...")
    video_path = download_video(url, job_id)

    report("Transcribing...")
    duration = _get_video_duration(video_path)
    segments = transcribe(video_path)

    report("Finding the best hook...")
    highlight = pick_highlight(segments, duration)

    report("Rendering vertical short...")
    short_path = render_short(video_path, job_id, highlight["start"], highlight["end"])

    report("Done")
    return short_path, highlight
