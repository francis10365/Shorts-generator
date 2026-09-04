# Shorts Generator

Paste a long-form video link (YouTube, TikTok, Instagram, etc.) → get back a
45–50 second vertical clip, auto-selected for the strongest hook and best
retention, ready to post.

## How it works

1. **Download** — `yt-dlp` pulls the source video from the link.
2. **Transcribe** — a local Whisper model (`faster-whisper`) transcribes it with timestamps. Runs on CPU, no API cost.
3. **Pick the hook** — the transcript is sent to Claude, which picks the single 45–50s window with the strongest opening hook and a self-contained payoff.
4. **Render** — `ffmpeg` cuts that window and crops it to 1080×1920 (9:16 vertical).
5. **Mobile frontend** — a single page: paste link → progress bar → preview & download.

## Why this needs a server (not just your phone)

Downloading video, transcribing audio, and re-encoding video are all CPU-heavy
jobs — no phone browser can do this in-page. The fix: run the small server
below once, on a cheap host, then just open its URL from your phone's
browser. Bookmark it (or "Add to Home Screen") and it behaves like an app.

## 1. Local setup (to test first)

Requirements: Python 3.10+, `ffmpeg` installed on the machine, an Anthropic API key.

```bash
cd backend
pip install -r requirements.txt

export ANTHROPIC_API_KEY="your-key-here"
uvicorn app:app --host 0.0.0.0 --port 8000
