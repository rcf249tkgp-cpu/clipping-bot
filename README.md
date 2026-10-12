# clipper

Turns long videos into short vertical clips (1080×1920) ready for TikTok, Reels and Shorts.

```
input/<creator>/video.mp4
   │ 1. transcribe locally (faster-whisper, word timestamps)
   │ 2. Claude picks the best self-contained 20–60s moments with a strong hook
   │ 3. FFmpeg cuts, crops to 9:16 centered on the speaker, burns word-by-word captions
   ▼
output/ready/<creator>/<YYYY-MM-DD>/<video>_01_<title>.mp4   + .txt (caption + hashtags)
tracking/clips.csv                                          one row per clip
```

## Install

You need **Python 3.10+** and **FFmpeg**.

### 1. FFmpeg

| OS | Command |
|---|---|
| macOS | `brew install ffmpeg` (install [Homebrew](https://brew.sh) first if needed) |
| Windows | `winget install Gyan.FFmpeg`, then open a **new** terminal |
| Ubuntu/Debian | `sudo apt install ffmpeg` |

Check it works: `ffmpeg -version`

### 2. Python packages

From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. API key

```bash
cp .env.example .env               # Windows: copy .env.example .env
```

Open `.env` and paste your key from <https://console.anthropic.com/> after `ANTHROPIC_API_KEY=`.

## Use

1. Put videos in `input/<creator>/`, e.g. `input/examplecreator/episode1.mp4`.
2. Add the creator to `config/creators.yaml` (folders without an entry use the defaults).
3. Run:

```bash
python -m clipper                          # every creator
python -m clipper --creator examplecreator # one creator
python -m clipper --force                  # redo videos that were already processed
```

Videos already processed are listed in `tracking/processed.json` and skipped. Delete a video's line there (or use `--force`) to redo it.

The first run downloads the Whisper model (`small` is about 500 MB). Transcription runs on your CPU, or on an NVIDIA GPU if you have one. On a laptop CPU, the `small` model needs roughly 1/3 to 1/2 of the video's length. Use `base` for speed or `medium` for accuracy.

## Settings (`config/creators.yaml`)

Everything under `defaults` applies to every creator. Anything set under a creator overrides it.

| Section | Keys |
|---|---|
| `clips` | `count`, `min_seconds`, `max_seconds` |
| `captions` | `font`, `font_size`, `bold`, `uppercase`, `words_per_line`, `primary_color`, `highlight_color`, `outline_color`, `outline`, `shadow`, `position` (0 = top, 1 = bottom) |
| `whisper` | `model`, `device`, `compute_type`, `language` |
| `claude` | `model`, `effort` |
| `video` | `width`, `height`, `crf`, `preset` |
| `description` | one sentence about the channel; Claude uses it to pick better moments |

**Fonts:** `font` must be installed on your machine. To use a custom font (e.g. Montserrat ExtraBold), put the `.ttf` file in a `fonts/` folder in this project and set `font:` to its family name.

## How it works

- **Transcription** is cached in `work/<creator>/<video>/transcript.json`, and Claude's picks in `moments.json`. If a later step fails, re-running does not repeat the slow or paid steps. Delete `work/` to start fresh.
- **Moment picking** sends the timestamped transcript to `claude-opus-5-5` with a JSON schema, so the reply is always valid JSON. Clip edges are snapped to word boundaries. Clips outside the length range, or overlapping another clip, are dropped. Server-side refusal fallback is enabled, so if the model declines a request, the API retries it on a fallback model automatically.
- **Cropping** samples 12 frames per clip with OpenCV face detection and centers the 9:16 crop on the largest face. If no face is found, it crops the center.
- **Captions** are an `.ass` subtitle file showing `words_per_line` words at a time, with the spoken word highlighted and slightly enlarged.

## Tests

```bash
pip install pytest
python -m pytest -q
```

The end-to-end test renders real clips with FFmpeg. It uses a fake transcript and a local stand-in for the Claude API, so it needs no API key and no Whisper download.
