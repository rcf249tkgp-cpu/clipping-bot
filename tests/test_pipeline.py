"""Tests for clipper. The end-to-end test runs real FFmpeg rendering, with Whisper and the
Claude API replaced by stand-ins (a fake transcript and a local HTTP server)."""

from __future__ import annotations

import csv
import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from clipper import captions, moments, render
from clipper.config import creator_settings, load_config
from clipper.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]

SENTENCES = [
    "Nobody tells you this but most startups die from indigestion not starvation",
    "I learned that the hard way when we raised too much money too early",
    "We hired fifty people in six months and lost our focus completely",
    "Here is the one question I ask before every single hire now",
    "Would I be excited to work for this person someday",
    "If the answer is no we do not make the offer no matter what",
]


def fake_transcript(duration: float, words_per_sec: float = 2.5) -> dict:
    """Evenly timed speech, one sentence per segment, filling `duration` seconds."""
    segments, t, i = [], 0.5, 0
    while t < duration - 3:
        text = SENTENCES[i % len(SENTENCES)]
        words = []
        for w in text.split():
            words.append({"start": round(t, 3), "end": round(t + 0.32, 3), "word": w})
            t += 1 / words_per_sec
        segments.append({"start": words[0]["start"], "end": words[-1]["end"], "text": text, "words": words})
        t += 0.6
        i += 1
    return {"language": "en", "duration": duration, "segments": segments}


def make_video(path: Path, seconds: int, size: str = "1280x720") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=30:duration={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=220:duration={seconds}",
            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(path),
        ],
        check=True,
    )


# ---------- unit tests ----------

def test_crop_box_landscape_follows_speaker():
    assert render.crop_box(1920, 1080, 1080, 1920, None) == (608, 1080, 656, 0)
    w, h, x, y = render.crop_box(1920, 1080, 1080, 1920, 0.25)
    assert (w, h, y) == (608, 1080, 0) and x == 176
    assert render.crop_box(1920, 1080, 1080, 1920, 0.0)[2] == 0          # clamped left
    assert render.crop_box(1920, 1080, 1080, 1920, 1.0)[2] == 1920 - 608  # clamped right


def test_crop_box_vertical_source_is_untouched():
    assert render.crop_box(1080, 1920, 1080, 1920, None) == (1080, 1920, 0, 0)


def test_ass_colors_and_highlight():
    assert captions._ass_color("#FFE600") == "&H0000E6FF"
    style = creator_settings(load_config(ROOT / "config" / "creators.yaml"), "examplecreator")["captions"]
    words = [{"word": "hello", "start": 0.0, "end": 0.3}, {"word": "world", "start": 0.4, "end": 0.7}]
    ass = captions.build_ass(words, style, 1080, 1920)
    dialogues = [line for line in ass.splitlines() if line.startswith("Dialogue")]
    assert len(dialogues) == 2  # one event per spoken word
    assert "HELLO" in dialogues[0] and "\\c&H00FFE500" in dialogues[0]  # #00E5FF highlight


def test_clean_moments_snaps_filters_and_dedupes():
    transcript = fake_transcript(200)
    words = [w for s in transcript["segments"] for w in s["words"]]
    clips = {"count": 5, "min_seconds": 20, "max_seconds": 60}
    raw = [
        {"start": 10.1, "end": 40.0, "title": "ok", "hook": "", "caption": "", "hashtags": ["startups", "#money"]},
        {"start": 20.0, "end": 45.0, "title": "overlap", "hook": "", "caption": "", "hashtags": []},
        {"start": 100.0, "end": 105.0, "title": "too short", "hook": "", "caption": "", "hashtags": []},
        {"start": 110.0, "end": 190.0, "title": "too long", "hook": "", "caption": "", "hashtags": []},
    ]
    out = moments.clean_moments(raw, transcript, words, clips)
    assert [c["title"] for c in out] == ["ok", "too long"]
    assert out[0]["hashtags"] == ["#startups", "#money"]
    assert any(abs(out[0]["start"] + 0.15 - w["start"]) < 1e-6 for w in words)  # on a word boundary
    assert out[1]["end"] - out[1]["start"] <= 60


# ---------- Claude request shape (local fake API server) ----------

class _FakeClaude(BaseHTTPRequestHandler):
    requests: list = []
    reply: list = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        _FakeClaude.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
        payload = {
            "id": "msg_test", "type": "message", "role": "assistant", "model": body["model"],
            "content": [{"type": "text", "text": json.dumps({"clips": _FakeClaude.reply})}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def fake_claude(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), _FakeClaude)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _FakeClaude.requests = []
    yield _FakeClaude
    server.shutdown()


# ---------- end to end ----------

def test_end_to_end(tmp_path, fake_claude):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "creators.yaml").write_text(
        (ROOT / "config" / "creators.yaml").read_text().replace("preset: medium", "preset: ultrafast")
    )
    video = tmp_path / "input" / "examplecreator" / "episode1.mp4"
    make_video(video, 95)
    transcript = fake_transcript(95)

    fake_claude.reply = [
        {"start": 3.0, "end": 30.0, "title": "Startups die from indigestion",
         "hook": "Nobody tells you this", "caption": "Raised too much too soon?", "hashtags": ["#startups", "founder"]},
        {"start": 40.0, "end": 70.0, "title": "The one hiring question",
         "hook": "Here is the one question", "caption": "Steal this hiring rule.", "hashtags": ["#hiring"]},
    ]

    config = load_config(tmp_path / "config" / "creators.yaml")
    pipe = Pipeline(tmp_path, config, transcribe_fn=lambda v, cache, s: transcript)
    assert pipe.run() == 2

    # Claude was called once with structured output + fallbacks.
    req = fake_claude.requests[0]
    assert req["path"].startswith("/v1/messages")
    assert req["body"]["model"] == "claude-opus-5-5"
    assert req["body"]["output_config"]["format"]["type"] == "json_schema"
    assert req["body"]["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in req["headers"].get("anthropic-beta", "")
    assert "[0.5-" in req["body"]["messages"][0]["content"]  # timestamped transcript

    day_dirs = list((tmp_path / "output" / "ready" / "examplecreator").iterdir())
    assert len(day_dirs) == 1
    mp4s = sorted(day_dirs[0].glob("*.mp4"))
    assert len(mp4s) == 2
    for mp4 in mp4s:
        info = render.probe(mp4)
        assert (info["width"], info["height"]) == (1080, 1920)
        assert 20 <= info["duration"] <= 61
        txt = mp4.with_suffix(".txt").read_text()
        assert "#" in txt
    assert "#founder" in mp4s[0].with_suffix(".txt").read_text()

    rows = list(csv.DictReader(open(tmp_path / "tracking" / "clips.csv")))
    assert len(rows) == 2 and rows[0]["creator"] == "examplecreator" and rows[0]["status"] == "ready"

    # Second run skips the already-processed video.
    assert pipe.run() == 0
    assert len(fake_claude.requests) == 1
    assert len(list(csv.DictReader(open(tmp_path / "tracking" / "clips.csv")))) == 2
