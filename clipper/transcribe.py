"""Local transcription with faster-whisper (word-level timestamps)."""

from __future__ import annotations

import json
from pathlib import Path

_MODELS: dict[tuple, object] = {}


def _model(settings: dict):
    from faster_whisper import WhisperModel

    key = (settings["model"], settings["device"], settings["compute_type"])
    if key not in _MODELS:
        print(f"  loading whisper model '{settings['model']}' (first run downloads it)...")
        _MODELS[key] = WhisperModel(
            settings["model"], device=settings["device"], compute_type=settings["compute_type"]
        )
    return _MODELS[key]


def transcribe(video: Path, settings: dict) -> dict:
    """Return {"language", "duration", "segments": [{start, end, text, words: [{start, end, word}]}]}."""
    model = _model(settings)
    segments, info = model.transcribe(
        str(video),
        language=settings.get("language"),
        word_timestamps=True,
        vad_filter=True,
    )
    out = []
    for seg in segments:
        words = [
            {"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word.strip()}
            for w in (seg.words or [])
            if w.word.strip()
        ]
        out.append(
            {"start": round(seg.start, 3), "end": round(seg.end, 3), "text": seg.text.strip(), "words": words}
        )
        print(f"\r  transcribed {seg.end:7.1f}s / {info.duration:.1f}s", end="", flush=True)
    print()
    return {"language": info.language, "duration": info.duration, "segments": out}


def load_or_transcribe(video: Path, cache_file: Path, settings: dict) -> dict:
    """Transcripts are cached so a failed later step never re-runs Whisper."""
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    transcript = transcribe(video, settings)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(transcript, ensure_ascii=False), encoding="utf-8")
    return transcript


def all_words(transcript: dict) -> list[dict]:
    return [w for seg in transcript["segments"] for w in seg["words"]]
