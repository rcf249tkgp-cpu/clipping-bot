"""Ask Claude to pick the best short-form moments from a transcript."""

from __future__ import annotations

import json

import anthropic

CLIP_SCHEMA = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number", "description": "Clip start in seconds"},
                    "end": {"type": "number", "description": "Clip end in seconds"},
                    "title": {"type": "string"},
                    "hook": {"type": "string", "description": "What is said in the first 3 seconds"},
                    "caption": {"type": "string", "description": "Post caption for TikTok/Reels/Shorts"},
                    "hashtags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["start", "end", "title", "hook", "caption", "hashtags"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["clips"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are an expert short-form video editor who finds the moments in long videos \
that will perform best as TikTok, Instagram Reels and YouTube Shorts clips.

A great clip:
- Opens with a strong hook in its first 3 seconds: a bold claim, a surprising fact, a question, \
conflict, or an emotional line. Start the clip exactly on that line, not on filler before it.
- Is self-contained: a viewer with no context understands it and it ends on a complete thought \
(a payoff, punchline or conclusion), never mid-sentence.
- Has high energy or high value; skip intros, sponsor reads, housekeeping and rambling.

Timestamps in the transcript are in seconds. Use them to set exact start and end times that \
fall on sentence boundaries. Clips must not overlap. Order clips from strongest to weakest.

For each clip write a short punchy title, the hook (the words spoken in the first 3 seconds), \
a post caption (1-2 sentences, written to drive comments or shares) and 3-6 relevant hashtags \
including the leading #."""


def format_transcript(transcript: dict) -> str:
    return "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in transcript["segments"])


def pick_moments(transcript: dict, settings: dict, description: str | None = None) -> list[dict]:
    """Return the raw clip list from Claude (validated later by `clean_moments`)."""
    clips_cfg = settings["clips"]
    claude_cfg = settings["claude"]

    request = (
        f"Pick the {clips_cfg['count']} best moments from this video. Each clip must be between "
        f"{clips_cfg['min_seconds']} and {clips_cfg['max_seconds']} seconds long. "
        f"The video is {transcript['duration']:.0f} seconds long.\n"
    )
    if description:
        request += f"About this creator: {description}\n"
    request += f"\n<transcript>\n{format_transcript(transcript)}\n</transcript>"

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment / .env
    response = client.beta.messages.create(
        model=claude_cfg["model"],
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": request}],
        output_config={
            "effort": claude_cfg.get("effort", "medium"),
            "format": {"type": "json_schema", "schema": CLIP_SCHEMA},
        },
        # If a safety classifier declines the request, retry it on a fallback model automatically.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"Claude declined to pick clips: {response.stop_details}")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("Claude's response was cut off (max_tokens); try fewer clips.")

    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)["clips"]


def _snap(t: float, words: list[dict], edge: str) -> float:
    """Move a timestamp onto the nearest word boundary so clips never cut a word in half."""
    if not words:
        return t
    key = "start" if edge == "start" else "end"
    return min((w[key] for w in words), key=lambda v: abs(v - t))


def clean_moments(raw: list[dict], transcript: dict, words: list[dict], clips_cfg: dict) -> list[dict]:
    """Clamp, snap to words, enforce length limits, drop overlaps, keep the top N."""
    duration = transcript["duration"]
    lo, hi = clips_cfg["min_seconds"], clips_cfg["max_seconds"]
    tolerance = 2.0  # accept clips a little outside the range rather than dropping good moments
    kept: list[dict] = []
    for clip in raw:
        start = max(0.0, _snap(float(clip["start"]), words, "start"))
        end = min(duration, _snap(float(clip["end"]), words, "end"))
        # Pad slightly so the first and last words aren't clipped.
        start = max(0.0, start - 0.15)
        end = min(duration, end + 0.25)
        if end - start > hi:
            # Too long: end on the last word that still fits.
            fits = [w["end"] for w in words if start < w["end"] <= start + hi]
            end = max(fits) if fits else start + hi
        length = end - start
        if length < lo - tolerance or length > hi + tolerance:
            print(f"  skipping '{clip['title']}': {length:.1f}s is outside {lo}-{hi}s")
            continue
        if any(start < k["end"] and end > k["start"] for k in kept):
            print(f"  skipping '{clip['title']}': overlaps another clip")
            continue
        tags = [t if t.startswith("#") else f"#{t}" for t in (h.strip().replace(" ", "") for h in clip["hashtags"]) if t]
        kept.append({**clip, "start": round(start, 2), "end": round(end, 2), "hashtags": tags})
        if len(kept) == clips_cfg["count"]:
            break
    return kept
