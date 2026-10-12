"""Glue: input/<creator>/*.mp4 -> output/ready/<creator>/<date>/ clips."""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

from . import captions, moments, render, tracking, transcribe
from .config import creator_settings

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}


def slugify(text: str, max_len: int = 50) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "clip"


class Pipeline:
    def __init__(self, root: Path, config: dict, transcribe_fn=None, pick_fn=None):
        self.root = root
        self.config = config
        self.input_dir = root / "input"
        self.ready_dir = root / "output" / "ready"
        self.work_dir = root / "work"
        self.fonts_dir = root / "fonts"
        self.csv_path = root / "tracking" / "clips.csv"
        self.processed = tracking.ProcessedIndex(root / "tracking" / "processed.json")
        # Injectable for tests / offline runs.
        self.transcribe_fn = transcribe_fn or transcribe.load_or_transcribe
        self.pick_fn = pick_fn or moments.pick_moments

    def creators(self) -> list[str]:
        if not self.input_dir.is_dir():
            return []
        return sorted(p.name for p in self.input_dir.iterdir() if p.is_dir() and not p.name.startswith("."))

    def videos(self, creator: str) -> list[Path]:
        folder = self.input_dir / creator
        return sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS and not p.name.startswith("."))

    def run(self, only_creator: str | None = None, force: bool = False) -> int:
        creators = [only_creator] if only_creator else self.creators()
        if not creators:
            print(f"No creator folders found in {self.input_dir}/")
            return 0
        total = 0
        for creator in creators:
            if not (self.input_dir / creator).is_dir():
                print(f"No folder {self.input_dir / creator}")
                continue
            if creator not in self.config["creators"]:
                print(f"[{creator}] not in config/creators.yaml, using defaults")
            settings = creator_settings(self.config, creator)
            for video in self.videos(creator):
                if not force and self.processed.done(creator, video):
                    print(f"[{creator}] skip {video.name} (already processed)")
                    continue
                try:
                    total += self.process_video(creator, video, settings)
                except Exception as e:  # keep going with the next video
                    print(f"[{creator}] FAILED {video.name}: {e}")
        print(f"Done: {total} clip(s) created.")
        return total

    def process_video(self, creator: str, video: Path, settings: dict) -> int:
        print(f"[{creator}] {video.name}")
        work = self.work_dir / creator / video.stem
        work.mkdir(parents=True, exist_ok=True)

        print("  1/4 transcribing")
        transcript = self.transcribe_fn(video, work / "transcript.json", settings["whisper"])
        words = transcribe.all_words(transcript)
        if not words:
            raise RuntimeError("no speech found")

        print("  2/4 asking Claude for the best moments")
        moments_file = work / "moments.json"
        if moments_file.exists():  # reuse picks if a previous run failed during rendering
            raw = json.loads(moments_file.read_text(encoding="utf-8"))
        else:
            raw = self.pick_fn(transcript, settings, settings.get("description"))
            moments_file.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
        clips = moments.clean_moments(raw, transcript, words, settings["clips"])
        if not clips:
            raise RuntimeError("Claude returned no usable clips")

        print(f"  3/4 rendering {len(clips)} clip(s)")
        info = render.probe(video)
        vcfg = settings["video"]
        date = dt.date.today().isoformat()
        out_dir = self.ready_dir / creator / date
        made = []
        for i, clip in enumerate(clips, 1):
            name = f"{video.stem[:40]}_{i:02d}_{slugify(clip['title'])}"
            out_mp4 = out_dir / f"{name}.mp4"

            cx = render.speaker_center_x(video, clip["start"], clip["end"])
            crop = render.crop_box(info["width"], info["height"], vcfg["width"], vcfg["height"], cx)
            ass_file = work / f"{name}.ass"
            ass_file.write_text(
                captions.build_ass(
                    captions.clip_words(words, clip["start"], clip["end"]),
                    settings["captions"], vcfg["width"], vcfg["height"],
                ),
                encoding="utf-8",
            )
            render.render_clip(video, out_mp4, clip["start"], clip["end"], ass_file, crop, vcfg, self.fonts_dir)

            hashtags = " ".join(clip["hashtags"])
            out_mp4.with_suffix(".txt").write_text(f"{clip['caption']}\n\n{hashtags}\n", encoding="utf-8")
            where = "speaker" if cx is not None else "center"
            print(f"    [{i}] {clip['start']:.1f}-{clip['end']:.1f}s  crop={where}  {out_mp4.name}")
            made.append((clip, out_mp4))

        print("  4/4 updating tracking/clips.csv")
        now = dt.datetime.now().isoformat(timespec="seconds")
        for clip, out_mp4 in made:
            tracking.append_clip(self.csv_path, {
                "created_at": now,
                "creator": creator,
                "source_video": video.name,
                "clip_file": out_mp4.relative_to(self.root).as_posix(),
                "start": clip["start"],
                "end": clip["end"],
                "duration": round(clip["end"] - clip["start"], 2),
                "title": clip["title"],
                "hook": clip["hook"],
                "caption": clip["caption"],
                "hashtags": " ".join(clip["hashtags"]),
                "status": "ready",
            })
        self.processed.mark(creator, video, {"processed_at": now, "clips": len(made)})
        return len(made)
