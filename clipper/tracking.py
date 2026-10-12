"""tracking/clips.csv (one row per clip) and tracking/processed.json (videos already done)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

CSV_FIELDS = [
    "created_at", "creator", "source_video", "clip_file", "start", "end", "duration",
    "title", "hook", "caption", "hashtags", "status",
]


def append_clip(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


class ProcessedIndex:
    """Remembers which source videos were already clipped (keyed by creator/filename + size)."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    @staticmethod
    def key(creator: str, video: Path) -> str:
        return f"{creator}/{video.name}:{video.stat().st_size}"

    def done(self, creator: str, video: Path) -> bool:
        return self.key(creator, video) in self.data

    def mark(self, creator: str, video: Path, info: dict) -> None:
        self.data[self.key(creator, video)] = info
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)
