"""Cut, crop to vertical (centered on the speaker) and burn in captions with FFmpeg."""

from __future__ import annotations

import json
import statistics
import subprocess
from pathlib import Path


def probe(video: Path) -> dict:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height:stream_side_data=rotation:format=duration",
            "-of", "json", str(video),
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    data = json.loads(out)
    stream = data["streams"][0]
    w, h = int(stream["width"]), int(stream["height"])
    rotation = next((abs(int(sd.get("rotation", 0))) for sd in stream.get("side_data_list", [])), 0)
    if rotation in (90, 270):  # phone videos stored sideways
        w, h = h, w
    return {"width": w, "height": h, "duration": float(data["format"]["duration"])}


def speaker_center_x(video: Path, start: float, end: float, samples: int = 12) -> float | None:
    """Median horizontal face position (0..1) over the clip, or None if no face is found."""
    try:
        import cv2
    except ImportError:
        return None
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(str(video))
    centers = []
    try:
        for i in range(samples):
            t = start + (end - start) * (i + 0.5) / samples
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            min_side = max(40, frame.shape[0] // 12)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(min_side, min_side))
            if len(faces):
                x, _, fw, _ = max(faces, key=lambda f: f[2] * f[3])  # largest face = main speaker
                centers.append((x + fw / 2) / frame.shape[1])
    finally:
        cap.release()
    return statistics.median(centers) if centers else None


def crop_box(src_w: int, src_h: int, out_w: int, out_h: int, center_x: float | None) -> tuple[int, int, int, int]:
    """Largest out_w:out_h box inside the source, horizontally centered on center_x (0..1)."""
    target = out_w / out_h
    if src_w / src_h > target:  # wider than 9:16 -> crop the sides
        ch = src_h
        cw = int(round(src_h * target))
        cx = (center_x if center_x is not None else 0.5) * src_w
        x = int(round(min(max(cx - cw / 2, 0), src_w - cw)))
        y = 0
    else:  # taller/narrower -> crop top and bottom, keep the upper part where faces usually are
        cw = src_w
        ch = int(round(src_w / target))
        x = 0
        y = int(round((src_h - ch) * 0.35))
    cw -= cw % 2
    ch -= ch % 2
    return cw, ch, x, y


def _filter_path(path: Path) -> str:
    """Escape a path for use inside an FFmpeg filter argument (handles Windows 'C:')."""
    return "'" + path.resolve().as_posix().replace(":", "\\:").replace("'", "\\'") + "'"


def render_clip(
    video: Path,
    out_file: Path,
    start: float,
    end: float,
    ass_file: Path,
    crop: tuple[int, int, int, int],
    video_cfg: dict,
    fonts_dir: Path | None = None,
) -> None:
    cw, ch, x, y = crop
    ass_arg = f"filename={ass_file.name}"
    if fonts_dir and fonts_dir.is_dir():
        ass_arg += f":fontsdir={_filter_path(fonts_dir)}"
    vf = (
        f"crop={cw}:{ch}:{x}:{y},"
        f"scale={video_cfg['width']}:{video_cfg['height']}:flags=lanczos,setsar=1,"
        f"ass={ass_arg}"
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-i", str(video.resolve()), "-t", f"{end - start:.3f}",
        "-vf", vf,
        "-c:v", "libx264", "-preset", str(video_cfg["preset"]), "-crf", str(video_cfg["crf"]),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
        "-movflags", "+faststart",
        str(out_file.resolve()),
    ]
    # Run from the subtitle's folder so its path needs no escaping inside the filter.
    result = subprocess.run(cmd, cwd=ass_file.parent, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {out_file.name}:\n{result.stderr.strip()}")
