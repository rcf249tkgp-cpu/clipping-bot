"""Command line entry point: python -m clipper [--creator NAME] [--force]"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

from .config import load_config
from .pipeline import Pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clipper", description="Turn long videos into vertical short clips.")
    parser.add_argument("--creator", help="only process input/<creator>/")
    parser.add_argument("--force", action="store_true", help="re-process videos that were already done")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="project folder (default: current dir)")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    load_dotenv(root / ".env")

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            print(f"Error: '{tool}' not found on PATH. Install FFmpeg first (see README).")
            return 1
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")
        return 1

    config = load_config(root / "config" / "creators.yaml")
    Pipeline(root, config).run(only_creator=args.creator, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
