"""Load config/creators.yaml and resolve per-creator settings."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data.setdefault("defaults", {})
    data["creators"] = data.get("creators") or {}
    return data


def creator_settings(config: dict, creator: str) -> dict:
    """Defaults merged with the creator's overrides (unknown creators get defaults)."""
    return _deep_merge(config["defaults"], config["creators"].get(creator) or {})
