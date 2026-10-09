"""Locate the numbered scripts after the repository was split into folders.

Scripts load one another by file name (they start with digits, so they cannot
be imported normally). ``script("20_event_panel.py")`` returns the absolute
path wherever that file now lives, so a move never breaks a loader.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .config import ROOT

FOLDERS = ("pipeline", "publish", "research", "tools")


@lru_cache(maxsize=None)
def _index() -> dict:
    out = {}
    for folder in FOLDERS:
        for p in (ROOT / folder).rglob("*.py"):
            if p.name in out:
                raise RuntimeError(f"duplicate script name {p.name}: {out[p.name]} and {p}")
            out[p.name] = p
    return out


def script(name: str) -> Path:
    try:
        return _index()[name]
    except KeyError:
        raise FileNotFoundError(f"no script named {name} under {', '.join(FOLDERS)}") from None
