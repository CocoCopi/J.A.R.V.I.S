"""Long-term memory tools backed by a small JSON file in the Jarvis data dir."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List

from .base import ToolError, tool

MAX_NOTES = 500


def _store_path() -> Path:
    override = os.environ.get("JARVIS_DATA_DIR")
    base = Path(override).expanduser() if override else Path.home() / ".local" / "share" / "jarvis"
    base.mkdir(parents=True, exist_ok=True)
    return base / "memory.json"


def _load() -> List[Dict[str, object]]:
    path = _store_path()
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def _save(notes: List[Dict[str, object]]) -> None:
    path = _store_path()
    path.write_text(json.dumps(notes[-MAX_NOTES:], indent=2), encoding="utf-8")


@tool(category="memory")
def remember(note: str, tag: str = "general") -> str:
    """Save a fact, preference or task for later recall.

    Use this whenever the user says "remember", or states a durable preference.

    Args:
        note: The thing to remember, written as a self-contained sentence.
        tag: Bucket such as ``preference``, ``todo``, ``person``.
    """
    notes = _load()
    notes.append({"note": note.strip(), "tag": tag.strip() or "general", "at": time.time()})
    _save(notes)
    return "remembered (%s): %s" % (tag, note.strip())


@tool(category="memory")
def recall(query: str = "", tag: str = "") -> str:
    """Search saved memories.

    Args:
        query: Case-insensitive substring to match; empty lists the newest notes.
        tag: Only return notes with this tag.
    """
    notes = _load()
    if tag:
        notes = [n for n in notes if str(n.get("tag", "")).lower() == tag.lower()]
    if query:
        needle = query.lower()
        notes = [n for n in notes if needle in str(n.get("note", "")).lower()]
    if not notes:
        return "no matching memories"
    rows: List[str] = []
    for entry in notes[-25:]:
        stamp = time.strftime("%Y-%m-%d", time.localtime(float(entry.get("at", 0))))
        rows.append("- [%s] %s (%s)" % (entry.get("tag", "general"), entry.get("note", ""), stamp))
    return "\n".join(rows)


@tool(category="memory")
def forget(query: str) -> str:
    """Delete memories whose text contains *query*.

    Args:
        query: Substring identifying the notes to delete.
    """
    needle = query.strip().lower()
    if not needle:
        raise ToolError("refusing to forget everything; provide a query")
    notes = _load()
    kept = [n for n in notes if needle not in str(n.get("note", "")).lower()]
    removed = len(notes) - len(kept)
    _save(kept)
    return "forgot %d note(s) matching %r" % (removed, query)
