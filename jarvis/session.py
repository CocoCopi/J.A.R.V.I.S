"""Conversation persistence.

Sessions are JSON files under ``~/.local/share/jarvis/sessions`` so a phone can
resume a chat after Termux is killed by the OS.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from .providers.base import Message, ToolCall


def sessions_dir() -> Path:
    override = os.environ.get("JARVIS_DATA_DIR")
    base = Path(override).expanduser() if override else Path.home() / ".local" / "share" / "jarvis"
    path = base / "sessions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def new_session_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _message_to_json(message: Message) -> Dict[str, object]:
    data = asdict(message)
    data.pop("images", None)  # binary payloads are never persisted
    return data


def _message_from_json(data: Dict[str, object]) -> Message:
    calls = [
        ToolCall(
            id=str(call.get("id", "")),
            name=str(call.get("name", "")),
            arguments=dict(call.get("arguments") or {}),
            raw_arguments=str(call.get("raw_arguments") or ""),
        )
        for call in (data.get("tool_calls") or [])
    ]
    return Message(
        role=str(data.get("role", "user")),
        content=str(data.get("content") or ""),
        tool_calls=calls,
        tool_call_id=data.get("tool_call_id"),
        name=data.get("name"),
    )


def save(session_id: str, messages: List[Message], meta: Optional[Dict[str, object]] = None) -> Path:
    """Persist *messages* to ``<session_id>.json``."""
    path = sessions_dir() / ("%s.json" % session_id)
    payload = {
        "id": session_id,
        "updated": time.time(),
        "meta": meta or {},
        "messages": [_message_to_json(m) for m in messages],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load(session_id: str) -> List[Message]:
    """Load a saved session, or an empty list if it does not exist."""
    path = sessions_dir() / ("%s.json" % session_id)
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [_message_from_json(m) for m in payload.get("messages", [])]


def latest() -> Optional[str]:
    """Return the most recently updated session id."""
    directory = sessions_dir()
    files = sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0].stem if files else None


def list_sessions(limit: int = 20) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sorted(sessions_dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        messages = payload.get("messages", [])
        preview = ""
        for message in messages:
            if message.get("role") == "user" and message.get("content"):
                preview = str(message["content"])[:60]
                break
        rows.append(
            {
                "id": path.stem,
                "updated": time.strftime(
                    "%Y-%m-%d %H:%M", time.localtime(float(payload.get("updated", 0)))
                ),
                "turns": len(messages),
                "preview": preview,
            }
        )
        if len(rows) >= limit:
            break
    return rows
