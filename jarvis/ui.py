"""Tiny terminal UI layer: colours, banners, streaming output.

Honours ``NO_COLOR`` and ``JARVIS_PLAIN`` and falls back to no colour when the
output is not a TTY, which matters when Jarvis is driven from a script.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, TextIO

RESET = "\033[0m"
_CODES = {
    "bold": "1",
    "dim": "2",
    "italic": "3",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "grey": "90",
    "bright_cyan": "96",
    "bright_green": "92",
}


def _enabled(stream: Optional[TextIO] = None) -> bool:
    if os.environ.get("NO_COLOR") or os.environ.get("JARVIS_PLAIN"):
        return False
    stream = stream or sys.stdout
    try:
        return bool(stream.isatty())
    except Exception:  # pragma: no cover
        return False


def style(text: str, *names: str, stream: Optional[TextIO] = None) -> str:
    """Wrap *text* in ANSI codes when colour is appropriate."""
    if not names or not _enabled(stream):
        return text
    codes = ";".join(_CODES[n] for n in names if n in _CODES)
    return "\033[%sm%s%s" % (codes, text, RESET)


def info(message: str, stream: Optional[TextIO] = None) -> None:
    print(style(message, "grey", stream=stream), file=stream or sys.stdout)


def ok(message: str, stream: Optional[TextIO] = None) -> None:
    print(style(message, "green", stream=stream), file=stream or sys.stdout)


def warn(message: str, stream: Optional[TextIO] = None) -> None:
    print(style(message, "yellow", stream=stream), file=stream or sys.stderr)


def error(message: str, stream: Optional[TextIO] = None) -> None:
    print(style(message, "red", stream=stream), file=stream or sys.stderr)


def heading(message: str, stream: Optional[TextIO] = None) -> None:
    print(style(message, "bold", "cyan", stream=stream), file=stream or sys.stdout)


def prompt(label: str) -> str:
    try:
        return input(style(label, "bold", "bright_green"))
    except EOFError:
        return ""


def banner(provider: str, model: str, voice: str = "none", tools: int = 0) -> None:
    """Print the interactive-session header."""
    lines = [
        "",
        style("  J A R V I S", "bold", "bright_cyan"),
        style("  %s  ·  %s  ·  %d tools  ·  voice: %s" % (provider, model, tools, voice), "grey"),
        style("  /help for commands, /exit to quit", "grey"),
        "",
    ]
    print("\n".join(lines))


def stream_printer(prefix: str = "", stream: Optional[TextIO] = None):
    """Return a callable that writes streamed tokens without extra newlines."""
    out = stream or sys.stdout
    state = {"started": False}

    def write(chunk: str) -> None:
        if not state["started"]:
            if prefix:
                out.write(prefix)
            state["started"] = True
        out.write(chunk)
        out.flush()

    def done() -> None:
        if state["started"]:
            out.write("\n")
            out.flush()
            state["started"] = False

    write.done = done  # type: ignore[attr-defined]
    return write
