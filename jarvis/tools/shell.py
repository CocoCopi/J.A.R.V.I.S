"""Shell + code execution tools.

These are what make Jarvis actually *do* things on a Termux device: install
packages, drive Git, launch apps, poke at ``/sdcard``, whatever the user asks.
They are the only tools marked as needing confirmation by default.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import tempfile
from typing import List, Optional

from .base import ToolResult, tool

MAX_OUTPUT = 20_000

#: Patterns that should never run without an explicit yes.
DANGEROUS_PATTERNS = (
    r"\brm\s+(-[a-zA-Z]*\s+)*(-rf|-fr|-r)\s+(/|\$HOME|~|\*)",
    r"\bmkfs(\.\w+)?\b",
    r"\bdd\b[^\n]*\bof=/dev/(sd|block|mmc|disk)",
    r":\(\)\s*\{.*\};\s*:",
    r"\bchmod\s+(-R\s+)?0?777\s+/\s*$",
    r"\b(shutdown|reboot|halt|poweroff)\b",
    r">\s*/dev/(sd|block|mmc|disk)",
    r"\bwipe\b|\bfastboot\s+erase\b",
    r"\bpm\s+uninstall\b|\buninstall\b",
)
_DANGER_RE = [re.compile(p) for p in DANGEROUS_PATTERNS]

#: Commands that are worth a warning but are not blocked.
_CAUTION_RE = re.compile(r"\b(sudo|su|pkg\s+uninstall|pip\s+uninstall|rm\s+-r)\b")


def looks_dangerous(command: str) -> Optional[str]:
    """Return a reason string when *command* matches a destructive pattern."""
    for pattern in _DANGER_RE:
        match = pattern.search(command)
        if match:
            return "destructive pattern %r" % match.group(0)
    match = _CAUTION_RE.search(command)
    if match:
        return "potentially disruptive %r" % match.group(0)
    return None


def _truncate(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return "%s\n... [%d chars omitted] ...\n%s" % (text[:half], len(text) - limit, text[-half:])


def _format(completed: "subprocess.CompletedProcess[str]") -> str:
    chunks: List[str] = []
    if completed.stdout:
        chunks.append(completed.stdout.rstrip())
    if completed.stderr:
        chunks.append("[stderr]\n" + completed.stderr.rstrip())
    body = "\n".join(chunks) or "(no output)"
    if completed.returncode != 0:
        body = "%s\n[exit code %d]" % (body, completed.returncode)
    return _truncate(body)


@tool(confirm=True, category="shell")
def run_shell(command: str, cwd: Optional[str] = None, timeout: int = 120) -> ToolResult:
    """Run a shell command on this device and return its output.

    Uses the platform shell (``/bin/sh`` on Termux/proot). Prefer this for
    installing packages, git, file wrangling, launching Android intents, etc.

    Args:
        command: The shell command line to execute.
        cwd: Directory to run in (defaults to the current directory).
        timeout: Seconds to wait before killing the command.
    """
    if not command or not command.strip():
        return ToolResult(output="no command provided", ok=False)
    workdir = os.path.expanduser(cwd) if cwd else None
    if workdir and not os.path.isdir(workdir):
        return ToolResult(output="no such directory: %s" % workdir, ok=False)
    env = dict(os.environ)
    env.setdefault("PAGER", "cat")
    env.setdefault("GIT_PAGER", "cat")
    env.setdefault("TERM", "dumb")
    env["DEBIAN_FRONTEND"] = "noninteractive"
    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=max(1, int(timeout)),
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            output="command timed out after %ss: %s" % (timeout, command), ok=False
        )
    except OSError as exc:
        return ToolResult(output="could not run command: %s" % exc, ok=False)
    return ToolResult(output=_format(completed), ok=completed.returncode == 0)


@tool(confirm=True, category="shell")
def python_exec(code: str, timeout: int = 60) -> ToolResult:
    """Execute Python code in a subprocess and return stdout/stderr.

    Useful for maths, data munging, and anything the shell tools do not cover.

    Args:
        code: Python source to run with the same interpreter as Jarvis.
        timeout: Seconds before the interpreter is killed.
    """
    if not code.strip():
        return ToolResult(output="no code provided", ok=False)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
        handle.write(code)
        path = handle.name
    try:
        completed = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=max(1, int(timeout)),
        )
        return ToolResult(output=_format(completed), ok=completed.returncode == 0)
    except subprocess.TimeoutExpired:
        return ToolResult(output="python timed out after %ss" % timeout, ok=False)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


@tool(category="shell")
def which(program: str) -> str:
    """Check whether a command exists on PATH (e.g. ``termux-tts-speak``).

    Args:
        program: Executable name to look up.
    """
    from shutil import which as _which

    found = _which(program)
    return found or "%s: not found" % program


@tool(category="shell")
def command_help(program: str, max_lines: int = 40) -> str:
    """Show a command's ``--help`` output so you can learn its interface.

    Args:
        program: Command to inspect, e.g. ``termux-notification``.
        max_lines: Truncate the help to this many lines.
    """
    try:
        completed = subprocess.run(
            [program, "--help"],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return "could not run %s --help: %s" % (program, exc)
    text = (completed.stdout or "") + (completed.stderr or "")
    lines = text.strip().splitlines()
    return "\n".join(lines[: max(1, int(max_lines))]) or "(no help output)"


def quote(value: str) -> str:
    """Shell-quote a value (exported for other tool modules)."""
    return shlex.quote(value)
