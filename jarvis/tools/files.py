"""Filesystem tools.

Paths accept ``~`` and environment variables, so ``~/storage/shared`` works out
of the box on Termux (after ``termux-setup-storage``).
"""

from __future__ import annotations

import fnmatch
import os
import shutil
from pathlib import Path
from typing import List

from .base import ToolResult, tool

MAX_READ = 200_000
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".cache", ".gradle"}


def _resolve(path: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path))))


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "K", "M", "G", "T"):
        if value < 1024 or unit == "T":
            return "%.1f%s" % (value, unit) if unit != "B" else "%dB" % int(value)
        value /= 1024
    return "%dB" % size


@tool(category="files")
def read_file(path: str, max_bytes: int = MAX_READ, start_line: int = 1) -> ToolResult:
    """Read a text file and return its contents.

    Args:
        path: File to read (supports ~ and $VARS).
        max_bytes: Safety cap on how much to load.
        start_line: Line number to start from (1-based).
    """
    target = _resolve(path)
    if not target.exists():
        return ToolResult(output="no such file: %s" % target, ok=False)
    if target.is_dir():
        return ToolResult(output="%s is a directory; use list_dir" % target, ok=False)
    try:
        with target.open("r", encoding="utf-8", errors="replace") as handle:
            text = handle.read(int(max_bytes))
    except OSError as exc:
        return ToolResult(output="could not read %s: %s" % (target, exc), ok=False)
    start = max(1, int(start_line))
    if start > 1:
        lines = text.splitlines()
        text = "\n".join(lines[start - 1 :])
    truncated = "\n... [truncated at %d bytes]" % max_bytes if len(text) >= max_bytes else ""
    return ToolResult(output="%s%s" % (text, truncated), ok=True)


@tool(confirm=True, category="files")
def write_file(path: str, content: str, append: bool = False) -> ToolResult:
    """Create or overwrite a text file (optionally appending).

    Args:
        path: Destination path; parent directories are created automatically.
        content: Text to write.
        append: Append instead of overwriting.
    """
    target = _resolve(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a" if append else "w", encoding="utf-8") as handle:
            handle.write(content)
    except OSError as exc:
        return ToolResult(output="could not write %s: %s" % (target, exc), ok=False)
    action = "appended %d chars to" % len(content) if append else "wrote %d chars to" % len(content)
    return ToolResult(output="%s %s" % (action, target), ok=True)


@tool(category="files")
def list_dir(path: str = ".", show_hidden: bool = False) -> str:
    """List the entries of a directory with sizes.

    Args:
        path: Directory to list.
        show_hidden: Include dotfiles.
    """
    target = _resolve(path)
    if not target.is_dir():
        return "not a directory: %s" % target
    rows: List[str] = []
    try:
        entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except OSError as exc:
        return "could not list %s: %s" % (target, exc)
    for entry in entries:
        if not show_hidden and entry.name.startswith("."):
            continue
        if entry.is_dir():
            rows.append("  %s/" % entry.name)
        else:
            try:
                size = _human(entry.stat().st_size)
            except OSError:
                size = "?"
            rows.append("  %-40s %8s" % (entry.name, size))
    header = "%s (%d entries)" % (target, len(rows))
    return "\n".join([header] + rows)


@tool(category="files")
def make_dir(path: str) -> str:
    """Create a directory (and parents).

    Args:
        path: Directory to create.
    """
    target = _resolve(path)
    target.mkdir(parents=True, exist_ok=True)
    return "created %s" % target


@tool(category="files")
def move_path(source: str, destination: str) -> str:
    """Move or rename a file/directory.

    Args:
        source: Existing path.
        destination: New path.
    """
    src, dst = _resolve(source), _resolve(destination)
    if not src.exists():
        return "no such path: %s" % src
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return "moved %s -> %s" % (src, dst)


@tool(category="files")
def copy_path(source: str, destination: str) -> str:
    """Copy a file or directory tree.

    Args:
        source: Existing path.
        destination: Destination path.
    """
    src, dst = _resolve(source), _resolve(destination)
    if not src.exists():
        return "no such path: %s" % src
    if src.is_dir():
        shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
    return "copied %s -> %s" % (src, dst)


@tool(confirm=True, category="files")
def delete_path(path: str, recursive: bool = False) -> str:
    """Delete a file or directory.

    Args:
        path: Path to remove.
        recursive: Required to delete a non-empty directory.
    """
    target = _resolve(path)
    if not target.exists():
        return "no such path: %s" % target
    if target.is_dir():
        if not recursive:
            return "refusing to delete directory %s without recursive=true" % target
        shutil.rmtree(target)
    else:
        target.unlink()
    return "deleted %s" % target


@tool(category="files")
def find_files(pattern: str, path: str = ".", max_results: int = 50) -> str:
    """Find files by shell-style glob (``*.mp3``, ``config*``).

    Args:
        pattern: Glob such as ``*.jpg`` or ``*.py``.
        path: Root directory to search.
        max_results: Stop after this many matches.
    """
    root = _resolve(path)
    matches: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if fnmatch.fnmatch(name, pattern):
                matches.append(os.path.join(dirpath, name))
                if len(matches) >= int(max_results):
                    return "\n".join(matches) + "\n[stopped at %d results]" % max_results
    return "\n".join(matches) or "no matches for %s under %s" % (pattern, root)


@tool(category="files")
def search_text(pattern: str, path: str = ".", file_glob: str = "*.py", max_results: int = 30) -> str:
    """Search inside text files for a substring or regex.

    Args:
        pattern: Regular expression to look for.
        path: Root directory to search.
        file_glob: Only inspect files matching this glob.
        max_results: Stop after this many matching lines.
    """
    import re

    root = _resolve(path)
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return "invalid regex: %s" % exc
    hits: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if not fnmatch.fnmatch(name, file_glob):
                continue
            full = os.path.join(dirpath, name)
            try:
                if os.path.getsize(full) > 2_000_000:
                    continue
                with open(full, "r", encoding="utf-8", errors="ignore") as handle:
                    for lineno, line in enumerate(handle, 1):
                        if regex.search(line):
                            hits.append("%s:%d: %s" % (full, lineno, line.rstrip()[:200]))
                            if len(hits) >= int(max_results):
                                return "\n".join(hits) + "\n[stopped at %d hits]" % max_results
            except OSError:
                continue
    return "\n".join(hits) or "no matches for %r" % pattern


@tool(category="files")
def file_info(path: str) -> str:
    """Show size, type, permissions and modification time for a path.

    Args:
        path: Path to inspect.
    """
    import stat
    import time

    target = _resolve(path)
    if not target.exists():
        return "no such path: %s" % target
    info = target.stat()
    kind = "directory" if target.is_dir() else "file"
    return (
        "%s\n  type: %s\n  size: %s (%d bytes)\n  mode: %s\n  modified: %s"
        % (
            target,
            kind,
            _human(info.st_size),
            info.st_size,
            stat.filemode(info.st_mode),
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(info.st_mtime)),
        )
    )
