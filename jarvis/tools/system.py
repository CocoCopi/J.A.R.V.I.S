"""General system + productivity tools that work on any Linux (Termux included)."""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

from .base import tool


@tool(category="system")
def current_datetime(offset_days: int = 0) -> str:
    """Get the current local date and time.

    Args:
        offset_days: Days to add (negative for the past).
    """
    now = datetime.now() + timedelta(days=offset_days)
    return now.strftime("%A, %d %B %Y at %H:%M:%S (%Z%z)").strip()


@tool(category="system")
def utc_now() -> str:
    """Get the current UTC timestamp in ISO-8601 form."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@tool(category="system")
def system_info() -> str:
    """Summarise the machine: OS, CPU, memory, disk, python, uptime."""
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
    }
    try:
        usage = shutil.disk_usage(os.getcwd())
        info["disk_free_gb"] = round(usage.free / 1e9, 2)
        info["disk_total_gb"] = round(usage.total / 1e9, 2)
    except OSError:
        pass
    try:
        with open("/proc/meminfo", "r") as handle:
            for line in handle:
                if line.startswith("MemTotal"):
                    info["mem_total_mb"] = int(line.split()[1]) // 1024
                elif line.startswith("MemAvailable"):
                    info["mem_available_mb"] = int(line.split()[1]) // 1024
                if "mem_total_mb" in info and "mem_available_mb" in info:
                    break
    except OSError:
        pass
    try:
        info["uptime_seconds"] = int(float(open("/proc/uptime").read().split()[0]))
    except (OSError, IndexError, ValueError):
        pass
    try:
        load = os.getloadavg()
        info["load_average"] = [round(v, 2) for v in load]
    except (OSError, AttributeError):
        pass
    return json.dumps(info, indent=2)


@tool(category="system")
def get_env(name: str) -> str:
    """Read an environment variable (never exposes API keys to the shell).

    Args:
        name: Variable name, e.g. ``HOME``.
    """
    value = os.environ.get(name)
    if value is None:
        return "%s is not set" % name
    return value


@tool(category="system")
def list_processes(pattern: Optional[str] = None, limit: int = 20) -> str:
    """List running processes, optionally filtered.

    Args:
        pattern: Substring to filter command lines by.
        limit: Maximum number of rows.
    """
    try:
        completed = subprocess.run(
            ["ps", "-eo", "pid,pcpu,pmem,comm,args"],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return "could not read process list: %s" % exc
    lines = completed.stdout.splitlines()
    if pattern:
        needle = pattern.lower()
        lines = [lines[0]] + [l for l in lines[1:] if needle in l.lower()]
    return "\n".join(lines[: max(2, int(limit))])


@tool(category="system", confirm=True)
def open_app(package: str, activity: Optional[str] = None) -> str:
    """Launch an Android app via ``am start`` (Termux/proot on Android).

    Args:
        package: Package name, e.g. ``com.termux`` or ``org.videolan.vlc``.
        activity: Optional explicit activity, e.g. ``.MainActivity``.
    """
    from ..termux import has_command, run_termux

    if not has_command("am"):
        return "`am` is not available here (Android-only tool)"
    component = package if activity is None else "%s/%s" % (package, activity)
    ok, out = run_termux(["am", "start", "-n", component], timeout=20)
    if not ok:
        ok, out = run_termux(
            ["am", "start", "-a", "android.intent.action.MAIN", "-p", package], timeout=20
        )
    return "launched %s" % component if ok else "could not launch %s: %s" % (package, out)


@tool(category="system")
def sleep(seconds: float) -> str:
    """Pause for a number of seconds (useful between other tool calls).

    Args:
        seconds: How long to wait; capped at 60.
    """
    import time

    duration = min(max(0.0, float(seconds)), 60.0)
    time.sleep(duration)
    return "waited %.1fs" % duration
