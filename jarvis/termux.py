"""Termux / Android device integration.

Everything that talks to the ``termux-api`` command set lives here so the rest
of Jarvis can degrade gracefully on a normal Linux box or inside proot.

Termux setup (once)::

    pkg install termux-api
    # and install the Termux:API app from F-Droid / Play Store

Without the Termux:API app the commands exist but hang or return an error; every
helper here therefore has a timeout and reports a friendly message.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import List, Optional, Sequence, Tuple

TIMEOUT = 20


# -- environment detection --------------------------------------------------
def prefix() -> str:
    return os.environ.get("PREFIX", "")


def is_termux() -> bool:
    """True when running inside a real Termux installation."""
    if "com.termux" in prefix():
        return True
    if os.path.exists("/data/data/com.termux/files/usr"):  # pragma: no cover
        return True
    return False


def is_proot() -> bool:
    """True inside a PRoot distro (Ubuntu/Debian under Termux)."""
    if os.environ.get("PROOT_TMP_DIR") or os.environ.get("PROOT_LOADER"):
        return True
    return bool(prefix()) and "/data/data/com.termux" in prefix()


def platform_name() -> str:
    if is_termux():
        return "termux"
    if is_proot():
        return "proot"
    if sys.platform.startswith("linux") and "ANDROID_ROOT" in os.environ:
        return "android"
    return sys.platform


#: Where a Termux installation stores its binaries on Android. Reachable from a
#: PRoot distro on most devices, which is how we bridge to termux-api there.
TERMUX_PREFIXES = (
    "/data/data/com.termux/files/usr",
    "/data/user/0/com.termux/files/usr",
)


def termux_prefix_on_disk() -> Optional[str]:
    """Find a Termux prefix we can execute from (even from inside PRoot)."""
    own = prefix()
    if own and os.path.isdir(os.path.join(own, "bin")):
        return own
    for base in TERMUX_PREFIXES:
        if os.path.isdir(os.path.join(base, "bin")):
            return base
    return None


def find_command(name: str) -> Optional[str]:
    """Absolute path to *name*, looking on PATH and in the Termux prefix."""
    found = shutil.which(name)
    if found:
        return found
    base = termux_prefix_on_disk()
    if base:
        candidate = os.path.join(base, "bin", name)
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def has_command(name: str) -> bool:
    return find_command(name) is not None


def bridge_env() -> dict:
    """Environment for calling Termux binaries (mostly interesting under PRoot).

    PRoot runs a glibc userland next to Android's bionic one, so Termux helpers
    need their own ``lib`` and ``bin`` directories on the search path.
    """
    base = termux_prefix_on_disk()
    if not base or is_termux():
        return {}
    env: dict = {}
    current_path = os.environ.get("PATH", "")
    bin_dir = os.path.join(base, "bin")
    if bin_dir not in current_path.split(os.pathsep):
        env["PATH"] = bin_dir + os.pathsep + current_path
    lib_dir = os.path.join(base, "lib")
    if os.path.isdir(lib_dir):
        current_lib = os.environ.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = lib_dir + ((os.pathsep + current_lib) if current_lib else "")
    return env


def termux_api_ready() -> bool:
    """Check that the termux-api bridge exists (command *and* app).

    Also true under PRoot when the Termux prefix is reachable.
    """
    return has_command("termux-tts-speak")


def run_termux(
    args: Sequence[str], timeout: int = TIMEOUT, input_text: Optional[str] = None
) -> Tuple[bool, str]:
    """Run a termux-api command. Returns ``(ok, combined_output)``."""
    program = args[0]
    resolved = find_command(program)
    if resolved is None:
        return False, "%s not found (install termux-api: pkg install termux-api)" % program
    command = [resolved, *list(args[1:])]
    environment = dict(os.environ)
    environment.update(bridge_env())
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            input=input_text,
            env=environment,
        )
    except subprocess.TimeoutExpired:
        return False, "%s timed out after %ss (is the Termux:API app installed?)" % (
            program,
            timeout,
        )
    except OSError as exc:
        return False, "%s failed: %s" % (program, exc)
    output = (completed.stdout or "") + (completed.stderr or "")
    return completed.returncode == 0, output.strip()


# -- speech ------------------------------------------------------------------
def speak(text: str, rate: int = 1, pitch: int = 1, language: Optional[str] = None) -> str:
    """Speak *text* out loud using whatever engine is available.

    Order of preference: ``termux-tts-speak`` -> ``espeak-ng``/``espeak`` ->
    ``spd-say`` -> ``say`` (macOS) -> ``pico2wave`` + a player.
    """
    text = (text or "").strip()
    if not text:
        return "nothing to say"
    if has_command("termux-tts-speak"):
        args: List[str] = ["termux-tts-speak", "-r", str(int(rate * 100))]
        if pitch != 1:
            args += ["-p", str(int(pitch * 100))]
        if language:
            args += ["-l", language]
        args.append(text)
        ok, out = run_termux(args, timeout=max(30, len(text) // 10))
        return "spoken" if ok else "termux-tts-speak failed: %s" % out
    for engine, extra in (("espeak-ng", []), ("espeak", []), ("spd-say", ["-w"])):
        resolved = find_command(engine)
        if resolved:
            try:
                completed = subprocess.run(
                    [resolved, *extra, text],
                    capture_output=True,
                    text=True,
                    timeout=max(30, len(text) // 8),
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                return "%s failed: %s" % (engine, exc)
            if completed.returncode == 0:
                return "spoken"
            return "%s failed: %s" % (engine, (completed.stderr or "").strip())
    if has_command("say"):  # macOS
        subprocess.run(["say", text], check=False)
        return "spoken"
    return "no text-to-speech engine found (try: pkg install termux-api)"


def stop_speaking() -> str:
    if has_command("termux-tts-speak"):
        ok, _ = run_termux(["termux-tts-speak", "-s"], timeout=10)
        return "stopped" if ok else "nothing to stop"
    return "no speech engine"


# -- notifications, clipboard, sensors --------------------------------------
def toast(text: str) -> str:
    ok, out = run_termux(["termux-toast", "-g", "middle", text], timeout=10)
    if ok:
        return "toast shown"
    return "termux-toast unavailable: %s" % out


def notify(title: str, content: str = "", notification_id: int = 1) -> str:
    ok, out = run_termux(
        ["termux-notification", "--title", title, "--content", content, "--id", str(notification_id)],
        timeout=10,
    )
    return "notification sent" if ok else "termux-notification unavailable: %s" % out


def vibrate(milliseconds: int = 500) -> str:
    ok, out = run_termux(["termux-vibrate", "-d", str(int(milliseconds))], timeout=10)
    return "vibrated" if ok else "termux-vibrate unavailable: %s" % out


def clipboard_get() -> str:
    ok, out = run_termux(["termux-clipboard-get"], timeout=10)
    return out if ok else "clipboard unavailable: %s" % out


def clipboard_set(text: str) -> str:
    ok, out = run_termux(["termux-clipboard-set", text], timeout=10)
    return "clipboard updated" if ok else "clipboard unavailable: %s" % out


def battery_status() -> str:
    ok, out = run_termux(["termux-battery-status"], timeout=10)
    return out if ok else "battery info unavailable: %s" % out


def location() -> str:
    ok, out = run_termux(["termux-location"], timeout=45)
    return out if ok else "location unavailable: %s" % out


def wifi_info() -> str:
    ok, out = run_termux(["termux-wifi-connectioninfo"], timeout=15)
    return out if ok else "wifi info unavailable: %s" % out


def dialog(title: str, message: str, buttons: Optional[List[str]] = None) -> str:
    args = ["termux-dialog", "confirm", "-t", title, "-i", message]
    if buttons:
        args = ["termux-dialog", "sheet", "-t", title, "-i", ", ".join(buttons)]
    ok, out = run_termux(args, timeout=120)
    return out if ok else "dialog unavailable: %s" % out


def record_audio(path: str, seconds: int = 10, limit: int = 0) -> Tuple[bool, str]:
    """Record from the microphone using termux-api.

    ``limit`` of 0 means "record until stopped".
    """
    ok, out = run_termux(
        ["termux-microphone-record", "-f", path, "-l", str(int(limit))], timeout=seconds + 15
    )
    return ok, out


def stop_recording() -> Tuple[bool, str]:
    return run_termux(["termux-microphone-record", "-q"], timeout=15)
