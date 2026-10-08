"""System-prompt helpers shared by the CLI and the live voice session."""

from __future__ import annotations

import os
import platform
import sys

from . import termux


def environment_summary() -> str:
    """A short description of the runtime, injected into the system prompt."""
    lines = [
        "- Platform: %s (%s)" % (termux.platform_name(), platform.platform()),
        "- Working directory: %s" % os.getcwd(),
        "- Python: %s" % sys.version.split()[0],
    ]
    if termux.is_termux():
        lines.append("- Termux: yes (termux-api available: %s)" % termux.termux_api_ready())
    elif termux.is_proot():
        lines.append("- PRoot distro under Termux: yes (termux-api usually NOT reachable)")
    else:
        lines.append("- Not running on Android; device tools will report unavailable")
    return "\n".join(lines)


def live_system_prompt() -> str:
    """Instructions for the realtime Gemini Live session."""
    from .agent import DEFAULT_SYSTEM_PROMPT

    return (
        DEFAULT_SYSTEM_PROMPT
        + "\nYou are in LIVE voice mode: the user is speaking to you in real time. "
        "Keep replies to one or two spoken sentences unless asked for detail. "
        "Never read out code, URLs or long lists verbatim; summarise instead.\n\n"
        "Runtime:\n" + environment_summary()
    )
