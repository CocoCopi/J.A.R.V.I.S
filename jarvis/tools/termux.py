"""Tools that drive the Android device through the ``termux-api`` bridge.

On a non-Termux system these report a clear "not available" message instead of
hanging, so the model can fall back to something else.
"""

from __future__ import annotations

import os
import time

from .. import termux
from .base import tool


@tool(category="device")
def say(text: str, rate: int = 1, pitch: int = 1) -> str:
    """Speak text out loud through the phone's speaker.

    Args:
        text: Words to speak.
        rate: Speech speed multiplier (1 = normal).
        pitch: Voice pitch multiplier (1 = normal).
    """
    return termux.speak(text, rate=rate, pitch=pitch)


@tool(category="device")
def stop_speaking() -> str:
    """Interrupt any speech Jarvis is currently producing."""
    return termux.stop_speaking()


@tool(category="device")
def notify_user(title: str, message: str = "") -> str:
    """Post an Android notification.

    Args:
        title: Notification title.
        message: Notification body text.
    """
    return termux.notify(title, message, notification_id=int(time.time()) % 100000)


@tool(category="device")
def toast(message: str) -> str:
    """Show a brief Android toast message (no notification tray entry).

    Args:
        message: Text to flash on screen.
    """
    return termux.toast(message)


@tool(category="device")
def vibrate(milliseconds: int = 500) -> str:
    """Vibrate the device.

    Args:
        milliseconds: Duration in milliseconds.
    """
    return termux.vibrate(milliseconds)


@tool(category="device")
def clipboard_get() -> str:
    """Read the current Android clipboard contents."""
    return termux.clipboard_get()


@tool(category="device")
def clipboard_set(text: str) -> str:
    """Put text on the Android clipboard.

    Args:
        text: Text to copy.
    """
    return termux.clipboard_set(text)


@tool(category="device")
def battery_status() -> str:
    """Report battery level, charging state and temperature."""
    return termux.battery_status()


@tool(category="device")
def get_location(timeout_hint: int = 45) -> str:
    """Get the device's current GPS/network location as JSON.

    Args:
        timeout_hint: How long the fix is allowed to take.
    """
    return termux.location()


@tool(category="device")
def wifi_info() -> str:
    """Report the currently connected Wi-Fi network and IP details."""
    return termux.wifi_info()


@tool(category="device", confirm=True)
def send_sms(number: str, message: str) -> str:
    """Send an SMS (requires the SMS permission for Termux:API).

    Args:
        number: Recipient phone number.
        message: Message body.
    """
    ok, out = termux.run_termux(["termux-sms-send", "-n", str(number), str(message)], timeout=30)
    return "SMS sent to %s" % number if ok else "could not send SMS: %s" % out


@tool(category="device", confirm=True)
def make_call(number: str) -> str:
    """Place a phone call.

    Args:
        number: Number to dial.
    """
    ok, out = termux.run_termux(["termux-telephony-call", str(number)], timeout=30)
    return "calling %s" % number if ok else "call failed: %s" % out


@tool(category="device")
def take_photo(path: str = "~/jarvis-photo.jpg", camera_id: int = 0) -> str:
    """Take a photo with the device camera.

    Args:
        path: Where to store the JPEG.
        camera_id: 0 = rear camera, 1 = front camera.
    """
    target = os.path.expanduser(path)
    ok, out = termux.run_termux(
        ["termux-camera-photo", "-c", str(camera_id), target], timeout=45
    )
    if ok and os.path.exists(target):
        return "photo saved to %s" % target
    return "camera failed: %s" % (out or "unknown error")


@tool(category="device")
def torch(on: bool = True) -> str:
    """Turn the camera flash (torch) on or off.

    Args:
        on: True to enable the torch, False to disable it.
    """
    ok, out = termux.run_termux(["termux-torch", "on" if on else "off"], timeout=15)
    return "torch %s" % ("on" if on else "off") if ok else "torch unavailable: %s" % out


@tool(category="device")
def device_info() -> str:
    """Describe the runtime environment (Termux, proot, plain Linux)."""
    lines = [
        "platform: %s" % termux.platform_name(),
        "termux: %s" % termux.is_termux(),
        "proot: %s" % termux.is_proot(),
        "prefix: %s" % (termux.prefix() or "<none>"),
        "termux-api ready: %s" % termux.termux_api_ready(),
    ]
    for command in (
        "termux-tts-speak",
        "termux-microphone-record",
        "termux-notification",
        "termux-battery-status",
        "termux-clipboard-get",
        "termux-camera-photo",
        "ffmpeg",
        "sox",
    ):
        lines.append("%s: %s" % (command, termux.has_command(command)))
    return "\n".join(lines)
