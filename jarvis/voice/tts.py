"""Text to speech helpers built on top of :mod:`jarvis.termux`.

The model writes markdown; nobody wants to *hear* markdown. :func:`clean_for_speech`
turns a reply into something worth listening to, and :class:`SpeechQueue` lets
the CLI start talking while the model is still streaming tokens.
"""

from __future__ import annotations

import re
import threading
from typing import Callable, List, Optional

from .. import termux

_CODE_FENCE = re.compile(r"```.*?```", re.S)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_URL = re.compile(r"https?://\S+")
_EMPHASIS = re.compile(r"(\*\*|__|\*|_|~~)")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
_BULLET = re.compile(r"^\s*[-*+]\s+", re.M)
_NUMBERED = re.compile(r"^\s*\d+[.)]\s+", re.M)
_PATH = re.compile(r"(?<![\w/])/(?:[\w.@+-]+/){2,}[\w.@+-]*")
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\u2190-\u21FF\u2b00-\u2bff]+"
)

MAX_CHARS = 600


def clean_for_speech(text: str) -> str:
    """Strip the things that read badly out loud."""
    if not text:
        return ""
    out = _CODE_FENCE.sub(" code block omitted ", text)
    out = _INLINE_CODE.sub(r"\1", out)
    out = _LINK.sub(r"\1", out)
    out = _URL.sub("link", out)
    out = _HEADING.sub("", out)
    out = _BULLET.sub("", out)
    out = _NUMBERED.sub("", out)
    out = _PATH.sub("file path", out)
    out = _EMPHASIS.sub("", out)
    out = _EMOJI.sub("", out)
    out = re.sub(r"\n{2,}", ". ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out[:MAX_CHARS]


def split_sentences(text: str) -> List[str]:
    """Split text into speakable chunks at sentence boundaries."""
    pieces: List[str] = []
    for part in re.split(r"(?<=[.!?])\s+|\n+", text):
        part = part.strip()
        if not part:
            continue
        while len(part) > 280:
            cut = part.rfind(" ", 0, 280)
            cut = cut if cut > 40 else 280
            pieces.append(part[:cut].strip())
            part = part[cut:].strip()
        if part:
            pieces.append(part)
    return pieces


def speak(text: str, rate: int = 1, pitch: int = 1, clean: bool = True) -> str:
    """Speak *text* (blocking). Returns a short status string."""
    payload = clean_for_speech(text) if clean else text
    if not payload:
        return "nothing to say"
    return termux.speak(payload, rate=rate, pitch=pitch)


def stop_speaking() -> str:
    """Interrupt ongoing speech."""
    return termux.stop_speaking()


def speak_async(text: str, **kwargs) -> threading.Thread:
    """Speak in a background thread so the terminal stays responsive."""
    thread = threading.Thread(target=speak, args=(text,), kwargs=kwargs, daemon=True)
    thread.start()
    return thread


class SpeechQueue:
    """Speak text incrementally as a model streams its reply.

    Usage::

        queue = SpeechQueue(enabled=True)
        queue.feed(delta)          # as tokens arrive
        queue.flush()              # at end of turn
    """

    def __init__(
        self,
        enabled: bool = True,
        rate: int = 1,
        pitch: int = 1,
        on_spoken: Optional[Callable[[str], None]] = None,
        min_chars: int = 40,
    ) -> None:
        self.enabled = enabled
        self.rate = rate
        self.pitch = pitch
        self.on_spoken = on_spoken
        self.min_chars = min_chars
        self._buffer = ""
        self._lock = threading.Lock()
        self._threads: List[threading.Thread] = []
        self.spoken_text = ""

    @property
    def supported(self) -> bool:
        return termux.has_command("termux-tts-speak") or termux.has_command("espeak-ng") or termux.has_command(
            "espeak"
        ) or termux.has_command("spd-say") or termux.has_command("say")

    def feed(self, delta: str) -> None:
        if not self.enabled or not delta:
            return
        with self._lock:
            self._buffer += delta
            ready, self._buffer = self._extract(self._buffer)
        for sentence in ready:
            self._speak(sentence)

    def flush(self) -> None:
        with self._lock:
            remainder, self._buffer = self._buffer.strip(), ""
        if remainder:
            self._speak(remainder)
        for thread in self._threads:
            thread.join(timeout=120)

    def _extract(self, buffer: str) -> "tuple[List[str], str]":
        ready: List[str] = []
        while True:
            match = re.search(r"[.!?]\s|\n", buffer)
            if not match:
                break
            candidate = buffer[: match.end()].strip()
            buffer = buffer[match.end() :]
            cleaned = clean_for_speech(candidate)
            if len(cleaned) >= self.min_chars:
                ready.append(cleaned)
        return ready, buffer

    def _speak(self, text: str) -> None:
        if not text:
            return
        self.spoken_text += text + " "
        if self.on_spoken:
            self.on_spoken(text)
        thread = speak_async(text, rate=self.rate, pitch=self.pitch, clean=False)
        self._threads.append(thread)
