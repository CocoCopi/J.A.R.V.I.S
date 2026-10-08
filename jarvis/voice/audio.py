"""Raw audio plumbing for Termux/Android (and desktop fallbacks).

Gemini Live speaks and listens in 16-bit little-endian mono PCM:

* input  (microphone)  -> 16 kHz
* output (speaker)     -> 24 kHz

Termux has no PyAudio by default, so we drive the device through command line
tools instead:

* capture: ``termux-microphone-record`` writing a WAV file we poll
* playback: ``sox``/``ffplay`` reading raw PCM from stdin, or ``termux-media-player``

If ``sounddevice`` happens to be installed (``pkg install portaudio`` +
``pip install sounddevice numpy``) it is used automatically - it has far lower
latency and is recommended for live conversations.
"""

from __future__ import annotations

import array
import os
import shutil
import struct
import subprocess
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from .. import termux

INPUT_RATE = 16_000  # mic -> Gemini Live
OUTPUT_RATE = 24_000  # Gemini Live -> speaker
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit

CHUNK_MS = 100  # how much audio we push per iteration


# -- PCM helpers (no numpy required) ---------------------------------------
def resample_pcm(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linearly resample 16-bit mono PCM. Pure Python, fast enough for 16 kHz."""
    if src_rate == dst_rate or not pcm:
        return pcm
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if not samples:
        return b""
    ratio = dst_rate / float(src_rate)
    out_len = max(1, int(len(samples) * ratio))
    out = array.array("h", bytes(out_len * 2))
    for index in range(out_len):
        position = index / ratio
        left = int(position)
        right = min(left + 1, len(samples) - 1)
        frac = position - left
        out[index] = int(samples[left] * (1 - frac) + samples[right] * frac)
    return out.tobytes()


def pcm_to_wav(pcm: bytes, rate: int = OUTPUT_RATE, channels: int = CHANNELS, width: int = SAMPLE_WIDTH) -> bytes:
    """Wrap raw PCM in a minimal RIFF/WAVE header."""
    byte_rate = rate * channels * width
    block_align = channels * width
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(pcm),
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        channels,
        rate,
        byte_rate,
        block_align,
        width * 8,
        b"data",
        len(pcm),
    )
    return header + pcm


def strip_wav_header(data: bytes) -> bytes:
    """Return the ``data`` chunk of a WAV file (tolerating extra chunks)."""
    if len(data) < 12 or data[:4] != b"RIFF":
        return data
    offset = 12
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        size = struct.unpack("<I", data[offset + 4 : offset + 8])[0]
        body = offset + 8
        if chunk_id == b"data":
            return data[body : body + size]
        offset = body + size + (size % 2)
    return b""


def pcm_params(wav: bytes) -> Tuple[int, int, int]:
    """Return ``(rate, channels, width)`` from a WAV header, defaulting sensibly."""
    rate, channels, width = INPUT_RATE, CHANNELS, SAMPLE_WIDTH
    if len(wav) >= 28 and wav[:4] == b"RIFF":
        try:
            channels = struct.unpack("<H", wav[22:24])[0] or 1
            rate = struct.unpack("<I", wav[24:28])[0] or INPUT_RATE
            width = max(1, struct.unpack("<H", wav[34:36])[0] // 8) if len(wav) >= 36 else 2
        except struct.error:  # pragma: no cover
            pass
    return rate, channels, width


def to_mono_16k(pcm: bytes, rate: int, width: int = 2) -> bytes:
    """Normalise arbitrary PCM to 16-bit mono at :data:`INPUT_RATE`."""
    if width == 1:  # 8-bit unsigned -> 16-bit signed
        pcm = array.array("h", ((b - 128) << 8 for b in pcm)).tobytes()
        width = 2
    if rate != INPUT_RATE:
        pcm = resample_pcm(pcm, rate, INPUT_RATE)
    return pcm


# -- speakers ---------------------------------------------------------------
class Speaker(ABC):
    """A sink for raw PCM audio."""

    def __init__(self, rate: int = OUTPUT_RATE) -> None:
        self.rate = rate

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def write(self, pcm: bytes) -> None: ...

    def close(self) -> None:
        pass

    def __enter__(self) -> "Speaker":
        self.open()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


class PipeSpeaker(Speaker):
    """Stream PCM into a command's stdin (sox ``play`` or ``ffplay``)."""

    def __init__(self, command: List[str], rate: int = OUTPUT_RATE, name: str = "pipe") -> None:
        super().__init__(rate)
        self.command = command
        self.name = name
        self.process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def open(self) -> None:
        if self.process is not None:
            return
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def write(self, pcm: bytes) -> None:
        if not pcm:
            return
        with self._lock:
            if self.process is None or self.process.stdin is None:
                return
            if self.process.poll() is not None:
                # Player died (device busy); restart once.
                self.process = None
                self.open()
                return
            try:
                self.process.stdin.write(pcm)
                self.process.stdin.flush()
            except (BrokenPipeError, ValueError):
                self.process = None

    def close(self) -> None:
        with self._lock:
            process, self.process = self.process, None
        if process is None:
            return
        try:
            if process.stdin:
                process.stdin.close()
            process.wait(timeout=5)
        except Exception:  # noqa: BLE001 - never block shutdown
            process.kill()


class TermuxMediaSpeaker(Speaker):
    """Fallback: buffer PCM into short WAV files and hand them to termux-api."""

    def __init__(self, rate: int = OUTPUT_RATE, chunk_seconds: float = 2.0) -> None:
        super().__init__(rate)
        self.chunk_seconds = chunk_seconds
        self._buffer = bytearray()
        self._dir = tempfile.mkdtemp(prefix="jarvis-audio-")
        self._counter = 0
        self._threads: List[threading.Thread] = []

    def open(self) -> None:
        pass

    def write(self, pcm: bytes) -> None:
        self._buffer.extend(pcm)
        limit = int(self.rate * SAMPLE_WIDTH * self.chunk_seconds)
        while len(self._buffer) >= limit:
            self._flush(bytes(self._buffer[:limit]))
            del self._buffer[:limit]

    def _flush(self, pcm: bytes) -> None:
        self._counter += 1
        path = os.path.join(self._dir, "chunk-%04d.wav" % self._counter)
        Path(path).write_bytes(pcm_to_wav(pcm, self.rate))
        self._play(path)

    def _play(self, path: str) -> None:
        if termux.has_command("termux-media-player"):
            termux.run_termux(["termux-media-player", "play", path], timeout=60)
            return
        if termux.has_command("aplay"):
            subprocess.run(["aplay", "-q", path], check=False)

    def close(self) -> None:
        if self._buffer:
            self._flush(bytes(self._buffer))
            self._buffer.clear()


class SoundDeviceSpeaker(Speaker):
    """Low-latency playback via ``sounddevice`` when it is installed."""

    def __init__(self, rate: int = OUTPUT_RATE) -> None:
        super().__init__(rate)
        self.stream = None

    def open(self) -> None:
        import sounddevice  # noqa: PLC0415 - optional dependency

        self.stream = sounddevice.RawOutputStream(
            samplerate=self.rate, channels=CHANNELS, dtype="int16", blocksize=0
        )
        self.stream.start()

    def write(self, pcm: bytes) -> None:
        if self.stream is not None and pcm:
            self.stream.write(pcm)

    def close(self) -> None:
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            finally:
                self.stream = None


def create_speaker(rate: int = OUTPUT_RATE) -> Speaker:
    """Pick the best available playback backend for this device."""
    if _sounddevice_available():
        try:
            return SoundDeviceSpeaker(rate)
        except Exception:  # noqa: BLE001 - fall through to CLI players
            pass
    if termux.has_command("play"):  # sox
        return PipeSpeaker(
            ["play", "-q", "-t", "raw", "-r", str(rate), "-e", "signed", "-b", "16", "-c", "1", "-"],
            rate,
            name="sox",
        )
    if termux.has_command("ffplay"):
        return PipeSpeaker(
            [
                "ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
                "-f", "s16le", "-ar", str(rate), "-ac", "1", "-i", "pipe:0",
            ],
            rate,
            name="ffplay",
        )
    if termux.has_command("aplay"):
        return PipeSpeaker(
            ["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-r", str(rate), "-c", "1", "-"],
            rate,
            name="aplay",
        )
    return TermuxMediaSpeaker(rate)


def speaker_help() -> str:
    return (
        "No streaming audio player found. Install one:\n"
        "  pkg install sox        (recommended: `play`)\n"
        "  pkg install ffmpeg     (provides ffplay)\n"
        "  pkg install termux-api (buffered fallback)"
    )


def _sounddevice_available() -> bool:
    if os.environ.get("JARVIS_NO_SOUNDDEVICE"):
        return False
    try:
        import sounddevice
        import numpy  # sounddevice needs numpy for buffer handling
    except Exception:  # noqa: BLE001 - optional dependency
        return False
    _ = (sounddevice, numpy)  # both must import for the fast path to work
    return True


# -- microphones -------------------------------------------------------------
class Microphone(ABC):
    """A source of raw 16-bit mono PCM at :data:`INPUT_RATE`."""

    def __init__(self, rate: int = INPUT_RATE) -> None:
        self.rate = rate
        self._stop = threading.Event()

    @abstractmethod
    def chunks(self) -> Iterator[bytes]:
        """Yield PCM chunks until :meth:`stop` is called."""

    def stop(self) -> None:
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def close(self) -> None:
        self.stop()

    def __enter__(self) -> "Microphone":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


class TermuxMicrophone(Microphone):
    """Capture through ``termux-microphone-record``.

    Two modes:

    * ``chunked`` (default): record fixed-length WAV files back to back. Adds a
      couple of seconds of latency but works with every termux-api build.
    * ``streaming``: start a single unlimited recording and tail the growing
      file, which feels far more responsive but needs a WAV-capable encoder.
    """

    def __init__(
        self,
        rate: int = INPUT_RATE,
        *,
        streaming: bool = True,
        chunk_seconds: float = 3.0,
    ) -> None:
        super().__init__(rate)
        self.streaming = streaming
        self.chunk_seconds = chunk_seconds
        self._dir = tempfile.mkdtemp(prefix="jarvis-mic-")

    def chunks(self) -> Iterator[bytes]:
        if not termux.has_command("termux-microphone-record"):
            raise RuntimeError(
                "termux-microphone-record not found. Install it with: pkg install termux-api"
            )
        if self.streaming:
            try:
                yield from self._stream_file()
                return
            except RuntimeError:
                # Encoder/file tailing unsupported -> fall back to chunks.
                pass
        yield from self._chunked()

    # -- streaming --------------------------------------------------------
    def _stream_file(self) -> Iterator[bytes]:
        path = os.path.join(self._dir, "stream.wav")
        if os.path.exists(path):
            os.unlink(path)
        ok, out = termux.run_termux(
            ["termux-microphone-record", "-f", path, "-l", "0", "-r", str(self.rate), "-c", "1", "-e", "wav"],
            timeout=15,
        )
        if not ok:
            raise RuntimeError("could not start recording: %s" % out)
        try:
            header: Optional[bytes] = None
            rate, width = self.rate, 2
            consumed = 0
            deadline = time.time() + 5
            while not os.path.exists(path) and time.time() < deadline and not self.stopped:
                time.sleep(0.1)
            if not os.path.exists(path):
                raise RuntimeError("microphone produced no file")
            while not self.stopped:
                size = os.path.getsize(path)
                if size <= consumed:
                    time.sleep(0.05)
                    continue
                with open(path, "rb") as handle:
                    handle.seek(consumed)
                    raw = handle.read(size - consumed)
                consumed = size
                if header is None:
                    if len(raw) < 44:
                        continue
                    header = raw[:44]
                    rate, _channels, width = pcm_params(raw[:64])
                    raw = strip_wav_header(raw)
                pcm = raw if (rate == INPUT_RATE and width == 2) else to_mono_16k(raw, rate, width)
                if pcm:
                    yield pcm
        finally:
            termux.stop_recording()

    # -- chunked ----------------------------------------------------------
    def _chunked(self) -> Iterator[bytes]:
        index = 0
        while not self.stopped:
            index += 1
            path = os.path.join(self._dir, "chunk-%04d.wav" % index)
            ok, out = termux.run_termux(
                [
                    "termux-microphone-record", "-f", path, "-l", str(int(self.chunk_seconds)),
                    "-r", str(self.rate), "-c", "1", "-e", "wav",
                ],
                timeout=self.chunk_seconds + 20,
            )
            if not ok:
                raise RuntimeError("recording failed: %s" % out)
            if not os.path.exists(path):
                continue
            data = Path(path).read_bytes()
            rate, _channels, width = pcm_params(data)
            pcm = strip_wav_header(data)
            if not pcm:
                # Non-WAV container: try ffmpeg.
                pcm = _decode_with_ffmpeg(path, self.rate)
                rate, width = self.rate, 2
            if not pcm:
                continue
            yield to_mono_16k(pcm, rate, width)


class SoundDeviceMicrophone(Microphone):
    """Live capture through ``sounddevice`` (low latency, needs portaudio)."""

    def __init__(self, rate: int = INPUT_RATE) -> None:
        super().__init__(rate)
        self._queue: "List[bytes]" = []
        self._stream = None

    def _callback(self, indata, _frames, _time_info, _status) -> None:  # noqa: ANN001
        self._queue.append(bytes(indata))

    def chunks(self) -> Iterator[bytes]:
        import sounddevice  # noqa: PLC0415

        block = int(self.rate * CHUNK_MS / 1000)
        with sounddevice.RawInputStream(
            samplerate=self.rate,
            channels=CHANNELS,
            dtype="int16",
            blocksize=block,
            callback=self._callback,
        ):
            while not self.stopped:
                if self._queue:
                    yield self._queue.pop(0)
                else:
                    time.sleep(0.01)

    def close(self) -> None:
        super().close()


def create_microphone(rate: int = INPUT_RATE, streaming: bool = True) -> Microphone:
    """Pick the best available capture backend."""
    if _sounddevice_available():
        try:
            return SoundDeviceMicrophone(rate)
        except Exception:  # noqa: BLE001
            pass
    return TermuxMicrophone(rate, streaming=streaming)


def _decode_with_ffmpeg(path: str, rate: int) -> bytes:
    if not shutil.which("ffmpeg"):
        return b""
    try:
        completed = subprocess.run(
            ["ffmpeg", "-loglevel", "quiet", "-i", path, "-f", "s16le", "-ac", "1", "-ar", str(rate), "-"],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return b""
    return completed.stdout or b""


def mic_help() -> str:
    return (
        "No microphone backend available. On Termux install:\n"
        "  pkg install termux-api     (plus the Termux:API app)\n"
        "For low latency also try:\n"
        "  pkg install portaudio && pip install sounddevice numpy"
    )
