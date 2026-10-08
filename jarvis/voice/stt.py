"""Speech to text.

Two interchangeable backends:

* **gemini** - send the recording to Gemini as audio input; no extra service and
  works with the same key you already use. This is the default.
* **openai_compat** - POST to ``/audio/transcriptions`` (works with OpenAI,
  Groq's whisper, LM Studio's whisper.cpp server, or a local
  ``faster-whisper-server``) when ``OPENAI_BASE_URL`` is configured.

Recordings are captured with the same :mod:`jarvis.voice.audio` backends used by
live mode, so a phone needs only ``termux-api``.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import requests

from .. import termux
from ..providers.base import ProviderError
from .audio import INPUT_RATE, Microphone, create_microphone, pcm_to_wav

log = logging.getLogger("jarvis.stt")

TRANSCRIBE_PROMPT = (
    "Transcribe this audio exactly as spoken. Return only the transcript text, "
    "with no preamble, labels or punctuation beyond what was said."
)


def record_to_wav(
    seconds: float = 5.0,
    *,
    microphone: Optional[Microphone] = None,
    rate: int = INPUT_RATE,
) -> Path:
    """Record *seconds* of audio to a temporary WAV file."""
    mic = microphone or create_microphone(rate)
    deadline = time.time() + max(1.0, seconds)
    chunks = bytearray()
    try:
        for chunk in mic.chunks():
            chunks.extend(chunk)
            if time.time() >= deadline:
                break
    except RuntimeError as exc:  # no capture backend on this device
        raise ProviderError("no microphone available: %s" % exc) from exc
    except OSError as exc:
        raise ProviderError("audio capture failed: %s" % exc) from exc
    finally:
        mic.stop()
    if not chunks:
        raise ProviderError(
            "no audio captured on %s - check the microphone permission for Termux:API"
            % termux.platform_name()
        )
    path = Path(tempfile.mkstemp(prefix="jarvis-stt-", suffix=".wav")[1])
    path.write_bytes(pcm_to_wav(bytes(chunks), rate))
    return path


def transcribe_file(
    path,
    *,
    config=None,
    provider_name: str = "gemini",
    model: Optional[str] = None,
    language: Optional[str] = None,
) -> str:
    """Transcribe a WAV/MP3/OGG file and return the text."""
    target = Path(path)
    if not target.is_file():
        raise ProviderError("no such audio file: %s" % target)
    backend = (provider_name or "gemini").lower()
    if backend in {"gemini", "google"}:
        return _transcribe_gemini(target, config, model, language)
    if backend in {"openai_compat", "openai", "whisper", "local"}:
        return _transcribe_openai(target, config, model, language)
    raise ProviderError("unknown STT backend %r (use gemini or openai_compat)" % provider_name)


def transcribe_microphone(
    seconds: float = 5.0,
    *,
    config=None,
    provider_name: str = "gemini",
    model: Optional[str] = None,
) -> str:
    """Record from the microphone and transcribe the result."""
    path = record_to_wav(seconds)
    try:
        return transcribe_file(path, config=config, provider_name=provider_name, model=model)
    finally:
        try:
            path.unlink()
        except OSError:
            pass


# -- backends ---------------------------------------------------------------
def _transcribe_gemini(path: Path, config, model: Optional[str], language: Optional[str]) -> str:
    try:
        from google import genai
        from google.genai import types
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ProviderError(
            "google-genai is required for Gemini transcription "
            '(pip install "jarvis-termux[gemini]")'
        ) from exc

    api_key = None
    if config is not None:
        api_key = config.provider("gemini").api_key
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ProviderError("GEMINI_API_KEY is not set (needed for speech-to-text)")

    client = genai.Client(api_key=api_key)
    model_name = model or (config.provider("gemini").model if config else None)
    if not model_name or model_name.lower() in {"auto", "default"}:
        from ..providers.gemini import DEFAULT_MODELS

        model_name = DEFAULT_MODELS[0]

    prompt = TRANSCRIBE_PROMPT
    if language:
        prompt += " The language is %s." % language
    data = path.read_bytes()
    mime = "audio/wav" if path.suffix.lower() == ".wav" else "audio/ogg"
    try:
        result = client.models.generate_content(
            model=model_name,
            contents=[
                types.Part.from_bytes(data=data, mime_type=mime),
                prompt,
            ],
            config=types.GenerateContentConfig(temperature=0),
        )
    except Exception as exc:  # noqa: BLE001 - SDK-specific error types
        raise ProviderError("transcription failed: %s" % exc) from exc
    return (getattr(result, "text", "") or "").strip()


def _transcribe_openai(path: Path, config, model: Optional[str], language: Optional[str]) -> str:
    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if config is not None:
        provider = config.provider("openai_compat")
        base_url = base_url or provider.base_url
        api_key = api_key or (provider.api_key or "")
    if not base_url:
        raise ProviderError(
            "OPENAI_BASE_URL is not set, so there is no whisper endpoint to use "
            "(set it, or use --stt-provider gemini)"
        )
    url = "%s/audio/transcriptions" % base_url.rstrip("/")
    model_name = model or "whisper-1"
    data = {"model": model_name}
    if language:
        data["language"] = language
    headers = {"Authorization": "Bearer %s" % api_key} if api_key else {}
    with path.open("rb") as handle:
        try:
            response = requests.post(
                url,
                headers=headers,
                data=data,
                files={"file": (path.name, handle, "audio/wav")},
                timeout=180,
            )
        except requests.RequestException as exc:
            raise ProviderError("transcription request failed: %s" % exc) from exc
    if response.status_code >= 400:
        raise ProviderError(
            "transcription failed: HTTP %s %s" % (response.status_code, response.text[:200])
        )
    try:
        body = response.json()
    except ValueError as exc:
        raise ProviderError("transcription returned non-JSON") from exc
    return str(body.get("text", "")).strip()
