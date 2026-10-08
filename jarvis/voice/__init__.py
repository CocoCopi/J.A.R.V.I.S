"""Voice subsystem: microphones, speakers, TTS and the Gemini Live session."""

from .audio import (
    INPUT_RATE,
    OUTPUT_RATE,
    Microphone,
    Speaker,
    create_microphone,
    create_speaker,
    pcm_to_wav,
    resample_pcm,
)
from .tts import speak, speak_async, stop_speaking
from .stt import transcribe_file, transcribe_microphone

__all__ = [
    "INPUT_RATE",
    "OUTPUT_RATE",
    "Microphone",
    "Speaker",
    "create_microphone",
    "create_speaker",
    "pcm_to_wav",
    "resample_pcm",
    "speak",
    "speak_async",
    "stop_speaking",
    "transcribe_file",
    "transcribe_microphone",
]
