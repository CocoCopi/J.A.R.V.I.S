"""Gemini Live: realtime, hands-free voice conversation with tool calling.

The session is bidirectional: the phone streams microphone PCM up, Gemini streams
spoken PCM back down, and any function call it makes is executed through the
normal Jarvis tool belt.

Requires ``pip install "jarvis-termux[gemini]"`` and a Gemini key. The Live API
is currently available on the Gemini API (AI Studio) keys.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ..agent import Confirmer
from ..prompts import live_system_prompt
from ..providers.base import ToolCall
from ..tools import all_tools, run_tool, specs as tool_specs
from ..tools.base import ToolResult
from .audio import (
    INPUT_RATE,
    OUTPUT_RATE,
    Microphone,
    Speaker,
    create_microphone,
    create_speaker,
    mic_help,
    speaker_help,
)

log = logging.getLogger("jarvis.live")

#: Prebuilt Gemini voices that sound good for an assistant.
VOICES = ("Puck", "Charon", "Kore", "Fenrir", "Aoede", "Leda", "Orus", "Zephyr")

SEND_QUEUE_MAX = 200


@dataclass
class LiveCallbacks:
    """Hooks for rendering a live conversation (all optional)."""

    on_user_text: Optional[Callable[[str], None]] = None
    on_model_text: Optional[Callable[[str], None]] = None
    on_tool_start: Optional[Callable[[ToolCall], None]] = None
    on_tool_end: Optional[Callable[[ToolCall, ToolResult], None]] = None
    on_status: Optional[Callable[[str], None]] = None

    def user_text(self, text: str) -> None:
        if self.on_user_text and text.strip():
            self.on_user_text(text.strip())

    def model_text(self, text: str) -> None:
        if self.on_model_text and text:
            self.on_model_text(text)

    def status(self, message: str) -> None:
        if self.on_status:
            self.on_status(message)


@dataclass
class LiveOptions:
    voice: str = "Puck"
    temperature: float = 0.7
    tool_names: Optional[List[str]] = None
    enable_input_transcription: bool = True
    enable_output_transcription: bool = True
    #: Push-to-talk style: wait for the user to speak, no barge-in detection.
    silence_ms: int = 700
    language: Optional[str] = None


class LiveToolBridge:
    """Runs Jarvis tools for the live session, honouring confirmations."""

    def __init__(self, config, confirmer: Optional[Confirmer] = None) -> None:
        self.config = config
        self.confirmer = confirmer or Confirmer(config)

    def specs(self) -> List[Any]:
        return tool_specs(None, set(self.confirmer.denied))

    def _needs_confirmation(self, name: str, args: Dict[str, Any]) -> Optional[str]:
        return self.confirmer.needs_confirmation(name, args)

    def run(self, name: str, args: Dict[str, Any]) -> ToolResult:
        reason = self._needs_confirmation(name, args)
        if reason and not self.confirmer.approve(name, reason, args):
            return ToolResult(output="declined by user: %s" % reason, ok=False)
        return run_tool(name, args)

    def available_names(self) -> List[str]:
        if self.tool_names is not None:
            return [t.name for t in all_tools() if t.name in set(self.tool_names)]
        return [t.name for t in all_tools() if t.name not in self.confirmer.denied]

    tool_names: Optional[List[str]] = None


class LiveSession:
    """A single realtime voice conversation with Gemini."""

    def __init__(
        self,
        provider,
        config,
        *,
        options: Optional[LiveOptions] = None,
        callbacks: Optional[LiveCallbacks] = None,
        confirmer: Optional[Confirmer] = None,
        microphone: Optional[Microphone] = None,
        speaker: Optional[Speaker] = None,
        tools_enabled: bool = True,
    ) -> None:
        self.provider = provider
        self.config = config
        self.options = options or LiveOptions()
        self.callbacks = callbacks or LiveCallbacks()
        self.tools_enabled = tools_enabled
        self.bridge = LiveToolBridge(config, confirmer)
        self.bridge.tool_names = self.options.tool_names
        self.microphone = microphone
        self.speaker = speaker
        self._audio_queue: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue(maxsize=SEND_QUEUE_MAX)
        self._stop = asyncio.Event()
        self._text_input: "asyncio.Queue[str]" = asyncio.Queue()
        self.turns = 0

    # -- public API -------------------------------------------------------
    def stop(self) -> None:
        """Ask the session to finish (safe to call from another thread)."""
        try:
            loop = asyncio.get_event_loop_policy().get_event_loop()
        except RuntimeError:  # pragma: no cover
            loop = None
        if loop and loop.is_running():
            loop.call_soon_threadsafe(self._stop.set)
        else:
            self._stop.set()
        if self.microphone:
            self.microphone.stop()

    def send_text(self, text: str) -> None:
        """Inject a typed message into the live conversation."""
        self._text_input.put_nowait(text)

    def run(self) -> int:
        """Blocking entry point used by the CLI."""
        try:
            return asyncio.run(self._run_async())
        except KeyboardInterrupt:
            return 0

    # -- internals --------------------------------------------------------
    async def _run_async(self) -> int:
        from google.genai import types

        microphone = self.microphone or create_microphone(INPUT_RATE)
        speaker = self.speaker or create_speaker(OUTPUT_RATE)
        config = self._connect_config(types)
        self.callbacks.status(
            "connecting to %s (voice=%s)..." % (self.provider.live_model, self.options.voice)
        )
        try:
            async with self.provider.client.aio.live.connect(
                model=self.provider.live_model, config=config
            ) as session:
                self.callbacks.status("live - speak now (Ctrl+C or 'quit' to stop)")
                speaker.open()
                tasks = [
                    asyncio.create_task(self._pump_microphone(session, microphone, types)),
                    asyncio.create_task(self._pump_text(session)),
                    asyncio.create_task(self._receive(session, speaker, types)),
                ]
                await self._stop.wait()
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:  # pragma: no cover
            pass
        except Exception as exc:  # noqa: BLE001 - surface a readable error
            self.callbacks.status("live session error: %s" % _friendly(exc))
            return 1
        finally:
            try:
                microphone.stop()
            except Exception:  # noqa: BLE001
                pass
            speaker.close()
        return 0

    def _connect_config(self, types):
        speech = types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.options.voice)
            )
        )
        if self.options.language:
            try:
                speech.language_code = self.options.language
            except Exception:  # pragma: no cover
                pass
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=types.Content(
                parts=[types.Part(text=live_system_prompt())]
            ),
            speech_config=speech,
            temperature=self.options.temperature,
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    silence_duration_ms=self.options.silence_ms
                )
            ),
        )
        if self.tools_enabled:
            specs = self.bridge.specs()
            if specs:
                config.tools = [
                    types.Tool(
                        function_declarations=[
                            types.FunctionDeclaration(
                                name=spec.name,
                                description=spec.description,
                                parameters_json_schema=spec.parameters,
                            )
                            for spec in specs
                        ]
                    )
                ]
        return config

    # -- microphone -------------------------------------------------------
    async def _pump_microphone(self, session, microphone: Microphone, types) -> None:
        loop = asyncio.get_running_loop()

        def reader() -> None:
            try:
                for chunk in microphone.chunks():
                    if self._stop.is_set():
                        break
                    future = asyncio.run_coroutine_threadsafe(
                        self._audio_queue.put(chunk), loop
                    )
                    try:
                        future.result(timeout=5)
                    except Exception:  # noqa: BLE001 - queue full/closed
                        break
            except Exception as exc:  # noqa: BLE001
                log.debug("microphone stream ended: %s", exc)
                self.callbacks.status("microphone stopped: %s" % exc)
            finally:
                asyncio.run_coroutine_threadsafe(self._audio_queue.put(None), loop)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        try:
            while not self._stop.is_set():
                chunk = await self._audio_queue.get()
                if chunk is None:
                    break
                await session.send_realtime_input(
                    audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=%d" % INPUT_RATE)
                )
        except asyncio.CancelledError:
            pass

    async def _pump_text(self, session) -> None:
        while not self._stop.is_set():
            try:
                text = await asyncio.wait_for(self._text_input.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            if text.strip().lower() in {"quit", "exit", "stop"}:
                self._stop.set()
                return
            if text.strip():
                await session.send_realtime_input(text=text)

    # -- model output -----------------------------------------------------
    async def _receive(self, session, speaker: Speaker, types) -> None:
        try:
            async for message in session.receive():
                if self._stop.is_set():
                    break
                await self._handle_message(message, session, speaker, types)
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001
            self.callbacks.status("connection lost: %s" % _friendly(exc))
            self._stop.set()

    async def _handle_message(self, message, session, speaker: Speaker, types) -> None:
        if getattr(message, "tool_call", None):
            await self._handle_tool_call(message.tool_call, session, types)
        server_content = getattr(message, "server_content", None)
        if server_content is None:
            if getattr(message, "go_away", None):
                self.callbacks.status("server asked us to reconnect")
                self._stop.set()
            return

        if getattr(server_content, "interrupted", False):
            speaker.close()
            speaker.open()
            self.callbacks.status("[interrupted]")

        input_transcription = getattr(server_content, "input_transcription", None)
        if input_transcription and getattr(input_transcription, "text", None):
            self.callbacks.user_text(input_transcription.text)

        output_transcription = getattr(server_content, "output_transcription", None)
        if output_transcription and getattr(output_transcription, "text", None):
            self.callbacks.model_text(output_transcription.text)

        model_turn = getattr(server_content, "model_turn", None)
        for part in (getattr(model_turn, "parts", None) or []):
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                data = inline.data
                mime = (getattr(inline, "mime_type", "") or "")
                if "rate=24000" in mime or "audio/pcm" in mime:
                    speaker.write(data)
            text = getattr(part, "text", None)
            if text and not output_transcription:
                self.callbacks.model_text(text)

        if getattr(server_content, "turn_complete", False):
            self.turns += 1
            speaker.close()
            speaker.open()

    async def _handle_tool_call(self, tool_call, session, types) -> None:
        from google.genai import types as genai_types

        responses = []
        for function_call in getattr(tool_call, "function_calls", None) or []:
            call = ToolCall(
                id=getattr(function_call, "id", "") or "live_%d" % int(time.time() * 1000),
                name=getattr(function_call, "name", "") or "",
                arguments=ToolCall.parse_arguments(getattr(function_call, "args", None)),
            )
            if self.callbacks.on_tool_start:
                self.callbacks.on_tool_start(call)
            self.callbacks.status("running %s..." % call.name)
            loop = asyncio.get_running_loop()
            outcome = await loop.run_in_executor(
                None, lambda c=call: self.bridge.run(c.name, c.arguments)
            )
            if self.callbacks.on_tool_end:
                self.callbacks.on_tool_end(call, outcome)
            responses.append(
                genai_types.FunctionResponse(
                    id=call.id,
                    name=call.name,
                    response={"result": _jsonable(outcome)},
                )
            )
        if responses:
            await session.send_tool_response(function_responses=responses)

    # -- capability report -------------------------------------------------
    @staticmethod
    def capabilities() -> str:
        lines = ["live mode requirements:"]
        lines.append("  microphone: %s" % mic_help().replace("\n", "\n  "))
        lines.append("  speaker:    %s" % speaker_help().replace("\n", "\n  "))
        return "\n".join(lines)


def _jsonable(result: ToolResult) -> Any:
    if result.data is not None:
        return result.data
    return {"ok": result.ok, "output": result.output[:8000]}


def _friendly(exc: Exception) -> str:
    text = str(exc)
    if "API key" in text or "401" in text or "403" in text:
        return "%s (check GEMINI_API_KEY)" % text
    if "not found" in text.lower() and "model" in text.lower():
        return "%s (try another --live-model)" % text
    return text
