"""Google Gemini provider via the official ``google-genai`` SDK.

Powers two things:

* plain text chat with function calling (``jarvis chat``),
* Gemini **Live** realtime voice sessions (``jarvis live``) - see
  :mod:`jarvis.voice.live`.

Install with ``pip install "jarvis-termux[gemini]"`` or ``pip install google-genai``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Iterable, Iterator, List, Optional

from .base import Message, Provider, ProviderError, Response, ToolCall, ToolSpec, Usage
from .registry import register

log = logging.getLogger("jarvis.provider.gemini")

DEFAULT_MODELS = (
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-1.5-flash",
)
DEFAULT_LIVE_MODELS = (
    "gemini-2.0-flash-live-001",
    "gemini-2.5-flash-preview-native-audio-dialog",
    "gemini-1.5-flash-8b",
)


def _require_sdk():
    try:
        from google import genai
        from google.genai import types

        _ = (genai, types)  # imported for their side effect: the package exists
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on user env
        raise ProviderError(
            "the 'google-genai' package is required for the Gemini provider "
            '(install with: pip install "jarvis-termux[gemini]")'
        ) from exc


@register
class GeminiProvider(Provider):
    name = "gemini"
    label = "Google Gemini (text + Live voice mode)"
    required_env = ("GEMINI_API_KEY",)
    supports_tools = True
    supports_live = True
    signup_url = "https://aistudio.google.com/apikey"

    def __init__(self, config) -> None:  # noqa: ANN001
        _require_sdk()
        super().__init__(config)
        self.api_key = getattr(config, "api_key", None) or os.environ.get("GEMINI_API_KEY", "")
        self.model = self._resolve(self.model, DEFAULT_MODELS[0])
        self.live_model = self._resolve(self.live_model, DEFAULT_LIVE_MODELS[0])
        if not self.api_key:
            raise ProviderError(
                "GEMINI_API_KEY is not set (get one at %s)" % self.signup_url,
                provider=self.name,
            )
        from google import genai

        self._genai = genai
        self.client = genai.Client(
            api_key=self.api_key,
            http_options=self._http_options(),
        )

    @staticmethod
    def _http_options():
        """Longer timeout + retries: phone networks are flaky."""
        from google.genai import types

        try:
            return types.HttpOptions(timeout=180_000, retry_options=types.HttpRetryOptions(attempts=3))
        except Exception:  # pragma: no cover - older/newer SDK shapes
            return None

    @staticmethod
    def _resolve(value: Optional[str], fallback: str) -> str:
        if value and value.strip().lower() not in {"auto", "default"}:
            return value.strip()
        return fallback

    @classmethod
    def default_model(cls) -> str:
        return DEFAULT_MODELS[0]

    @classmethod
    def default_live_model(cls) -> str:
        return DEFAULT_LIVE_MODELS[0]

    # -- conversion -------------------------------------------------------
    def _build_tools(self, tools: Optional[Iterable[ToolSpec]]):
        from google.genai import types

        specs = list(tools or [])
        if not specs:
            return None
        declarations = [
            types.FunctionDeclaration(
                name=spec.name,
                description=spec.description,
                parameters_json_schema=spec.parameters,
            )
            for spec in specs
        ]
        return [types.Tool(function_declarations=declarations)]

    def _build_contents(self, messages: List[Message]):
        """Convert our history into ``google.genai`` contents + system instruction."""
        from google.genai import types

        contents: List[Any] = []
        system_parts: List[str] = []
        for message in messages:
            if message.role == "system":
                system_parts.append(message.content)
                continue
            if message.role == "tool":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=message.name or "tool",
                                response={"result": _coerce_json(message.content)},
                            )
                        ],
                    )
                )
                continue
            if message.role == "assistant":
                parts: List[Any] = []
                if message.content:
                    parts.append(types.Part(text=message.content))
                for call in message.tool_calls:
                    parts.append(
                        types.Part(
                            function_call=types.FunctionCall(
                                name=call.name, args=call.arguments or {}
                            )
                        )
                    )
                if parts:
                    contents.append(types.Content(role="model", parts=parts))
                continue
            # user
            parts = []
            for mime, data in message.images or []:
                parts.append(types.Part.from_bytes(data=data, mime_type=mime))
            if message.content or not parts:
                parts.append(types.Part(text=message.content))
            contents.append(types.Content(role="user", parts=parts))
        return contents, "\n\n".join(p for p in system_parts if p)

    # -- entry points -----------------------------------------------------
    def chat(
        self,
        messages: List[Message],
        tools: Optional[Iterable[ToolSpec]] = None,
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Response:
        from google.genai import types

        contents, system = self._build_contents(messages)
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system or None,
            tools=self._build_tools(tools),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        try:
            result = self.client.models.generate_content(
                model=self.model, contents=contents, config=config
            )
        except Exception as exc:  # SDK raises its own error hierarchy
            raise self._wrap(exc) from exc
        return self._parse(result)

    def stream(
        self,
        messages: List[Message],
        tools: Optional[Iterable[ToolSpec]] = None,
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        from google.genai import types

        contents, system = self._build_contents(messages)
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system or None,
            tools=self._build_tools(tools),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        text_parts: List[str] = []
        calls: List[ToolCall] = []
        usage = Usage()
        try:
            for chunk in self.client.models.generate_content_stream(
                model=self.model, contents=contents, config=config
            ):
                text, chunk_calls = _extract(chunk)
                if text:
                    text_parts.append(text)
                    yield text
                calls.extend(chunk_calls)
                if getattr(chunk, "usage_metadata", None):
                    usage = _usage(chunk.usage_metadata)
        except Exception as exc:
            raise self._wrap(exc) from exc
        self.last_response = Response(
            text="".join(text_parts),
            tool_calls=_dedupe(calls),
            usage=usage,
            model=self.model,
        )
        return

    # -- parsing / errors -------------------------------------------------
    def _parse(self, result: Any) -> Response:
        text, calls = _extract(result)
        return Response(
            text=text,
            tool_calls=_dedupe(calls),
            usage=_usage(getattr(result, "usage_metadata", None)),
            model=self.model,
            finish_reason=str(getattr(result, "finish_reason", "") or ""),
            raw=result,
        )

    def _wrap(self, exc: Exception) -> ProviderError:
        text = str(exc)
        status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if status is not None and not isinstance(status, int):
            status = None
        if not self.api_key:
            text += " (no API key configured)"
        return ProviderError(text, provider=self.name, status=status)

    def health(self) -> str:
        try:
            self.chat([Message.user("reply with the single word: ok")], None, max_tokens=16)
            return "ok"
        except ProviderError as exc:
            return "error: %s" % exc


# -- helpers ---------------------------------------------------------------
def _extract(result: Any):
    """Pull text + function calls out of an SDK response or chunk."""
    texts: List[str] = []
    calls: List[ToolCall] = []
    candidates = getattr(result, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        for part in (getattr(content, "parts", None) or []):
            if getattr(part, "text", None):
                texts.append(part.text)
            function_call = getattr(part, "function_call", None)
            if function_call is not None:
                name = getattr(function_call, "name", None)
                if name:
                    calls.append(
                        ToolCall(
                            id=getattr(function_call, "id", None) or "call_%d" % len(calls),
                            name=name,
                            arguments=ToolCall.parse_arguments(getattr(function_call, "args", None)),
                            raw_arguments="",
                        )
                    )
    if not texts and getattr(result, "text", None):
        texts.append(result.text)
    return "".join(texts), calls


def _dedupe(calls: List[ToolCall]) -> List[ToolCall]:
    seen = set()
    out: List[ToolCall] = []
    for call in calls:
        key = (call.name, str(sorted(call.arguments.items())))
        if key in seen and call.id.startswith("call_"):
            continue
        seen.add(key)
        out.append(call)
    return out


def _usage(metadata: Any) -> Usage:
    if metadata is None:
        return Usage()
    return Usage(
        prompt_tokens=int(getattr(metadata, "prompt_token_count", 0) or 0),
        completion_tokens=int(getattr(metadata, "candidates_token_count", 0) or 0),
        total_tokens=int(getattr(metadata, "total_token_count", 0) or 0),
    )


def _coerce_json(text: str) -> Any:
    import json

    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text
