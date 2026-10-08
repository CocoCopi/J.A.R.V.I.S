"""Generic OpenAI-compatible chat-completions provider.

NVIDIA NIM, OpenRouter, Groq, Together, LM Studio, llama.cpp's server and
Ollama all expose ``POST {base_url}/chat/completions``. This module implements
that wire format once; :mod:`jarvis.providers.nvidia_nim` specialises it.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional

import requests

from .base import Message, Provider, ProviderError, Response, ToolCall, ToolSpec, Usage
from .registry import register

log = logging.getLogger("jarvis.provider.openai")

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


@register
class OpenAICompatProvider(Provider):
    """Any endpoint speaking the OpenAI chat-completions dialect."""

    name = "openai_compat"
    label = "OpenAI-compatible endpoint (Ollama, OpenRouter, Groq, ...)"
    required_env = ("OPENAI_API_KEY", "OPENAI_BASE_URL")
    supports_tools = True
    signup_url = "https://platform.openai.com/api-keys"

    #: Base URL including ``/v1``.
    base_url = "http://127.0.0.1:11434/v1"
    default_model_id = "qwen2.5:7b"
    #: Extra request headers/body fields for subclasses.
    extra_headers: Dict[str, str] = {}
    timeout = 120.0
    max_retries = 3

    def __init__(self, config) -> None:  # noqa: ANN001
        super().__init__(config)
        if getattr(config, "base_url", None):
            self.base_url = config.base_url.rstrip("/")
        self.api_key = getattr(config, "api_key", None) or ""
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "jarvis-termux/0.3",
            }
        )
        self.session.headers.update(self.extra_headers)
        if self.api_key:
            self.session.headers["Authorization"] = "Bearer %s" % self.api_key

    # -- metadata ---------------------------------------------------------
    @classmethod
    def default_model(cls) -> str:
        return cls.default_model_id

    @classmethod
    def default_live_model(cls) -> Optional[str]:
        return None

    def _endpoint(self, path: str) -> str:
        return "%s/%s" % (self.base_url.rstrip("/"), path.lstrip("/"))

    # -- serialisation ----------------------------------------------------
    def _payload(
        self,
        messages: List[Message],
        tools: Optional[Iterable[ToolSpec]],
        temperature: float,
        max_tokens: Optional[int],
        stream: bool,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [self._message(m) for m in messages],
            "temperature": temperature,
            "stream": stream,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        tool_list = list(tools or [])
        if tool_list:
            payload["tools"] = [
                {"type": "function", "function": spec.to_dict()} for spec in tool_list
            ]
            payload["tool_choice"] = "auto"
        return payload

    @staticmethod
    def _message(message: Message) -> Dict[str, Any]:
        data = message.to_dict()
        # Ollama chokes on an explicit empty content-with-tool_calls message,
        # but a plain assistant turn with no content is fine as "".
        if data.get("role") == "assistant" and not data.get("content") and "content" not in data:
            data["content"] = ""
        return data

    # -- error handling ---------------------------------------------------
    def _request(self, payload: Dict[str, Any], *, stream: bool) -> requests.Response:
        url = self._endpoint("chat/completions")
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.post(
                    url, json=payload, timeout=self.timeout, stream=stream
                )
            except requests.RequestException as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(min(2 ** attempt * 0.5, 8))
                continue
            if response.status_code in RETRY_STATUS and attempt < self.max_retries:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if (retry_after or "").isdigit() else 2 ** attempt * 0.5
                log.debug("retrying %s after %ss (HTTP %s)", url, delay, response.status_code)
                time.sleep(min(delay, 10))
                continue
            if response.status_code >= 400:
                raise ProviderError(
                    self._error_message(response),
                    provider=self.name,
                    status=response.status_code,
                )
            return response
        raise ProviderError(
            "request to %s failed: %s" % (url, last_error),
            provider=self.name,
        )

    @staticmethod
    def _error_message(response: requests.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            return "HTTP %s: %s" % (response.status_code, response.text[:300])
        if isinstance(body, dict):
            error = body.get("error") or body.get("detail") or body
            if isinstance(error, dict):
                return "HTTP %s: %s" % (response.status_code, error.get("message") or error)
            return "HTTP %s: %s" % (response.status_code, error)
        return "HTTP %s" % response.status_code

    # -- entry points -----------------------------------------------------
    def chat(
        self,
        messages: List[Message],
        tools: Optional[Iterable[ToolSpec]] = None,
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Response:
        payload = self._payload(messages, tools, temperature, max_tokens, stream=False)
        response = self._request(payload, stream=False)
        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderError("invalid JSON from %s" % self.base_url, provider=self.name) from exc
        return self._parse(body)

    def stream(
        self,
        messages: List[Message],
        tools: Optional[Iterable[ToolSpec]] = None,
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        payload = self._payload(messages, tools, temperature, max_tokens, stream=True)
        response = self._request(payload, stream=True)
        acc = _StreamAccumulator()
        try:
            for raw_line in response.iter_lines(decode_unicode=False):
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for delta in acc.feed(chunk):
                    yield delta
        finally:
            response.close()
        self.last_response = acc.finish(self.model)
        return

    # -- parsing ----------------------------------------------------------
    def _parse(self, body: Dict[str, Any]) -> Response:
        acc = _StreamAccumulator()
        choices = body.get("choices") or []
        if not choices:
            raise ProviderError(
                "provider returned no choices: %s" % json.dumps(body)[:300],
                provider=self.name,
            )
        # ``feed`` is a generator, so it must be drained to accumulate anything.
        for _ in acc.feed({"choices": [choices[0]], "usage": body.get("usage")}):
            pass
        return acc.finish(body.get("model") or self.model)

    def health(self) -> str:
        if not self.api_key and "localhost" not in self.base_url and "127.0.0.1" not in self.base_url:
            return "error: no API key configured"
        return super().health()


class _StreamAccumulator:
    """Reassembles OpenAI streaming chunks (text + interleaved tool calls)."""

    def __init__(self) -> None:
        self.text: List[str] = []
        self.tool_calls: Dict[int, Dict[str, Any]] = {}
        self.finish_reason = ""
        self.usage = Usage()

    def feed(self, chunk: Dict[str, Any]) -> Iterator[str]:
        usage = chunk.get("usage")
        if usage:
            self.usage = Usage(
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
                total_tokens=int(usage.get("total_tokens") or 0),
            )
        for choice in chunk.get("choices") or []:
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or choice.get("message") or {}
            content = delta.get("content")
            if content:
                if isinstance(content, list):  # some NIM models stream parts
                    content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
                if content:
                    self.text.append(content)
                    yield content
            for call in delta.get("tool_calls") or []:
                index = call.get("index", 0)
                slot = self.tool_calls.setdefault(
                    index, {"id": "", "name": "", "arguments": ""}
                )
                if call.get("id"):
                    slot["id"] = call["id"]
                function = call.get("function") or {}
                if function.get("name"):
                    slot["name"] = function["name"]
                if function.get("arguments"):
                    slot["arguments"] += function["arguments"]

    def finish(self, model: str) -> Response:
        calls: List[ToolCall] = []
        for index in sorted(self.tool_calls):
            slot = self.tool_calls[index]
            if not slot["name"]:
                continue
            calls.append(
                ToolCall(
                    id=slot["id"] or "call_%d" % index,
                    name=slot["name"],
                    arguments=ToolCall.parse_arguments(slot["arguments"]),
                    raw_arguments=slot["arguments"],
                )
            )
        return Response(
            text="".join(self.text),
            tool_calls=calls,
            usage=self.usage,
            model=model,
            finish_reason=self.finish_reason,
        )
