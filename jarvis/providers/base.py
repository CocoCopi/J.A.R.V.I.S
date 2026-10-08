"""Provider-agnostic data types and the :class:`Provider` interface.

Everything above this layer (agent, CLI, voice) speaks only in terms of
:class:`Message`, :class:`ToolCall`, :class:`ToolSpec` and :class:`Response`.
Adding a new AI backend therefore means implementing one small class and
registering it - no changes to the agent loop.
"""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional


class ProviderError(RuntimeError):
    """Raised for any provider-side failure (auth, network, bad response)."""

    def __init__(self, message: str, *, provider: str = "", status: Optional[int] = None):
        super().__init__(message)
        self.provider = provider
        self.status = status

    def friendly(self) -> str:
        """A one-line, user-facing description of the failure."""
        parts = []
        if self.provider:
            parts.append("[%s]" % self.provider)
        parts.append(str(self))
        if self.status in (401, 403):
            parts.append("-> check your API key")
        elif self.status == 404:
            parts.append("-> check the model id / base URL")
        elif self.status == 429:
            parts.append("-> rate limited, retry shortly")
        return " ".join(parts)


@dataclass
class ToolSpec:
    """A tool exposed to the model, described with JSON Schema."""

    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass
class ToolCall:
    """A request from the model to run a tool."""

    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""

    @staticmethod
    def parse_arguments(raw: Any) -> Dict[str, Any]:
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return raw
        text = str(raw).strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"__raw__": text}
        return parsed if isinstance(parsed, dict) else {"value": parsed}


@dataclass
class Message:
    """One turn of conversation."""

    role: str  # system | user | assistant | tool
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    #: Optional images as ``(mime_type, bytes)`` tuples (vision models).
    images: List[Any] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"role": self.role}
        if self.content:
            data["content"] = self.content
        if self.tool_calls:
            data["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.raw_arguments
                        or json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in self.tool_calls
            ]
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.name:
            data["name"] = self.name
        return data

    @staticmethod
    def system(text: str) -> "Message":
        return Message(role="system", content=text)

    @staticmethod
    def user(text: str) -> "Message":
        return Message(role="user", content=text)

    @staticmethod
    def assistant(text: str = "", tool_calls: Optional[List[ToolCall]] = None) -> "Message":
        return Message(role="assistant", content=text, tool_calls=list(tool_calls or []))

    @staticmethod
    def tool(call_id: str, name: str, text: str) -> "Message":
        return Message(role="tool", content=text, tool_call_id=call_id, name=name)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens or (
            other.prompt_tokens + other.completion_tokens
        )


@dataclass
class Response:
    """A provider completion."""

    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    finish_reason: str = ""
    raw: Any = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class Provider(abc.ABC):
    """Base class every backend implements."""

    #: Short identifier used on the CLI (``--provider nvidia_nim``).
    name: str = "base"
    #: Human readable label shown by ``jarvis providers``.
    label: str = "Base provider"
    #: Env vars that must be present for this provider to be usable.
    required_env: tuple = ()
    #: Whether :meth:`chat` can accept tools.
    supports_tools: bool = True
    #: Whether this provider can drive ``jarvis live``.
    supports_live: bool = False
    #: Where the user can grab a key.
    signup_url: str = ""

    def __init__(self, config) -> None:  # noqa: ANN001 - avoids a circular import
        self.config = config
        self.model = (getattr(config, "model", None) or self.default_model())
        self.live_model = getattr(config, "live_model", None) or self.default_live_model()

    # -- to be implemented ------------------------------------------------
    @classmethod
    def default_model(cls) -> str:
        raise NotImplementedError

    @classmethod
    def default_live_model(cls) -> Optional[str]:
        return None

    @abc.abstractmethod
    def chat(
        self,
        messages: List[Message],
        tools: Optional[Iterable[ToolSpec]] = None,
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Response:
        """Run one completion."""

    # -- optional ---------------------------------------------------------
    def stream(
        self,
        messages: List[Message],
        tools: Optional[Iterable[ToolSpec]] = None,
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """Yield text deltas. Default implementation falls back to ``chat``."""
        response = self.chat(messages, tools, temperature=temperature, max_tokens=max_tokens)
        if response.text:
            yield response.text
        self.last_response = response

    #: Set by :meth:`stream` so callers can still see tool calls/usage.
    last_response: Optional[Response] = None

    def health(self) -> str:
        """Cheap reachability probe used by ``jarvis doctor``."""
        try:
            self.chat([Message.user("ping")], None, max_tokens=8)
            return "ok"
        except ProviderError as exc:
            return "error: %s" % exc

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return "<%s model=%s>" % (type(self).__name__, self.model)
