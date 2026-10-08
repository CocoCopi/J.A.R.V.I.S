"""The agent loop: prompt -> model -> tool calls -> results -> model -> answer.

This is the heart of Jarvis. Anything a provider can do, the agent can do, as
long as the provider emits tool calls. The loop is deliberately synchronous and
provider-agnostic so it runs happily in a phone terminal.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from . import session as session_store
from .providers.base import (
    Message,
    Provider,
    ProviderError,
    Response,
    ToolCall,
    ToolSpec,
    Usage,
)
from .tools import all_tools, get as get_tool, run_tool, specs as tool_specs
from .tools.base import Tool, ToolResult
from .tools.shell import looks_dangerous

log = logging.getLogger("jarvis.agent")

MAX_TOOL_OUTPUT = 12_000

DEFAULT_SYSTEM_PROMPT = """\
You are Jarvis, a personal assistant running directly on the user's Android \
device inside Termux. You have real tools and real consequences: when you call a \
tool it actually runs on the phone.

How to behave:
- Be decisive and brief. The user usually hears your reply through text-to-speech, \
so lead with the answer, then at most two short follow-up lines.
- Never claim you did something unless a tool call actually did it. If a tool \
fails, say what failed and try a different approach.
- Prefer acting over asking. Only ask when a destructive or irreversible action \
(delete, payment, message to a person) is about to happen and you are unsure.
- Chain tools freely: search the web, install a package, read a file, run a \
command, then summarise the result.
- Use `remember` for durable preferences and facts the user tells you about \
themselves, and `recall` before assuming something you cannot see.
- Format numbers and paths plainly; avoid heavy markdown (no tables, no big code \
fences) because the output is often spoken aloud.
"""


@dataclass
class AgentCallbacks:
    """Optional hooks so the CLI can render live progress."""

    on_text: Optional[Callable[[str], None]] = None
    on_tool_start: Optional[Callable[[ToolCall], None]] = None
    on_tool_end: Optional[Callable[[ToolCall, ToolResult], None]] = None
    on_step: Optional[Callable[[int, int], None]] = None
    on_info: Optional[Callable[[str], None]] = None

    def text(self, chunk: str) -> None:
        if self.on_text and chunk:
            self.on_text(chunk)

    def info(self, message: str) -> None:
        if self.on_info:
            self.on_info(message)


@dataclass
class AgentResult:
    text: str = ""
    steps: int = 0
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stopped_reason: str = "completed"


class Confirmer:
    """Decides whether a tool call is allowed to run.

    Rules, in order:

    1. ``config.deny_tools`` -> always refused.
    2. ``config.yolo`` -> always allowed.
    3. A tool flagged ``confirm=True``, a name in ``config.confirm_tools``, or a
       shell command matching a destructive pattern -> ask the hook.
    """

    def __init__(self, config, prompt_hook: Optional[Callable[[str, str, Dict], bool]] = None):
        self.config = config
        self.prompt_hook = prompt_hook or (lambda _name, _reason, _args: False)
        self.denied = set(config.deny_tools or ())
        self.always_confirm = set(config.confirm_tools or ())

    def needs_confirmation(self, name: str, args: Dict) -> Optional[str]:
        if name in self.denied:
            return "tool %r is disabled by configuration" % name
        if self.config.yolo:
            return None
        tool = get_tool(name)
        if tool is not None and tool.confirm:
            return "writes or executes on your device"
        if name in self.always_confirm:
            return "flagged in confirm_tools"
        if name == "run_shell":
            danger = looks_dangerous(str(args.get("command", "")))
            if danger:
                return danger
        return None

    def is_denied(self, name: str) -> bool:
        return name in self.denied

    def approve(self, name: str, reason: str, args: Dict) -> bool:
        return bool(self.prompt_hook(name, reason, args))


class Agent:
    """Runs the think/act loop for one conversation."""

    def __init__(
        self,
        provider: Provider,
        config,
        *,
        tool_names: Optional[List[str]] = None,
        disabled_tools: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        callbacks: Optional[AgentCallbacks] = None,
        confirmer: Optional[Confirmer] = None,
        history: Optional[List[Message]] = None,
        session_id: Optional[str] = None,
        extra_context: str = "",
        stream: bool = True,
        max_steps: Optional[int] = None,
        temperature: float = 0.4,
    ) -> None:
        self.provider = provider
        self.config = config
        self.tool_names = tool_names
        self.disabled_tools = list(disabled_tools or ())
        self.callbacks = callbacks or AgentCallbacks()
        self.confirmer = confirmer or Confirmer(config)
        self.stream = stream and getattr(provider, "supports_tools", True)
        self.temperature = temperature
        self.max_steps = int(max_steps or getattr(config, "max_steps", 12))
        self.session_id = session_id
        self.extra_context = extra_context
        self.usage = Usage()
        self.history: List[Message] = list(history or [])
        prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
        if extra_context:
            prompt = "%s\n\nCurrent context:\n%s" % (prompt, extra_context)
        self.system_prompt = prompt
        if not self.history or self.history[0].role != "system":
            self.history.insert(0, Message.system(prompt))

    # -- tool plumbing ----------------------------------------------------
    def tool_specs(self) -> List[ToolSpec]:
        if not getattr(self.provider, "supports_tools", True):
            return []
        names = self.tool_names
        if names is None:
            disabled = set(self.disabled_tools) | set(self.confirmer.denied)
            return [s for s in tool_specs(None, disabled) if s.name not in disabled]
        available = {t.name for t in all_tools()}
        selected = [n for n in names if n in available and n not in self.disabled_tools]
        missing = [n for n in names if n not in available]
        if missing:
            self.callbacks.info("ignoring unknown tools: %s" % ", ".join(missing))
        return tool_specs(selected)

    def _execute(self, call: ToolCall) -> ToolResult:
        args = dict(call.arguments or {})
        reason = self.confirmer.needs_confirmation(call.name, args)
        if reason and self.confirmer.is_denied(call.name):
            return ToolResult(output="refused: %s" % reason, ok=False)
        if reason and not self.confirmer.approve(call.name, reason, args):
            return ToolResult(
                output="user declined to run %s (%s)" % (call.name, reason), ok=False
            )
        tool: Optional[Tool] = get_tool(call.name)
        if tool is None:
            return ToolResult(output="unknown tool %r" % call.name, ok=False)
        result = run_tool(call.name, args)
        if len(result.output) > MAX_TOOL_OUTPUT:
            half = MAX_TOOL_OUTPUT // 2
            result.output = "%s\n... [truncated] ...\n%s" % (
                result.output[:half],
                result.output[-half:],
            )
        return result

    def _tool_message(self, call: ToolCall, result: ToolResult) -> Message:
        body = result.to_text()
        if not result.ok:
            body = "ERROR: %s" % body
        return Message.tool(call.id, call.name, body)

    # -- main loop --------------------------------------------------------
    def run(self, user_input: str, images: Optional[List] = None) -> AgentResult:
        message = Message(role="user", content=user_input, images=list(images or []))
        self.history.append(message)
        specs = self.tool_specs()
        result = AgentResult()
        for step in range(1, self.max_steps + 1):
            result.steps = step
            if self.callbacks.on_step:
                self.callbacks.on_step(step, self.max_steps)
            response = self._complete(specs)
            self.usage.add(response.usage)
            result.usage = self.usage
            if response.tool_calls:
                self.history.append(
                    Message.assistant(response.text, response.tool_calls)
                )
                for call in response.tool_calls:
                    result.tool_calls.append(call)
                    if self.callbacks.on_tool_start:
                        self.callbacks.on_tool_start(call)
                    outcome = self._execute(call)
                    if self.callbacks.on_tool_end:
                        self.callbacks.on_tool_end(call, outcome)
                    self.history.append(self._tool_message(call, outcome))
                self._persist()
                continue
            text = response.text.strip()
            if text:
                self.history.append(Message.assistant(text))
            result.text = text or result.text
            result.stopped_reason = "completed"
            break
        else:
            result.stopped_reason = "max_steps"
            result.text = (
                result.text
                or "I hit my step limit while working on that. Tell me to continue and I will."
            )
        self._persist()
        return result

    def _complete(self, specs: List[ToolSpec]) -> Response:
        """One provider turn, streaming text when possible."""
        provider = self.provider
        use_stream = self.stream and getattr(provider, "supports_tools", False)
        if use_stream:
            printed: List[str] = []
            provider.last_response = None  # never reuse a previous turn's response
            try:
                for delta in provider.stream(
                    self.history, specs, temperature=self.temperature
                ):
                    printed.append(delta)
                    self.callbacks.text(delta)
            except ProviderError:
                raise
            response = getattr(provider, "last_response", None)
            if response is None:
                response = Response(text="".join(printed))
            if not response.text and printed:
                response.text = "".join(printed)
            return response
        response = provider.chat(self.history, specs, temperature=self.temperature)
        self.callbacks.text(response.text)
        return response

    # -- session ----------------------------------------------------------
    def _persist(self) -> None:
        if not self.session_id:
            return
        try:
            session_store.save(
                self.session_id,
                self.history,
                meta={"model": getattr(self.provider, "model", ""), "provider": self.provider.name},
            )
        except OSError as exc:  # pragma: no cover - storage full etc.
            log.warning("could not persist session: %s", exc)

    # -- helpers ----------------------------------------------------------
    def reset(self) -> None:
        self.history = [Message.system(self.system_prompt)]

    def transcript(self) -> str:
        return json.dumps([m.to_dict() for m in self.history], indent=2)
