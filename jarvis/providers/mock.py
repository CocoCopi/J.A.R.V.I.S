"""Offline mock provider.

Two jobs:

1. Let people try Jarvis (and the tool loop) with no API key and no network.
2. Serve as the shortest possible template for a new backend.

Tiny bit of behaviour so it is actually useful::

    jarvis --provider mock ask "hello"          # echoes, lists tools
    jarvis --provider mock ask "shell: uname -a"  # asks to run a real command

Enable it in tests by passing ``--provider mock``.
"""

from __future__ import annotations

import time
from typing import Iterable, Iterator, List, Optional

from .base import Message, Provider, Response, ToolCall, ToolSpec, Usage
from .registry import register


@register
class MockProvider(Provider):
    name = "mock"
    label = "Offline mock (no network required; for tests and demos)"
    required_env = ()
    supports_tools = True
    supports_live = False

    @classmethod
    def default_model(cls) -> str:
        return "mock-1"

    def __init__(self, config) -> None:  # noqa: ANN001
        super().__init__(config)
        self.model = getattr(config, "model", None) or "mock-1"
        self.calls: List[str] = []

    def chat(
        self,
        messages: List[Message],
        tools: Optional[Iterable[ToolSpec]] = None,
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Response:
        specs = list(tools or [])
        prompt = _last_user_text(messages)
        self.calls.append(prompt)

        # If the previous turn was a tool result, summarise it.
        last_tool = next(
            (m for m in reversed(messages) if m.role == "tool"), None
        )
        if last_tool is not None and messages[-1].role == "tool":
            return Response(
                text="The tool said: %s" % last_tool.content.strip()[:400],
                usage=Usage(completion_tokens=8),
                model=self.model,
            )

        if prompt.lower().startswith("shell:") and any(s.name == "run_shell" for s in specs):
            command = prompt.split(":", 1)[1].strip()
            return Response(
                text="",
                tool_calls=[
                    ToolCall(
                        id="mock-call-1",
                        name="run_shell",
                        arguments={"command": command},
                    )
                ],
                usage=Usage(completion_tokens=12),
                model=self.model,
            )

        if "tool" in prompt.lower():
            names = ", ".join(s.name for s in specs[:12]) or "none"
            return Response(
                text="I have %d tools to hand, including: %s." % (len(specs), names),
                usage=Usage(completion_tokens=16),
                model=self.model,
            )

        return Response(
            text="Mock reply (no AI backend configured): you said %r." % prompt,
            usage=Usage(prompt_tokens=len(prompt.split()), completion_tokens=12),
            model=self.model,
        )

    def stream(
        self,
        messages: List[Message],
        tools: Optional[Iterable[ToolSpec]] = None,
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        response = self.chat(messages, tools, temperature=temperature, max_tokens=max_tokens)
        for word in response.text.split(" "):
            yield word + " "
            time.sleep(0.005)
        self.last_response = response
        return


def _last_user_text(messages: List[Message]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return ""
