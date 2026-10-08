"""A tiny stand-in for the NVIDIA NIM / OpenAI chat-completions API.

Used by the integration test to exercise the real HTTP client, SSE streaming,
tool-call extraction and the agent loop without an API key or network access.

Run standalone for manual poking::

    python tests/fake_nim.py 8765
    # then, in another shell:
    jarvis --provider nvidia_nim --base-url http://127.0.0.1:8765/v1 --yes ask "shell: echo hi"
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Tuple

TOOL_HINT = "shell:"


def _plan(messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """Decide the reply: text and/or a tool call."""
    user = ""
    for message in reversed(messages):
        if message.get("role") == "user":
            user = message.get("content") or ""
            break
    if any(m.get("role") == "tool" for m in messages):
        tool_output = [m for m in messages if m.get("role") == "tool"][-1].get("content", "")
        return "Command finished: %s" % tool_output.strip().replace("\n", " | "), []
    if user.lower().startswith(TOOL_HINT):
        return "", [
            {
                "id": "call_fake_1",
                "type": "function",
                "function": {
                    "name": "run_shell",
                    "arguments": json.dumps({"command": user.split(":", 1)[1].strip()}),
                },
            }
        ]
    return "Fake NIM says: %s" % (user or "(nothing)"), []


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args) -> None:  # keep the test output clean
        pass

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/").endswith("/models"):
            body = json.dumps({"data": [{"id": "meta/llama-3.3-70b-instruct"}, {"id": "qwen/qwen2.5-72b-instruct"}]})
            self._send(200, body.encode(), "application/json")
            return
        self._send(404, b"{}", "application/json")

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            self._send(400, b'{"error":{"message":"bad json"}}', "application/json")
            return

        auth = self.headers.get("Authorization") or ""
        if not auth.startswith("Bearer "):
            self._send(401, b'{"error":{"message":"missing api key"}}', "application/json")
            return

        text, tool_calls = _plan(payload.get("messages") or [])
        model = payload.get("model", "fake-model")
        if payload.get("stream"):
            self._stream(text, tool_calls, model)
        else:
            body: Dict[str, Any] = {
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls" if tool_calls else "stop",
                        "message": {"role": "assistant", "content": text, "tool_calls": tool_calls},
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            }
            self._send(200, json.dumps(body).encode(), "application/json")

    def _stream(self, text: str, tool_calls: List[Dict[str, Any]], model: str) -> None:
        chunks: List[Dict[str, Any]] = []
        for word in text.split(" "):
            chunks.append({"choices": [{"index": 0, "delta": {"content": word + " "}}]})
        if tool_calls:
            call = tool_calls[0]
            chunks.append(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": call["id"],
                                        "function": {"name": call["function"]["name"]},
                                    }
                                ]
                            },
                        }
                    ]
                }
            )
            arguments = call["function"]["arguments"]
            for start in range(0, len(arguments), 5):
                chunks.append(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {"index": 0, "function": {"arguments": arguments[start : start + 5]}}
                                    ]
                                },
                            }
                        ]
                    }
                )
        payload = bytearray()
        for chunk in chunks:
            payload += ("data: %s\n\n" % json.dumps(chunk)).encode()
        payload += b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}}\n\n'
        payload += b"data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(bytes(payload))
        self.wfile.flush()

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()


def start_server(port: int = 0) -> Tuple[ThreadingHTTPServer, int, threading.Thread]:
    """Start the fake API on *port* (0 = pick a free one). Returns server/port/thread."""
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1], thread


if __name__ == "__main__":
    listen = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    _server, actual, _thread = start_server(listen)
    print("fake NIM listening on http://127.0.0.1:%d/v1" % actual, flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        _server.shutdown()
