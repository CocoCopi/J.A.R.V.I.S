"""NVIDIA NIM provider (build.nvidia.com / self-hosted NIM microservices).

NIM speaks the OpenAI chat-completions dialect, so this is a thin specialisation
of :class:`~jarvis.providers.openai_compat.OpenAICompatProvider` that knows the
hosted endpoint, the recommended default model, and how to list the models your
key can actually reach.

Get a key at https://build.nvidia.com (it starts with ``nvapi-``).
"""

from __future__ import annotations

import logging
import os
from typing import List

import requests

from .base import ProviderError
from .openai_compat import OpenAICompatProvider
from .registry import register

log = logging.getLogger("jarvis.provider.nim")

#: Hosted NVIDIA inference endpoint. Override for a self-hosted NIM container.
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

#: Model used when none is configured, best first. All support tool calling.
DEFAULT_MODELS = (
    "meta/llama-3.3-70b-instruct",
    "nvidia/llama-3.3-nemotron-super-49b-v1",
    "meta/llama-3.1-70b-instruct",
    "qwen/qwen2.5-72b-instruct",
    "mistralai/mistral-nemotron",
    "meta/llama-3.1-8b-instruct",
)

#: Models that are chat/tool capable; used to filter ``/models`` output.
_TOOL_CAPABLE_HINTS = ("instruct", "nemotron", "qwen", "mistral", "deepseek", "gpt-oss")


@register
class NvidiaNimProvider(OpenAICompatProvider):
    name = "nvidia_nim"
    label = "NVIDIA NIM (build.nvidia.com)"
    required_env = ("NVIDIA_API_KEY",)
    supports_tools = True
    signup_url = "https://build.nvidia.com/explore/discover"

    base_url = NIM_BASE_URL
    default_model_id = DEFAULT_MODELS[0]

    def __init__(self, config) -> None:  # noqa: ANN001
        super().__init__(config)
        if not self.base_url or self.base_url == OpenAICompatProvider.base_url:
            self.base_url = os.environ.get("NVIDIA_BASE_URL", NIM_BASE_URL).rstrip("/")
        if not self.api_key:
            self.api_key = os.environ.get("NVIDIA_API_KEY", "")
            if self.api_key:
                self.session.headers["Authorization"] = "Bearer %s" % self.api_key
        # A NIM server on the LAN needs no key.
        self.requires_key = "integrate.api.nvidia.com" in self.base_url
        self.timeout = 180.0  # 70B models can be slow on the first token
        self.model = self.resolve_model()

    @classmethod
    def default_model(cls) -> str:
        return DEFAULT_MODELS[0]

    def resolve_model(self) -> str:
        """Pick a concrete model id from config, env, or the curated list."""
        configured = (self.model or "").strip()
        if configured and configured.lower() not in {"auto", "default"}:
            return configured
        return DEFAULT_MODELS[0]

    # -- model discovery --------------------------------------------------
    def list_models(self) -> List[str]:
        """Return chat-capable model ids the current key can use."""
        if not self.api_key and self.requires_key:
            raise ProviderError("NVIDIA_API_KEY is not set", provider=self.name)
        try:
            response = self.session.get(self._endpoint("models"), timeout=30)
        except requests.RequestException as exc:
            raise ProviderError("could not list models: %s" % exc, provider=self.name) from exc
        if response.status_code >= 400:
            raise ProviderError(
                self._error_message(response), provider=self.name, status=response.status_code
            )
        body = response.json()
        ids = [item.get("id", "") for item in body.get("data", []) if item.get("id")]
        chat = [i for i in ids if any(hint in i.lower() for hint in _TOOL_CAPABLE_HINTS)]
        return sorted(chat or ids)

    def health(self) -> str:
        if self.requires_key and not self.api_key:
            return "error: NVIDIA_API_KEY is not set (get one at %s)" % self.signup_url
        return super().health()
