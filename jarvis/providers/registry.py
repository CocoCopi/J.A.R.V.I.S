"""Provider registry.

Providers register themselves with :func:`register`, so adding a backend is a
matter of dropping a module in this package and importing it in
``jarvis/providers/__init__.py``.
"""

from __future__ import annotations

from typing import Dict, List, Type

from .base import Provider, ProviderError

_REGISTRY: Dict[str, Type[Provider]] = {}
_ALIASES: Dict[str, str] = {
    "nim": "nvidia_nim",
    "nvidia": "nvidia_nim",
    "nvidia-nim": "nvidia_nim",
    "google": "gemini",
    "openai": "openai_compat",
    "ollama": "openai_compat",
    "openrouter": "openai_compat",
    "groq": "openai_compat",
    "local": "openai_compat",
    "lmstudio": "openai_compat",
}


def register(cls: Type[Provider]) -> Type[Provider]:
    """Class decorator that adds *cls* to the registry."""
    _REGISTRY[cls.name] = cls
    return cls


def normalize(name: str) -> str:
    key = (name or "").strip().lower().replace(" ", "_")
    return _ALIASES.get(key.replace("-", "_"), key.replace("-", "_")) if key not in _REGISTRY else key


def available() -> List[str]:
    return sorted(_REGISTRY)


def get_class(name: str) -> Type[Provider]:
    key = normalize(name)
    if key not in _REGISTRY:
        raise ProviderError(
            "unknown provider %r (available: %s)" % (name, ", ".join(available())),
            provider=name,
        )
    return _REGISTRY[key]


def create(name: str, config) -> Provider:  # noqa: ANN001
    """Instantiate provider *name* using *config* (a :class:`~jarvis.config.Config`)."""
    cls = get_class(name)
    provider_config = config.provider(cls.name if cls.name in config.providers else name)
    return cls(provider_config)
