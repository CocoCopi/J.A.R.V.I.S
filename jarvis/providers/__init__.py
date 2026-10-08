"""AI backends.

Importing this package registers every bundled provider, so the rest of Jarvis
only ever deals with :class:`~jarvis.providers.base.Provider`.

Add your own::

    from jarvis.providers.base import Provider
    from jarvis.providers.registry import register

    @register
    class MyProvider(Provider):
        name = "mine"
        ...
"""

from .base import (
    Message,
    Provider,
    ProviderError,
    Response,
    ToolCall,
    ToolSpec,
    Usage,
)
from .registry import available, create, get_class, register

# Importing registers the providers. Keep the order stable: the generic
# OpenAI-compatible class must be registered before NVIDIA subclasses it.
from . import openai_compat as _openai_compat  # noqa: F401,E402
from . import nvidia_nim as _nvidia_nim  # noqa: F401,E402
from . import gemini as _gemini  # noqa: F401,E402
from . import mock as _mock  # noqa: F401,E402

__all__ = [
    "Message",
    "Provider",
    "ProviderError",
    "Response",
    "ToolCall",
    "ToolSpec",
    "Usage",
    "available",
    "create",
    "get_class",
    "register",
]
