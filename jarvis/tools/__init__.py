"""The Jarvis tool belt.

Importing this package registers every bundled tool. Third-party tools can be
added with ``@tool`` from :mod:`jarvis.tools.base`.
"""

from .base import (
    Tool,
    ToolError,
    ToolResult,
    all_tools,
    build_parameters,
    get,
    register_tool,
    run_tool,
    specs,
    tool,
)
from .registry_loader import (  # noqa: F401
    load_builtin_tools,
    load_plugin,
    load_plugins,
    plugin_dir,
)

__all__ = [
    "Tool",
    "ToolError",
    "ToolResult",
    "all_tools",
    "build_parameters",
    "get",
    "register_tool",
    "run_tool",
    "specs",
    "tool",
    "load_builtin_tools",
    "load_plugin",
    "load_plugins",
    "plugin_dir",
]
