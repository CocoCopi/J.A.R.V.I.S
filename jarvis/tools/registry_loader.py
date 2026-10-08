"""Loads the built-in tool modules and optional user plugins.

Plugins are Python files dropped in ``~/.config/jarvis/tools/*.py`` (or the
directory named by ``JARVIS_PLUGIN_DIR``). Each file just needs to import
``from jarvis.tools import tool`` and decorate functions - registering happens
as a side effect of importing.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
from pathlib import Path
from typing import List

log = logging.getLogger("jarvis.tools")

_BUILTIN_MODULES = (
    "jarvis.tools.shell",
    "jarvis.tools.files",
    "jarvis.tools.web",
    "jarvis.tools.termux",
    "jarvis.tools.system",
    "jarvis.tools.memory",
)

_loaded = False


def load_builtin_tools(force: bool = False) -> List[str]:
    """Import every built-in tool module once. Returns the module names."""
    global _loaded
    if _loaded and not force:
        return list(_BUILTIN_MODULES)
    for name in _BUILTIN_MODULES:
        importlib.import_module(name)
    _loaded = True
    return list(_BUILTIN_MODULES)


def plugin_dir() -> Path:
    override = os.environ.get("JARVIS_PLUGIN_DIR")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return base / "jarvis" / "tools"


def load_plugin(path) -> str:  # noqa: ANN001
    """Import a single plugin file and return its module name."""
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.suffix != ".py":
        raise ValueError("plugin must be a .py file: %s" % path)
    module_name = "jarvis_plugin_%s" % path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load plugin %s" % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module_name


def load_plugins() -> List[str]:
    """Load every ``*.py`` from the plugin directory. Failures are logged."""
    loaded: List[str] = []
    directory = plugin_dir()
    if not directory.is_dir():
        return loaded
    for path in sorted(directory.glob("*.py")):
        if path.name.startswith("_"):
            continue
        try:
            loaded.append(load_plugin(path))
        except Exception as exc:  # noqa: BLE001 - a bad plugin must not break Jarvis
            log.warning("plugin %s failed to load: %s", path.name, exc)
    return loaded
