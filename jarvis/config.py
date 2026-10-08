"""Configuration loading for Jarvis.

Resolution order (first hit wins):

1. Explicit CLI flags.
2. Process environment variables (``NVIDIA_API_KEY`` ...).
3. ``.env`` discovered by walking up from the current directory.
4. The user config file at ``$JARVIS_CONFIG`` or ``~/.config/jarvis/config.toml``.

The config file is a small TOML document written by ``jarvis config set``::

    default_provider = "nvidia_nim"
    yolo = false

    [providers.nvidia_nim]
    api_key = "nvapi-..."
    model = "meta/llama-3.3-70b-instruct"

    [providers.gemini]
    api_key = "AIza..."
    model = "gemini-2.0-flash"
    live_model = "gemini-2.0-flash-live-001"
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:  # Python 3.11+
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    try:
        import tomli as _toml  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover
        _toml = None  # type: ignore[assignment]

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - dotenv is a hard dep, but stay safe

    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:  # type: ignore[misc]
        return False


ENV_PREFIX = "JARVIS_"

#: Environment variable that holds the config file path.
CONFIG_ENV = "JARVIS_CONFIG"

#: Per-provider environment variables: provider name -> {field: ENV_VAR}
PROVIDER_ENV: Dict[str, Dict[str, str]] = {
    "nvidia_nim": {
        "api_key": "NVIDIA_API_KEY",
        "model": "NVIDIA_MODEL",
        "base_url": "NVIDIA_BASE_URL",
    },
    "gemini": {
        "api_key": "GEMINI_API_KEY",
        "model": "GEMINI_MODEL",
        "live_model": "GEMINI_LIVE_MODEL",
    },
    "openai_compat": {
        "api_key": "OPENAI_API_KEY",
        "model": "OPENAI_MODEL",
        "base_url": "OPENAI_BASE_URL",
    },
}


def user_config_path() -> Path:
    """Return the path of the user's config file."""
    override = os.environ.get(CONFIG_ENV)
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return base / "jarvis" / "config.toml"


def _discover_dotenv() -> Optional[Path]:
    """Walk up from cwd looking for a ``.env`` file."""
    here = Path.cwd()
    for candidate in (here, *here.parents):
        env_file = candidate / ".env"
        if env_file.is_file():
            return env_file
    package_env = Path(__file__).resolve().parent.parent / ".env"
    return package_env if package_env.is_file() else None


@dataclass
class ProviderConfig:
    """Resolved settings for a single provider."""

    name: str
    api_key: Optional[str] = None
    model: Optional[str] = None
    live_model: Optional[str] = None
    base_url: Optional[str] = None
    #: Raw TOML table for provider-specific extras.
    extra: Dict[str, Any] = field(default_factory=dict)

    def merged(self, **overrides: Optional[str]) -> "ProviderConfig":
        """Return a copy with non-``None`` overrides applied."""
        data = {
            "name": self.name,
            "api_key": self.api_key,
            "model": self.model,
            "live_model": self.live_model,
            "base_url": self.base_url,
            "extra": dict(self.extra),
        }
        for key, value in overrides.items():
            if value is not None:
                data[key] = value
        return ProviderConfig(**data)


@dataclass
class Config:
    """Top level Jarvis configuration."""

    default_provider: str = "nvidia_nim"
    voice: str = "none"
    yolo: bool = False
    log_level: str = "info"
    max_steps: int = 12
    deny_tools: tuple = ()
    confirm_tools: tuple = (
        "run_shell",
        "python_exec",
        "write_file",
        "delete_path",
        "termux_sms_send",
    )
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)
    source: Optional[Path] = None

    # -- lookups ----------------------------------------------------------
    def provider(self, name: Optional[str] = None) -> ProviderConfig:
        """Return the config for *name* (or the default provider)."""
        name = name or self.default_provider
        if name in self.providers:
            return self.providers[name]
        return ProviderConfig(name=name)

    @property
    def data_dir(self) -> Path:
        override = os.environ.get(ENV_PREFIX + "DATA_DIR")
        if override:
            return Path(override).expanduser()
        return Path.home() / ".local" / "share" / "jarvis"


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _as_tuple(value: Any) -> tuple:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def load_config(
    path: Optional[Path] = None,
    *,
    use_dotenv: bool = True,
    env: Optional[Dict[str, str]] = None,
) -> Config:
    """Load configuration from disk + environment."""
    environ = dict(os.environ if env is None else env)
    if use_dotenv:
        dotenv = _discover_dotenv()
        if dotenv is not None:
            load_dotenv(dotenv, override=False)

    config_path = Path(path).expanduser() if path else user_config_path()
    raw: Dict[str, Any] = {}
    source: Optional[Path] = None
    if config_path.is_file() and _toml is not None:
        with config_path.open("rb") as handle:
            raw = _toml.load(handle)
        source = config_path

    providers: Dict[str, ProviderConfig] = {}
    for name, table in (raw.get("providers") or {}).items():
        table = dict(table)
        providers[name] = ProviderConfig(
            name=name,
            api_key=table.pop("api_key", None),
            model=table.pop("model", None),
            live_model=table.pop("live_model", None),
            base_url=table.pop("base_url", None),
            extra=table,
        )

    # Environment variables always win over the config file.
    for name, mapping in PROVIDER_ENV.items():
        found = {field: environ[var] for field, var in mapping.items() if environ.get(var)}
        if not found:
            continue
        current = providers.get(name, ProviderConfig(name=name))
        providers[name] = current.merged(**found)

    default_provider = (
        environ.get(ENV_PREFIX + "PROVIDER")
        or raw.get("default_provider")
        or "nvidia_nim"
    )

    return Config(
        default_provider=str(default_provider),
        voice=str(environ.get(ENV_PREFIX + "VOICE") or raw.get("voice") or "none"),
        yolo=_as_bool(environ.get(ENV_PREFIX + "YOLO"), _as_bool(raw.get("yolo"))),
        log_level=str(environ.get(ENV_PREFIX + "LOG_LEVEL") or raw.get("log_level") or "info"),
        max_steps=int(environ.get(ENV_PREFIX + "MAX_STEPS") or raw.get("max_steps") or 12),
        deny_tools=_as_tuple(environ.get(ENV_PREFIX + "DENY_TOOLS") or raw.get("deny_tools")),
        confirm_tools=_as_tuple(raw.get("confirm_tools")) or Config.confirm_tools,
        providers=providers,
        raw=raw,
        source=source,
    )


def save_config(config: Config, path: Optional[Path] = None) -> Path:
    """Persist *config* as TOML (a tiny hand-rolled writer, no extra deps)."""
    target = Path(path).expanduser() if path else user_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Jarvis configuration - managed by `jarvis config`",
        'default_provider = "%s"' % config.default_provider,
        'voice = "%s"' % config.voice,
        "yolo = %s" % ("true" if config.yolo else "false"),
        'log_level = "%s"' % config.log_level,
        "max_steps = %d" % config.max_steps,
    ]
    if config.deny_tools:
        lines.append("deny_tools = [%s]" % ", ".join('"%s"' % t for t in config.deny_tools))
    if config.confirm_tools:
        lines.append(
            "confirm_tools = [%s]" % ", ".join('"%s"' % t for t in config.confirm_tools)
        )
    lines.append("")
    for name, provider in sorted(config.providers.items()):
        lines.append("[providers.%s]" % name)
        for key in ("api_key", "model", "live_model", "base_url"):
            value = getattr(provider, key)
            if value:
                lines.append('%s = "%s"' % (key, value))
        for key, value in provider.extra.items():
            lines.append("%s = %s" % (key, _toml_value(value)))
        lines.append("")
    target.write_text("\n".join(lines), encoding="utf-8")
    try:
        target.chmod(0o600)  # holds API keys
    except OSError:  # pragma: no cover - non-POSIX filesystems
        pass
    return target


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return '"%s"' % str(value).replace('"', '\\"')


def print_config(config: Config, stream=None) -> None:
    """Print an effective-config summary with API keys redacted."""
    stream = stream or sys.stdout
    print("effective configuration", file=stream)
    print("  source           : %s" % (config.source or "<environment only>"), file=stream)
    print("  default provider : %s" % config.default_provider, file=stream)
    print("  voice            : %s" % config.voice, file=stream)
    print("  yolo             : %s" % config.yolo, file=stream)
    for name in sorted(config.providers):
        provider = config.providers[name]
        print("  [%s]" % name, file=stream)
        print("    api_key  : %s" % _redact(provider.api_key), file=stream)
        print("    model    : %s" % (provider.model or "<default>"), file=stream)
        if provider.live_model:
            print("    live     : %s" % provider.live_model, file=stream)
        if provider.base_url:
            print("    base_url : %s" % provider.base_url, file=stream)


def _redact(secret: Optional[str]) -> str:
    if not secret:
        return "<unset>"
    if len(secret) <= 8:
        return "*" * len(secret)
    return "%s...%s (%d chars)" % (secret[:4], secret[-4:], len(secret))
