"""Command line interface - the ``jarvis`` command.

Subcommands::

    jarvis                       interactive chat (default)
    jarvis ask "..."             one-shot question, one-shot answer
    jarvis live                  realtime voice conversation (Gemini Live)
    jarvis voice [--seconds 5]   record -> transcribe -> answer -> speak
    jarvis say "text"            text to speech only
    jarvis tools [--json]        list available tools
    jarvis providers [--models]  list AI backends and status
    jarvis config ...            inspect/edit configuration
    jarvis doctor                environment + connectivity diagnostics
    jarvis sessions              list saved conversations
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Dict, List, Optional

from . import __version__, providers, session as session_store, ui
from .agent import Agent, AgentCallbacks, Confirmer
from .config import Config, load_config, print_config, save_config, user_config_path
from .prompts import environment_summary
from .providers.base import Provider, ProviderError, ToolCall
from .tools import all_tools, load_builtin_tools, load_plugins, run_tool
from .tools.base import ToolResult

HISTORY_LIMIT = 40


# -- provider wiring --------------------------------------------------------
def build_provider(
    config: Config,
    name: Optional[str] = None,
    *,
    model: Optional[str] = None,
    live_model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Provider:
    """Instantiate a provider with CLI overrides applied on top of config."""
    provider_name = name or config.default_provider
    cls = providers.get_class(provider_name)
    provider_config = config.provider(provider_name).merged(
        model=model, live_model=live_model, base_url=base_url, api_key=api_key
    )
    return cls(provider_config)


def require_provider(provider: Provider, wanted_live: bool = False) -> None:
    if wanted_live and not getattr(provider, "supports_live", False):
        raise SystemExit(
            "provider %r does not support live voice mode. Use --provider gemini."
            % provider.name
        )


# -- rendering helpers ------------------------------------------------------
def _short(value, limit: int = 160) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _render_tool_start(call: ToolCall) -> None:
    args = call.arguments or {}
    rendered = ", ".join("%s=%s" % (k, _short(v, 60)) for k, v in args.items())
    print(ui.style("  ⚙ %s" % call.name, "yellow") + ui.style("(%s)" % rendered, "grey"))


def _render_tool_end(call: ToolCall, result: ToolResult) -> None:
    body = _short(result.to_text(), 300)
    marker = "✓" if result.ok else "✗"
    colour = "green" if result.ok else "red"
    print(ui.style("  %s %s" % (marker, body), colour))


def make_confirmer(config: Config, *, interactive: bool, assume_yes: bool) -> Confirmer:
    """Build a :class:`Confirmer` with a terminal y/N prompt."""

    def ask(name: str, reason: str, args: Dict) -> bool:
        if assume_yes:
            return True
        if not interactive:
            ui.warn("refusing %s (%s) - rerun with --yes to allow it" % (name, reason))
            return False
        rendered = ", ".join("%s=%s" % (k, _short(v, 100)) for k, v in (args or {}).items())
        print(ui.style("\n  ⚠ %s wants to run %s" % ("Jarvis", name), "yellow"))
        print(ui.style("    reason: %s" % reason, "grey"))
        if rendered:
            print(ui.style("    args:   %s" % rendered, "grey"))
        answer = ui.prompt("  allow? [y/N/a(ll)] ").strip().lower()
        if answer in {"a", "all"}:
            config.yolo = True
            return True
        return answer in {"y", "yes"}

    return Confirmer(config, ask)


def make_agent(
    provider: Provider,
    config: Config,
    args: argparse.Namespace,
    *,
    session_id: Optional[str] = None,
    interactive: bool = True,
    history: Optional[List] = None,
    speak_replies: bool = False,
    callbacks: Optional[AgentCallbacks] = None,
) -> Agent:
    """Assemble an :class:`Agent` from parsed CLI arguments."""
    load_builtin_tools()
    plugins = load_plugins()
    if plugins:
        ui.info("loaded %d plugin(s)" % len(plugins))

    tool_names = None
    if getattr(args, "tools", None):
        tool_names = [t.strip() for t in args.tools.split(",") if t.strip()]

    confirmer = make_confirmer(
        config,
        interactive=interactive and not getattr(args, "no_confirm", False),
        assume_yes=bool(getattr(args, "yes", False)),
    )

    speech = None
    callbacks = callbacks or AgentCallbacks()
    base_text = callbacks.on_text

    if speak_replies:
        from .voice.tts import SpeechQueue

        queue = SpeechQueue(enabled=True, on_spoken=lambda s: None)
        if not queue.supported:
            ui.warn("no text-to-speech engine found; replies will not be spoken")
        else:
            def on_text(chunk: str) -> None:
                if base_text:
                    base_text(chunk)
                queue.feed(chunk)

            callbacks.on_text = on_text
            speech = queue

    agent = Agent(
        provider,
        config,
        tool_names=tool_names,
        callbacks=callbacks,
        confirmer=confirmer,
        session_id=session_id,
        history=history,
        extra_context=environment_summary(),
        stream=not getattr(args, "no_stream", False),
        temperature=getattr(args, "temperature", 0.4),
    )
    agent._speech_queue = speech  # type: ignore[attr-defined]
    return agent


def run_turn(agent: Agent, text: str, *, stream: bool = True) -> str:
    """Run one user turn and flush any speech queue."""
    printer = ui.stream_printer("  ") if stream else None
    previous = agent.callbacks.on_text
    if printer:
        def fan_out(chunk: str) -> None:
            if previous:
                previous(chunk)
            printer(chunk)

        agent.callbacks.on_text = fan_out
    try:
        result = agent.run(text)
    finally:
        if printer:
            printer.done()
            agent.callbacks.on_text = previous
        queue = getattr(agent, "_speech_queue", None)
        if queue is not None:
            queue.flush()
    return result.text


# -- commands ---------------------------------------------------------------
def cmd_chat(args: argparse.Namespace, config: Config) -> int:
    provider = build_provider(
        config, args.provider, model=args.model, live_model=args.live_model, base_url=args.base_url
    )
    session_id = args.session or session_store.new_session_id()
    history = session_store.load(session_id) if args.resume else []
    if history:
        ui.info("resuming session %s (%d messages)" % (session_id, len(history)))

    speak_replies = args.voice == "tts" or config.voice == "tts"
    agent = make_agent(
        provider,
        config,
        args,
        session_id=session_id,
        interactive=True,
        history=history,
        speak_replies=speak_replies,
    )
    tool_count = len(agent.tool_specs())
    ui.banner(
        provider.name,
        provider.model,
        voice="tts" if speak_replies else "none",
        tools=tool_count,
    )

    _setup_readline()

    if args.prompt:
        run_turn(agent, args.prompt, stream=not args.no_stream)
        return 0

    if not sys.stdin.isatty():
        data = sys.stdin.read()
        if data.strip():
            run_turn(agent, data.strip(), stream=not args.no_stream)
        return 0

    while True:
        try:
            line = ui.prompt("you › ").strip()
        except KeyboardInterrupt:
            print()
            continue
        if not line:
            continue
        if line.startswith("/"):
            if _handle_slash(line, agent, config, provider, args):
                continue
            break
        if line.startswith("!"):
            outcome = run_tool("run_shell", {"command": line[1:]})
            print("  " + outcome.to_text())
            continue
        try:
            run_turn(agent, line, stream=not args.no_stream)
        except ProviderError as exc:
            ui.error("  " + exc.friendly())
        except KeyboardInterrupt:
            print()
            ui.warn("  interrupted")
    return 0


def _handle_slash(line: str, agent: Agent, config: Config, provider: Provider, args) -> bool:
    """Handle a `/command`. Returns False to exit the loop."""
    parts = shlex.split(line)
    command = parts[0].lower()
    rest = line[len(parts[0]) :].strip()

    if command in {"/exit", "/quit", "/q"}:
        return False
    if command in {"/help", "/?"}:
        print(
            "\n".join(
                [
                    "  /help                 this message",
                    "  /tools                list available tools",
                    "  /model [name]         show or switch the model",
                    "  /provider [name]      show or switch the provider",
                    "  /voice [on|off]       speak replies out loud",
                    "  /yolo [on|off]        auto-approve tool confirmations",
                    "  /cost                 token usage for this session",
                    "  /clear                forget the conversation",
                    "  /session              show the session id and file",
                    "  /exit                 quit (also Ctrl+D)",
                    "  !<shell command>      run a shell command directly",
                ]
            )
        )
        return True
    if command == "/tools":
        for entry in all_tools():
            print("  %-18s %s  %s" % (entry.name, entry.category, _short(entry.description, 70)))
        return True
    if command == "/model":
        if rest:
            agent.provider.model = rest
            ui.ok("  model -> %s" % rest)
        else:
            ui.info("  model: %s" % agent.provider.model)
        return True
    if command == "/provider":
        if rest:
            try:
                agent.provider = build_provider(config, rest, model=None, base_url=None, live_model=None)
                ui.ok("  provider -> %s (%s)" % (rest, agent.provider.model))
            except ProviderError as exc:
                ui.error("  " + exc.friendly())
        else:
            ui.info("  provider: %s" % agent.provider.name)
        return True
    if command == "/voice":
        from .voice.tts import SpeechQueue

        if rest.lower() in {"on", "tts"}:
            queue = SpeechQueue()
            if queue.supported:
                agent._speech_queue = queue  # type: ignore[attr-defined]
                base = agent.callbacks.on_text
                agent.callbacks.on_text = lambda c: ((base(c) if base else None), queue.feed(c))[0]
                ui.ok("  spoken replies on")
            else:
                ui.warn("  no TTS engine available")
        else:
            agent._speech_queue = None  # type: ignore[attr-defined]
            ui.info("  spoken replies off")
        return True
    if command == "/yolo":
        config.yolo = rest.lower() in {"on", "true", "1", "yes"} if rest else not config.yolo
        agent.confirmer.config.yolo = config.yolo
        ui.info("  yolo: %s" % config.yolo)
        return True
    if command == "/cost":
        usage = agent.usage
        ui.info(
            "  prompt %d + completion %d = %d tokens"
            % (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)
        )
        return True
    if command == "/clear":
        agent.reset()
        ui.info("  conversation cleared")
        return True
    if command == "/session":
        ui.info("  session: %s" % (agent.session_id or "<none>"))
        return True
    ui.warn("  unknown command %s (try /help)" % command)
    return True


def cmd_ask(args: argparse.Namespace, config: Config) -> int:
    provider = build_provider(
        config, args.provider, model=args.model, live_model=args.live_model, base_url=args.base_url
    )
    speak = args.voice == "tts" or config.voice == "tts"
    agent = make_agent(provider, config, args, interactive=False, speak_replies=speak)
    # When piped, print only the final answer so `jarvis ask ... | cmd` is clean.
    stream = not args.no_stream and sys.stdout.isatty()
    try:
        text = run_turn(agent, args.question, stream=stream)
    except ProviderError as exc:
        ui.error(exc.friendly())
        return 2
    if not text and agent.history:
        text = agent.history[-1].content
    if not stream:
        print(text)
    return 0


def cmd_live(args: argparse.Namespace, config: Config) -> int:
    provider = build_provider(
        config,
        args.provider or "gemini",
        model=args.model,
        live_model=args.live_model,
        base_url=args.base_url,
    )
    require_provider(provider, wanted_live=True)
    from .voice.live import LiveCallbacks, LiveOptions, LiveSession, VOICES

    if args.voice_name and args.voice_name not in VOICES:
        ui.warn("unknown voice %r; using %s (options: %s)" % (args.voice_name, VOICES[0], ", ".join(VOICES)))

    def on_user(text: str) -> None:
        print(ui.style("  you   › ", "bright_green") + text)

    def on_model(text: str) -> None:
        sys.stdout.write(ui.style("  ", "cyan"))
        sys.stdout.write(text)
        sys.stdout.flush()

    session = LiveSession(
        provider,
        config,
        options=LiveOptions(
            voice=args.voice_name or VOICES[0],
            temperature=args.temperature,
            tool_names=[t.strip() for t in args.tools.split(",")] if args.tools else None,
        ),
        callbacks=LiveCallbacks(
            on_user_text=on_user,
            on_model_text=on_model,
            on_tool_start=_render_tool_start,
            on_tool_end=_render_tool_end,
            on_status=lambda message: print(ui.style("  %s" % message, "grey")),
        ),
        confirmer=make_confirmer(config, interactive=True, assume_yes=args.yes),
        tools_enabled=not args.no_tools,
    )

    import threading

    def watch_stdin() -> None:
        for line in sys.stdin:
            if line.strip().lower() in {"quit", "exit"}:
                session.stop()
                return
            session.send_text(line)

    if sys.stdin.isatty():
        threading.Thread(target=watch_stdin, daemon=True).start()
    ui.banner(provider.name, provider.live_model, voice="live/%s" % (args.voice_name or VOICES[0]), tools=0)
    print(ui.style("  tip: headphones prevent echo; Ctrl+C to end\n", "grey"))
    try:
        return session.run()
    except KeyboardInterrupt:
        print()
        return 0


def cmd_voice(args: argparse.Namespace, config: Config) -> int:
    """Push-to-talk: record, transcribe, think, speak."""
    from .voice import stt, tts

    provider = build_provider(
        config, args.provider, model=args.model, live_model=args.live_model, base_url=args.base_url
    )
    agent = make_agent(provider, config, args, interactive=True)

    if args.listen_only:
        text = stt.transcribe_microphone(seconds=args.seconds, config=config, provider_name=args.stt_provider)
        print(text)
        return 0

    while True:
        try:
            input(ui.style("\n  press Enter to talk (%ss), Ctrl+C to quit ", "grey"))
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        try:
            heard = stt.transcribe_microphone(
                seconds=args.seconds, config=config, provider_name=args.stt_provider
            )
        except ProviderError as exc:
            ui.error("  " + exc.friendly())
            return 2
        if not heard.strip():
            ui.warn("  (heard nothing)")
            continue
        print(ui.style("  you  › ", "bright_green") + heard)
        try:
            reply = run_turn(agent, heard, stream=not args.no_stream)
        except ProviderError as exc:
            ui.error("  " + exc.friendly())
            continue
        if not args.no_speak:
            tts.speak(reply or "")


def cmd_say(args: argparse.Namespace, config: Config) -> int:
    from .voice import tts

    text = args.text or sys.stdin.read()
    print(tts.speak(text, rate=args.rate, pitch=args.pitch))
    return 0


def cmd_tools(args: argparse.Namespace, config: Config) -> int:
    load_builtin_tools()
    load_plugins()
    entries = all_tools()
    if args.json:
        print(json.dumps([t.spec().to_dict() for t in entries], indent=2))
        return 0
    current = None
    for entry in entries:
        if entry.category != current:
            current = entry.category
            ui.heading("\n%s" % current)
        flag = " (confirm)" if entry.confirm else ""
        print("  %-18s %s%s" % (entry.name, _short(entry.description, 80), flag))
        if args.verbose:
            for name, schema in (entry.parameters.get("properties") or {}).items():
                print(ui.style("      %s: %s" % (name, schema.get("description") or schema.get("type")), "grey"))
    return 0


def cmd_providers(args: argparse.Namespace, config: Config) -> int:
    ui.heading("providers")
    for name in providers.available():
        cls = providers.get_class(name)
        provider_config = config.provider(name)
        configured = bool(provider_config.api_key) or not cls.required_env
        if name == "openai_compat":
            configured = bool(provider_config.base_url)
        marker = "✓" if configured else "·"
        print(
            "  %s %-14s %s\n      models: %s"
            % (
                marker,
                name,
                cls.label,
                provider_config.model or cls.default_model(),
            )
        )
        if not configured:
            print(ui.style("      set: %s" % ", ".join(cls.required_env), "grey"))
        if cls.supports_live:
            print(ui.style("      live model: %s" % (provider_config.live_model or cls.default_live_model()), "grey"))

    if args.models:
        name = args.provider or config.default_provider
        try:
            provider = build_provider(config, name)
        except ProviderError as exc:
            ui.error(exc.friendly())
            return 2
        lister = getattr(provider, "list_models", None)
        if not callable(lister):
            ui.warn("%s does not expose a model list" % name)
            return 0
        try:
            for model in lister():
                print("  %s" % model)
        except ProviderError as exc:
            ui.error(exc.friendly())
            return 2
    return 0


def cmd_config(args: argparse.Namespace, config: Config) -> int:
    if args.action in {None, "show"}:
        print_config(config)
        return 0
    if args.action == "path":
        print(user_config_path())
        return 0
    if args.action == "set":
        if not args.key or args.value is None:
            ui.error("usage: jarvis config set <key> <value>   (e.g. providers.nvidia_nim.model)")
            return 2
        _config_set(config, args.key, args.value)
        path = save_config(config)
        ui.ok("saved %s" % path)
        return 0
    ui.error("unknown config action %r" % args.action)
    return 2


def _config_set(config: Config, key: str, value: str) -> None:
    parts = key.split(".")
    if parts[0] == "providers" and len(parts) >= 3:
        name, field = parts[1], parts[2]
        provider = config.providers.setdefault(name, config.provider(name))
        if not hasattr(provider, field):
            provider.extra[field] = value
        else:
            setattr(provider, field, value)
        return
    if key in {"default_provider", "voice", "log_level"}:
        setattr(config, key, value)
        return
    if key in {"yolo"}:
        config.yolo = value.strip().lower() in {"1", "true", "yes", "on"}
        return
    if key == "max_steps":
        config.max_steps = int(value)
        return
    if key in {"deny_tools", "confirm_tools"}:
        setattr(config, key, tuple(v.strip() for v in value.split(",") if v.strip()))
        return
    raise SystemExit("unknown config key %r" % key)


def cmd_doctor(args: argparse.Namespace, config: Config) -> int:
    from . import termux
    from .voice.audio import create_microphone, create_speaker, mic_help, speaker_help

    ui.heading("jarvis doctor")
    print("  version:  %s" % __version__)
    print("  python:   %s" % sys.version.split()[0])
    print("  platform: %s" % termux.platform_name())
    print("  config:   %s" % (config.source or "<none - using environment>"))
    print()

    ui.heading("device")
    for command in (
        "termux-tts-speak",
        "termux-microphone-record",
        "termux-notification",
        "termux-battery-status",
        "termux-camera-photo",
        "play",
        "ffplay",
        "ffmpeg",
        "sox",
        "am",
    ):
        print("  %-26s %s" % (command, "ok" if termux.has_command(command) else "missing"))
    discovered = termux.termux_prefix_on_disk()
    print("  termux prefix:   %s" % (discovered or "<not found>"))
    if termux.is_proot() and discovered:
        bridged = sorted(termux.bridge_env())
        print("  proot bridge:    %s" % (", ".join(bridged) if bridged else "not needed"))
    print("  termux-api ready: %s" % termux.termux_api_ready())
    if not termux.termux_api_ready():
        print(ui.style("    -> pkg install termux-api, then install the Termux:API app", "grey"))
        if termux.is_proot():
            print(
                ui.style(
                    "    -> inside PRoot, Jarvis looks in /data/data/com.termux/files/usr/bin; "
                    "if that fails, run Jarvis from Termux itself",
                    "grey",
                )
            )
    print()

    ui.heading("audio")
    try:
        speaker = create_speaker()
        print("  speaker:    %s" % type(speaker).__name__)
        speaker.close()
    except Exception as exc:  # noqa: BLE001
        print("  speaker:    unavailable (%s)" % exc)
        print(ui.style("    " + speaker_help().replace("\n", "\n    "), "grey"))
    try:
        microphone = create_microphone()
        print("  microphone: %s" % type(microphone).__name__)
        microphone.stop()
    except Exception as exc:  # noqa: BLE001
        print("  microphone: unavailable (%s)" % exc)
        print(ui.style("    " + mic_help().replace("\n", "\n    "), "grey"))
    print()

    ui.heading("providers")
    for name in providers.available():
        cls = providers.get_class(name)
        provider_config = config.provider(name)
        configured = bool(provider_config.api_key)
        if name == "openai_compat":
            configured = bool(provider_config.base_url)
        line = "  %-14s %s" % (name, "configured" if configured or not cls.required_env else "not configured")
        if not configured and cls.required_env:
            line += ui.style("  (set %s)" % ", ".join(cls.required_env), "grey")
        print(line)
        if configured and args.online:
            try:
                provider = build_provider(config, name)
                print("      health: %s" % provider.health())
            except ProviderError as exc:
                print("      health: error: %s" % exc)
            except Exception as exc:  # noqa: BLE001
                print("      health: error: %s" % exc)
    print()

    ui.heading("tools")
    load_builtin_tools()
    plugins = load_plugins()
    entries = all_tools()
    print("  %d tools in %d categories" % (len(entries), len({t.category for t in entries})))
    print("  plugins loaded: %d (%s)" % (len(plugins), ", ".join(plugins) or "none"))
    print()

    ui.heading("environment")
    print(environment_summary().replace("\n", "\n  "))
    if args.online:
        print()
        ui.heading("network")
        import requests

        for label, url in (
            ("nvidia nim", "https://integrate.api.nvidia.com/v1/models"),
            ("gemini", "https://generativelanguage.googleapis.com"),
            ("search", "https://html.duckduckgo.com/html/?q=test"),
        ):
            try:
                response = requests.get(url, timeout=8)
                print("  %-12s reachable (HTTP %s)" % (label, response.status_code))
            except requests.RequestException as exc:
                print("  %-12s unreachable (%s)" % (label, exc))
    return 0


def cmd_sessions(args: argparse.Namespace, config: Config) -> int:
    rows = session_store.list_sessions()
    if not rows:
        ui.info("no saved sessions")
        return 0
    for row in rows:
        print("  %-16s %s  %2d turns  %s" % (row["id"], row["updated"], row["turns"], row["preview"]))
    return 0


def cmd_plugins(args: argparse.Namespace, config: Config) -> int:
    from .tools.registry_loader import plugin_dir

    ui.info("plugin directory: %s" % plugin_dir())
    plugin_dir().mkdir(parents=True, exist_ok=True)
    loaded = load_plugins()
    for name in loaded:
        print("  loaded %s" % name)
    if not loaded:
        ui.info("no plugins found (drop a .py file that uses @tool in that directory)")
    return 0


# -- plumbing ---------------------------------------------------------------
def _setup_readline() -> None:
    try:
        import readline
    except ImportError:  # pragma: no cover - Windows
        return
    history_file = config_dir() / "history"
    try:
        history_file.parent.mkdir(parents=True, exist_ok=True)
        readline.read_history_file(str(history_file))
    except (OSError, FileNotFoundError):
        pass
    readline.set_history_length(500)
    import atexit

    def save() -> None:
        try:
            readline.write_history_file(str(history_file))
        except OSError:
            pass

    atexit.register(save)


def config_dir() -> Path:
    override = os.environ.get("XDG_CONFIG_HOME")
    base = Path(override).expanduser() if override else Path.home() / ".config"
    return base / "jarvis"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jarvis",
        description="Jarvis - a tool-using voice assistant for Termux/Android.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version="jarvis %s" % __version__)
    parser.add_argument("--config", type=Path, default=None, help="path to config.toml")
    parser.add_argument("--provider", help="AI backend: " + ", ".join(providers.available()))
    parser.add_argument("--model", help="override the model id")
    parser.add_argument("--live-model", help="override the Gemini Live model id")
    parser.add_argument("--base-url", help="override the OpenAI-compatible base URL")
    parser.add_argument("--temperature", type=float, default=0.4, help="sampling temperature")
    parser.add_argument("--no-stream", action="store_true", help="wait for the full reply")
    parser.add_argument("--tools", help="comma separated allow-list of tool names")
    parser.add_argument("--yes", "-y", action="store_true", help="approve every tool call")
    parser.add_argument("--no-confirm", action="store_true", help="skip prompts (deny instead)")
    parser.add_argument("--plain", action="store_true", help="disable colours")

    sub = parser.add_subparsers(dest="command")

    chat = sub.add_parser("chat", help="interactive conversation (default)")
    chat.add_argument("prompt", nargs="?", help="optional first message")
    chat.add_argument("--session", help="session id to use")
    chat.add_argument("--resume", action="store_true", help="resume the given session")
    chat.add_argument("--voice", choices=["none", "tts"], help="speak replies")

    ask = sub.add_parser("ask", help="one-shot question")
    ask.add_argument("question", nargs="+")
    ask.add_argument("--voice", choices=["none", "tts"])

    live = sub.add_parser("live", help="realtime voice conversation (Gemini Live)")
    live.add_argument("--voice-name", dest="voice_name", help="Gemini voice, e.g. Puck, Charon")
    live.add_argument("--no-tools", action="store_true", help="conversation only, no tools")

    voice = sub.add_parser("voice", help="push-to-talk voice loop")
    voice.add_argument("--seconds", type=float, default=5.0, help="record length per turn")
    voice.add_argument("--stt-provider", default="gemini", choices=["gemini", "openai_compat"])
    voice.add_argument("--no-speak", action="store_true", help="don't speak the reply")
    voice.add_argument("--listen-only", action="store_true", help="print the transcript and exit")

    say = sub.add_parser("say", help="text to speech")
    say.add_argument("text", nargs="?")
    say.add_argument("--rate", type=int, default=1)
    say.add_argument("--pitch", type=int, default=1)

    tools_cmd = sub.add_parser("tools", help="list tools")
    tools_cmd.add_argument("--json", action="store_true")
    tools_cmd.add_argument("--verbose", "-v", action="store_true")

    prov = sub.add_parser("providers", help="list providers")
    prov.add_argument("--models", action="store_true", help="list models for the provider")

    conf = sub.add_parser("config", help="inspect or edit configuration")
    conf.add_argument("action", nargs="?", default="show", choices=["show", "path", "set"])
    conf.add_argument("key", nargs="?")
    conf.add_argument("value", nargs="?")

    doc = sub.add_parser("doctor", help="diagnostics")
    doc.add_argument("--online", action="store_true", help="also probe the network/providers")

    sub.add_parser("sessions", help="list saved conversations")
    sub.add_parser("plugins", help="show the plugin directory")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "plain", False):
        os.environ["JARVIS_PLAIN"] = "1"

    config = load_config(args.config)

    handlers = {
        "chat": cmd_chat,
        "ask": cmd_ask,
        "live": cmd_live,
        "voice": cmd_voice,
        "say": cmd_say,
        "tools": cmd_tools,
        "providers": cmd_providers,
        "config": cmd_config,
        "doctor": cmd_doctor,
        "sessions": cmd_sessions,
        "plugins": cmd_plugins,
    }

    if args.command is None:
        args.session = getattr(args, "session", None)
        args.resume = False
        args.prompt = None
        args.voice = None
        args.no_tools = False
        return cmd_chat(args, config)

    if args.command == "ask":
        args.question = " ".join(args.question)
    if args.command not in handlers:
        parser.print_help()
        return 2
    try:
        return handlers[args.command](args, config)
    except ProviderError as exc:
        ui.error(exc.friendly())
        return 2
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
