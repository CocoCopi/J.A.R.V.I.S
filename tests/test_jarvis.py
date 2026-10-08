"""Jarvis test suite (stdlib unittest only - no pytest needed).

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jarvis import providers, session as session_store  # noqa: E402
from jarvis.agent import Agent, AgentCallbacks, Confirmer  # noqa: E402
from jarvis.config import Config, ProviderConfig, load_config, save_config  # noqa: E402
from jarvis.providers.base import Message, Response, ToolCall, Usage  # noqa: E402
from jarvis.providers.openai_compat import _StreamAccumulator  # noqa: E402
from jarvis.tools import all_tools, get, load_builtin_tools, run_tool  # noqa: E402
from jarvis.tools.base import build_parameters, tool  # noqa: E402
from jarvis.tools.shell import looks_dangerous  # noqa: E402
from jarvis.voice import audio  # noqa: E402
from jarvis.voice.tts import clean_for_speech, split_sentences  # noqa: E402

load_builtin_tools()


def test_config() -> Config:
    return load_config(use_dotenv=False, env={})


# -- tools ------------------------------------------------------------------
class ToolSystemTests(unittest.TestCase):
    def test_schema_generation(self):
        def sample(command: str, retries: int = 3, verbose: bool = False, tags=None) -> str:
            """Do a thing.

            Args:
                command: What to run.
                retries: How many times.
                verbose: Chatty output.
                tags: Optional labels.
            """
            return command

        schema = build_parameters(sample)
        self.assertEqual(schema["type"], "object")
        self.assertEqual(schema["required"], ["command"])
        self.assertEqual(schema["properties"]["command"]["type"], "string")
        self.assertEqual(schema["properties"]["command"]["description"], "What to run.")
        self.assertEqual(schema["properties"]["retries"]["type"], "integer")
        self.assertEqual(schema["properties"]["retries"]["default"], 3)
        self.assertEqual(schema["properties"]["verbose"]["type"], "boolean")
        self.assertNotIn("default", schema["properties"]["tags"])

    def test_registered_tools_have_valid_schemas(self):
        entries = all_tools()
        self.assertGreaterEqual(len(entries), 30)
        for entry in entries:
            spec = entry.spec().to_dict()
            self.assertTrue(spec["name"])
            self.assertTrue(spec["description"])
            self.assertEqual(spec["parameters"]["type"], "object")

    def test_run_tool_success_and_failure(self):
        outcome = run_tool("make_dir", {"path": tempfile.mkdtemp() + "/nested/deep"})
        self.assertTrue(outcome.ok, outcome.output)

        missing = run_tool("read_file", {"path": "/definitely/not/here"})
        self.assertFalse(missing.ok)
        self.assertIn("no such file", missing.output)

        unknown = run_tool("nope_not_a_tool", {})
        self.assertFalse(unknown.ok)

        bad = run_tool("find_files", {"nope": 1})
        self.assertFalse(bad.ok)
        self.assertIn("bad arguments", bad.output)

    def test_unknown_arguments_are_dropped(self):
        outcome = run_tool("current_datetime", {"offset_days": 0, "bogus": "x"})
        self.assertTrue(outcome.ok, outcome.output)

    def test_custom_tool_registration(self):
        @tool(name="unit_test_tool", category="tests")
        def unit_test_tool(value: str) -> str:
            """Echo a value.

            Args:
                value: Anything.
            """
            return "got %s" % value

        self.assertIsNotNone(get("unit_test_tool"))
        self.assertEqual(run_tool("unit_test_tool", {"value": "x"}).output, "got x")

    def test_tool_returning_dict_is_json(self):
        @tool(name="unit_test_dict", category="tests")
        def unit_test_dict() -> dict:
            """Return a dict."""
            return {"a": 1}

        result = run_tool("unit_test_dict", {})
        self.assertEqual(json.loads(result.output), {"a": 1})

    def test_dangerous_command_detection(self):
        self.assertIsNotNone(looks_dangerous("rm -rf /"))
        self.assertIsNotNone(looks_dangerous("dd if=/dev/zero of=/dev/block/mmcblk0"))
        self.assertIsNone(looks_dangerous("ls -la"))


# -- providers ---------------------------------------------------------------
class ProviderTests(unittest.TestCase):
    def test_registry(self):
        names = providers.available()
        for expected in ("nvidia_nim", "gemini", "openai_compat", "mock"):
            self.assertIn(expected, names)
        self.assertIs(providers.get_class("nim"), providers.get_class("nvidia_nim"))

    def test_mock_provider_tool_call(self):
        provider = providers.get_class("mock")(ProviderConfig(name="mock"))
        specs = [t.spec() for t in all_tools() if t.name == "run_shell"]
        response = provider.chat([Message.user("shell: echo hi")], specs)
        self.assertTrue(response.has_tool_calls)
        self.assertEqual(response.tool_calls[0].name, "run_shell")
        self.assertEqual(response.tool_calls[0].arguments["command"], "echo hi")

    def test_openai_stream_accumulator_reassembles_tool_calls(self):
        acc = _StreamAccumulator()
        chunks = [
            {"choices": [{"delta": {"role": "assistant", "content": "Hel"}}]},
            {"choices": [{"delta": {"content": "lo"}}]},
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "id": "call_1", "function": {"name": "run_shell"}}
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"comm'}}]}}
                ]
            },
            {
                "choices": [
                    {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'and": "ls"}'}}]}}
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            },
        ]
        streamed = []
        for chunk in chunks:
            streamed.extend(acc.feed(chunk))
        response = acc.finish("test-model")
        self.assertEqual("".join(streamed), "Hello")
        self.assertEqual(response.text, "Hello")
        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.tool_calls[0].name, "run_shell")
        self.assertEqual(response.tool_calls[0].arguments, {"command": "ls"})
        self.assertEqual(response.usage.total_tokens, 7)

    def test_nvidia_nim_defaults_and_auto_model(self):
        cls = providers.get_class("nvidia_nim")
        provider = cls(ProviderConfig(name="nvidia_nim", model="auto"))
        self.assertEqual(provider.model, "meta/llama-3.3-70b-instruct")
        self.assertTrue(provider.base_url.endswith("/v1"))
        provider2 = cls(ProviderConfig(name="nvidia_nim", model="my/custom-model"))
        self.assertEqual(provider2.model, "my/custom-model")

    def test_gemini_history_conversion(self):
        from jarvis.providers.gemini import GeminiProvider

        instance = object.__new__(GeminiProvider)  # skip __init__ (needs a key)
        messages = [
            Message.system("be nice"),
            Message.user("hi"),
            Message.assistant("", [ToolCall(id="1", name="run_shell", arguments={"command": "ls"})]),
            Message.tool("1", "run_shell", '{"ok": true}'),
        ]
        contents, system = instance._build_contents(messages)
        self.assertEqual(system, "be nice")
        self.assertEqual(contents[0].role, "user")
        self.assertEqual(contents[1].role, "model")
        self.assertTrue(contents[1].parts[0].function_call.name == "run_shell")
        self.assertEqual(contents[2].role, "user")
        self.assertEqual(contents[2].parts[0].function_response.name, "run_shell")

    def test_gemini_tool_declarations(self):
        from jarvis.providers.gemini import GeminiProvider

        instance = object.__new__(GeminiProvider)
        tools = instance._build_tools(get("run_shell").spec() and [get("run_shell").spec()])
        self.assertEqual(len(tools), 1)
        declaration = tools[0].function_declarations[0]
        self.assertEqual(declaration.name, "run_shell")
        self.assertEqual(declaration.parameters_json_schema["required"], ["command"])


# -- agent loop --------------------------------------------------------------
class ScriptedProvider(providers.Provider):
    """A provider that follows a fixed script of responses."""

    name = "scripted_for_tests"
    label = "test"
    supports_tools = True

    def __init__(self, script):
        self.script = list(script)
        self.model = "scripted"
        self.last_response = None
        self.seen = []

    @classmethod
    def default_model(cls):
        return "scripted"

    def chat(self, messages, tools=None, *, temperature=0.7, max_tokens=None):
        self.seen.append([m.role for m in messages])
        return self.script.pop(0)

    def stream(self, messages, tools=None, *, temperature=0.7, max_tokens=None):
        response = self.chat(messages, tools, temperature=temperature, max_tokens=max_tokens)
        self.last_response = response
        if response.text:
            yield response.text
        return


class AgentLoopTests(unittest.TestCase):
    def _agent(self, script, config=None, **kwargs):
        config = config or test_config()
        provider = ScriptedProvider(script)
        events = []
        agent = Agent(
            provider,
            config,
            callbacks=AgentCallbacks(
                on_text=lambda c: events.append(("text", c)),
                on_tool_start=lambda c: events.append(("start", c.name)),
                on_tool_end=lambda c, r: events.append(("end", c.name, r.ok)),
            ),
            extra_context="",
            **kwargs,
        )
        return agent, provider, events

    def test_tool_call_then_answer(self):
        script = [
            Response(text="", tool_calls=[ToolCall(id="1", name="current_datetime", arguments={})]),
            Response(text="It is now.", usage=Usage(total_tokens=5)),
        ]
        agent, provider, events = self._agent(script, Config(yolo=True, max_steps=5))
        result = agent.run("what time is it")
        self.assertEqual(result.text, "It is now.")
        self.assertEqual(result.steps, 2)
        self.assertEqual(result.tool_calls[0].name, "current_datetime")
        self.assertIn(("start", "current_datetime"), events)
        self.assertIn(("end", "current_datetime", True), events)
        # The tool result must be in the history handed back to the model.
        self.assertEqual(provider.seen[-1][-1], "tool")

    def test_declined_tool_is_reported_to_model(self):
        script = [
            Response(text="", tool_calls=[ToolCall(id="1", name="run_shell", arguments={"command": "echo hi"})]),
            Response(text="Okay, I won't."),
        ]
        agent, _provider, events = self._agent(script)  # no confirmer hook -> deny
        result = agent.run("run echo hi")
        self.assertEqual(result.text, "Okay, I won't.")
        self.assertIn(("end", "run_shell", False), events)
        tool_message = [m for m in agent.history if m.role == "tool"][0]
        self.assertIn("declined", tool_message.content)

    def test_denied_tools_never_reach_the_model(self):
        config = Config(yolo=True, deny_tools=("run_shell",))
        agent, _provider, _events = self._agent([Response(text="done")], config)
        names = [spec.name for spec in agent.tool_specs()]
        self.assertNotIn("run_shell", names)

    def test_max_steps_stops_runaway_loops(self):
        looping = [
            Response(text="", tool_calls=[ToolCall(id="1", name="current_datetime", arguments={})])
            for _ in range(10)
        ]
        agent, _provider, _events = self._agent(looping, Config(yolo=True, max_steps=3))
        result = agent.run("loop forever")
        self.assertLessEqual(result.steps, 3)

    def test_streamed_text_is_forwarded_and_history_kept(self):
        # The mock provider streams word by word and records last_response.
        config = test_config()
        provider = providers.get_class("mock")(ProviderConfig(name="mock"))
        agent = Agent(provider, config, extra_context="")
        result = agent.run("hello")
        self.assertIn("Mock reply", result.text)
        self.assertEqual(agent.history[-1].role, "assistant")

    def test_confirmer_hook_can_approve(self):
        approved = []
        config = Config(max_steps=3)
        provider = ScriptedProvider(
            [
                Response(text="", tool_calls=[ToolCall(id="1", name="run_shell", arguments={"command": "echo ok"})]),
                Response(text="done"),
            ]
        )
        confirmer = Confirmer(
            config, lambda name, reason, args: approved.append(name) or True
        )
        agent = Agent(provider, config, confirmer=confirmer, extra_context="")
        agent.run("run echo ok")
        self.assertEqual(approved, ["run_shell"])
        tool_message = [m for m in agent.history if m.role == "tool"][0]
        self.assertIn("ok", tool_message.content)


# -- config / session --------------------------------------------------------
class ConfigTests(unittest.TestCase):
    def test_environment_overrides(self):
        env = {
            "NVIDIA_API_KEY": "nvapi-test",
            "NVIDIA_MODEL": "meta/llama-3.1-8b-instruct",
            "JARVIS_PROVIDER": "gemini",
            "JARVIS_YOLO": "true",
        }
        config = load_config(use_dotenv=False, env=env)
        self.assertEqual(config.default_provider, "gemini")
        self.assertTrue(config.yolo)
        self.assertEqual(config.provider("nvidia_nim").api_key, "nvapi-test")
        self.assertEqual(config.provider("nvidia_nim").model, "meta/llama-3.1-8b-instruct")

    def test_save_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            config = Config(
                default_provider="nvidia_nim",
                yolo=False,
                providers={
                    "nvidia_nim": ProviderConfig(name="nvidia_nim", api_key="nvapi-abc", model="m/1"),
                    "gemini": ProviderConfig(name="gemini", api_key="AIza-xyz", live_model="gemini-2.0-flash-live-001"),
                },
            )
            save_config(config, path)
            text = path.read_text()
            self.assertIn('api_key = "nvapi-abc"', text)
            reloaded = load_config(path, use_dotenv=False, env={})
            self.assertEqual(reloaded.provider("nvidia_nim").api_key, "nvapi-abc")
            self.assertEqual(reloaded.provider("gemini").live_model, "gemini-2.0-flash-live-001")
            mode = oct(path.stat().st_mode & 0o777)
            self.assertEqual(mode, "0o600")

    def test_provider_lookup_defaults(self):
        config = test_config()
        missing = config.provider("does_not_exist")
        self.assertEqual(missing.name, "does_not_exist")
        self.assertIsNone(missing.api_key)


class SessionTests(unittest.TestCase):
    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["JARVIS_DATA_DIR"] = tmp
            try:
                messages = [
                    Message.system("sys"),
                    Message.user("hi"),
                    Message.assistant("", [ToolCall(id="1", name="run_shell", arguments={"command": "ls"})]),
                    Message.tool("1", "run_shell", "ok"),
                ]
                session_store.save("t1", messages, meta={"model": "m"})
                loaded = session_store.load("t1")
                self.assertEqual([m.role for m in loaded], ["system", "user", "assistant", "tool"])
                self.assertEqual(loaded[2].tool_calls[0].arguments, {"command": "ls"})
                self.assertEqual(session_store.latest(), "t1")
                self.assertEqual(session_store.list_sessions()[0]["turns"], 4)
                self.assertEqual(session_store.load("nope"), [])
            finally:
                os.environ.pop("JARVIS_DATA_DIR", None)


# -- audio -------------------------------------------------------------------
class AudioTests(unittest.TestCase):
    def test_wav_roundtrip(self):
        pcm = b"\x00\x01" * 160
        wav = audio.pcm_to_wav(pcm, rate=16000)
        self.assertEqual(audio.strip_wav_header(wav), pcm)
        rate, channels, width = audio.pcm_params(wav)
        self.assertEqual((rate, channels), (16000, 1))
        self.assertEqual(width, 2)

    def test_resample_pcm_length(self):
        pcm = b"\x10\x00" * 480  # 480 samples @ 48k = 10 ms
        out = audio.resample_pcm(pcm, 48000, 16000)
        self.assertEqual(len(out), 160 * 2)

    def test_to_mono_16k_passthrough(self):
        pcm = b"\x00\x00" * 100
        self.assertEqual(audio.to_mono_16k(pcm, 16000, 2), pcm)

    def test_speech_cleaning(self):
        messy = "## Title\n\nUse `jarvis` at https://example.com\n- item one\n- item two\n"
        spoken = clean_for_speech(messy)
        self.assertNotIn("http", spoken)
        self.assertNotIn("#", spoken)
        self.assertIn("item one", spoken)
        self.assertTrue(split_sentences("One. Two! Three?"))
        self.assertEqual(clean_for_speech(""), "")


# -- live session ------------------------------------------------------------
class LiveTests(unittest.TestCase):
    def test_connect_config_is_accepted_by_the_sdk(self):
        from jarvis.voice.live import LiveOptions, LiveSession

        class DummyProvider:
            live_model = "gemini-2.0-flash-live-001"

        session = LiveSession(DummyProvider(), test_config(), options=LiveOptions(voice="Puck"))
        from google.genai import types

        config = session._connect_config(types)
        self.assertEqual(config.response_modalities, ["AUDIO"])
        self.assertTrue(config.speech_config.voice_config.prebuilt_voice_config.voice_name == "Puck")
        names = [d.name for d in config.tools[0].function_declarations]
        self.assertIn("run_shell", names)
        self.assertIn("say", names)

    def test_live_tool_bridge_respects_deny_list(self):
        from jarvis.voice.live import LiveToolBridge

        config = test_config()
        config.yolo = True
        config.deny_tools = ("run_shell",)
        bridge = LiveToolBridge(config)
        names = [spec.name for spec in bridge.specs()]
        self.assertNotIn("run_shell", names)


# -- CLI ---------------------------------------------------------------------
class CliTests(unittest.TestCase):
    def _run(self, *args, timeout=120):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT)
        env["JARVIS_PLAIN"] = "1"
        return subprocess.run(
            [sys.executable, "-m", "jarvis", *args],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def test_version_and_help(self):
        self.assertIn("jarvis", self._run("--version").stdout)
        self.assertIn("usage:", self._run("--help").stdout)

    def test_tools_json_is_valid(self):
        completed = self._run("tools", "--json")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        names = {entry["name"] for entry in payload}
        self.assertIn("run_shell", names)
        self.assertIn("web_search", names)

    def test_providers_lists_every_backend(self):
        completed = self._run("providers")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for name in ("nvidia_nim", "gemini", "mock"):
            self.assertIn(name, completed.stdout)

    def test_ask_pipes_clean_output(self):
        completed = self._run("--provider", "mock", "ask", "hello")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Mock reply", completed.stdout)

    def test_ask_requires_confirmation_for_shell(self):
        # Without --yes the tool must be refused, and the answer must say so.
        completed = self._run("--provider", "mock", "ask", "shell: echo nope")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("declined", completed.stdout.lower())

    def test_ask_runs_shell_with_yes(self):
        completed = self._run("--provider", "mock", "--yes", "ask", "shell: echo cli-ok")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("cli-ok", completed.stdout)

    def test_config_path_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            env = dict(os.environ, JARVIS_CONFIG=str(path), PYTHONPATH=str(ROOT))
            subprocess.run(
                [sys.executable, "-m", "jarvis", "config", "set", "providers.nvidia_nim.api_key", "nvapi-x"],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
                check=True,
            )
            self.assertIn("nvapi-x", path.read_text())
            shown = subprocess.run(
                [sys.executable, "-m", "jarvis", "config"],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
                check=True,
            )
            self.assertIn("nvidia_nim", shown.stdout)
            self.assertNotIn("nvapi-x", shown.stdout)  # keys are redacted

    def test_voice_without_a_microphone_fails_cleanly(self):
        try:
            import sounddevice  # noqa: F401
            self.skipTest("sounddevice installed; a real microphone may block this test")
        except ImportError:
            pass
        completed = self._run("voice", "--listen-only", "--seconds", "1")
        self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
        self.assertIn("microphone", completed.stdout + completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)

    def test_doctor_runs(self):
        completed = self._run("doctor")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("jarvis doctor", completed.stdout)


# -- Termux bridge -----------------------------------------------------------
class TermuxBridgeTests(unittest.TestCase):
    """The PRoot -> Termux binary bridge must resolve and degrade cleanly."""

    def setUp(self):
        from jarvis import termux

        self.termux = termux
        self.original = termux.TERMUX_PREFIXES
        self.tmp = tempfile.TemporaryDirectory()
        bin_dir = Path(self.tmp.name) / "bin"
        bin_dir.mkdir(parents=True)
        helper = bin_dir / "termux-fake-helper"
        helper.write_text("#!/bin/sh\necho bridged\n")
        helper.chmod(0o755)
        termux.TERMUX_PREFIXES = (self.tmp.name,)

    def tearDown(self):
        self.termux.TERMUX_PREFIXES = self.original
        self.tmp.cleanup()

    def test_prefix_and_binary_are_discovered(self):
        self.assertEqual(self.termux.termux_prefix_on_disk(), self.tmp.name)
        self.assertTrue(self.termux.has_command("termux-fake-helper"))
        self.assertEqual(
            self.termux.find_command("termux-fake-helper"),
            os.path.join(self.tmp.name, "bin", "termux-fake-helper"),
        )
        self.assertIsNone(self.termux.find_command("termux-does-not-exist"))

    def test_run_termux_executes_a_bridged_binary(self):
        ok, output = self.termux.run_termux(["termux-fake-helper"], timeout=10)
        self.assertTrue(ok, output)
        self.assertIn("bridged", output)

    def test_bridge_env_prefixes_path(self):
        env = self.termux.bridge_env()
        # On a real Termux install there is nothing to bridge; under PRoot the
        # Termux bin directory is prepended.
        if env:
            self.assertTrue(env["PATH"].startswith(os.path.join(self.tmp.name, "bin")))

    def test_missing_command_reports_a_hint(self):
        ok, output = self.termux.run_termux(["termux-not-here"], timeout=5)
        self.assertFalse(ok)
        self.assertIn("not found", output)


# -- HTTP integration (real requests, fake server) ---------------------------
class HttpIntegrationTests(unittest.TestCase):
    """Drive the real HTTP client against a fake NIM endpoint."""

    @classmethod
    def setUpClass(cls):
        from tests.fake_nim import start_server

        cls.server, cls.port, cls.thread = start_server(0)
        cls.base_url = "http://127.0.0.1:%d/v1" % cls.port

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _run(self, *args, timeout=120):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT)
        env["JARVIS_PLAIN"] = "1"
        env["NVIDIA_API_KEY"] = "nvapi-fake-for-tests"
        return subprocess.run(
            [sys.executable, "-m", "jarvis", "--provider", "nvidia_nim", "--base-url", self.base_url, *args],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def test_streaming_tool_call_over_http(self):
        completed = self._run("--yes", "ask", "shell: echo http-round-trip")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("http-round-trip", completed.stdout)
        self.assertIn("Command finished", completed.stdout)

    def test_non_streaming_chat_over_http(self):
        completed = self._run("--no-stream", "ask", "hello over the wire")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Fake NIM says: hello over the wire", completed.stdout)

    def test_model_listing_over_http(self):
        completed = self._run("providers", "--models")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("meta/llama-3.3-70b-instruct", completed.stdout)

    def test_unauthorized_is_reported_clearly(self):
        env = dict(os.environ)
        env.update({"PYTHONPATH": str(ROOT), "JARVIS_PLAIN": "1"})
        env.pop("NVIDIA_API_KEY", None)
        completed = subprocess.run(
            [sys.executable, "-m", "jarvis", "--provider", "nvidia_nim", "--base-url", self.base_url, "ask", "hi"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("401", completed.stderr + completed.stdout)
        self.assertIn("check your API key", completed.stderr + completed.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
