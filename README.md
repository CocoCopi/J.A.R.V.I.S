# Jarvis

A voice-driven, tool-using AI assistant that runs **on your phone**, inside
Termux (or a PRoot distro on Android). It talks to whichever AI backend you
prefer — **NVIDIA NIM** for reasoning and tool use, **Google Gemini** for
realtime voice — and it can actually *do* things: run shell commands, install
packages, read and write files, browse the web, speak, notify, vibrate, take a
photo, remember what you tell it.

```
you › find my biggest photos and tell me what's eating my storage
  ⚙ run_shell(df -h /sdcard)
  ✓ Filesystem  Size Used Avail Use% Mounted on ...  42G  38G  4.0G  91% /sdcard
  ⚙ find_files(*.jpg, /sdcard/DCIM, 50)
  ✓ /sdcard/DCIM/Camera/VID_20260104.jpg  1.2G
  ⚙ run_shell(du -sh /sdcard/DCIM/* | sort -h | tail -5)
  ✓ 9.4G /sdcard/DCIM/Camera
    ...
Your DCIM folder is 31 GB, and Camera alone is 9.4 GB — the biggest single
file is a 1.2 GB video from 4 January. Want me to list what's safe to delete?
```

---

## Features

| | |
|---|---|
| 🧠 **Any AI backend** | NVIDIA NIM, Gemini, or *any* OpenAI-compatible endpoint (Ollama, OpenRouter, Groq, LM Studio, llama.cpp). Adding your own is ~30 lines. |
| 🎙 **Live voice mode** | Full-duplex realtime conversation on the **Gemini Live API** — interrupt it mid-sentence, it hears you. |
| 🗣 **Push-to-talk voice** | Record → transcribe → think → speak, using `termux-api` for mic and speaker. |
| 🛠 **42 built-in tools** | Shell, files, web search/fetch, Android device control, memory notes. |
| 📱 **Made for mobile** | Streaming output, retries for flaky networks, session persistence, config kept in TOML, no daemons, no servers. |
| 🔒 **Safe by default** | Destructive commands require an explicit yes; `yolo` is opt-in; API keys are redacted everywhere. |
| 🔌 **Extensible** | Drop a `.py` file in `~/.config/jarvis/tools/` to add tools, or register a provider class. |

---

## Install (Termux)

```bash
pkg install python git
git clone <your-repo-url> ~/jarvis && cd ~/jarvis
./install.sh
```

`install.sh` installs `python`, `git`, `termux-api`, `sox` and `ffmpeg`, then
`pip install`s Jarvis with Gemini + voice support, and writes a starter config.

Then install the **[Termux:API](https://f-droid.org/packages/com.termux.api/)**
app from F-Droid or the Play Store — that is what gives Jarvis a microphone,
speaker, notifications, clipboard and battery access.

Add at least one API key:

```bash
jarvis config set providers.nvidia_nim.api_key nvapi-xxxxxxxx   # build.nvidia.com
jarvis config set providers.gemini.api_key     AIza-xxxxxxxx    # aistudio.google.com/apikey
jarvis doctor                                                   # sanity check
jarvis
```

Optional, to let Jarvis see your shared storage (photos, downloads, music):

```bash
termux-setup-storage
```

### PRoot (Ubuntu/Debian on Android)

Jarvis detects PRoot and automatically **bridges** to the Termux installation:
it looks for `/data/data/com.termux/files/usr/bin` and, when it finds it, puts
that `bin` and `lib` on the search path of every `termux-api` call it makes. So
TTS, notifications and the microphone usually work from inside PRoot too.

`jarvis doctor` prints the detected prefix and the bridge:

```
  termux prefix:   /data/data/com.termux/files/usr
  proot bridge:    LD_LIBRARY_PATH, PATH
  termux-api ready: True
```

If the bridge cannot work on your device (some ROMs block the path, or the
Android linker refuses the binary), everything else still works — only the
device tools report "unavailable". Running Jarvis from Termux itself is the
guaranteed path for full voice support.

### Desktop / server

```bash
pip install ".[gemini,rich]"
jarvis doctor
```

---

## AI providers

Two providers are wired up for you out of the box, and a third covers
everything else.

### NVIDIA NIM — the default brain

```bash
jarvis config set default_provider nvidia_nim
jarvis config set providers.nvidia_nim.api_key nvapi-...
jarvis config set providers.nvidia_nim.model "meta/llama-3.3-70b-instruct"
jarvis providers --models          # list what your key can actually use
```

* Endpoint: `https://integrate.api.nvidia.com/v1` (override with `--base-url`
  or `NVIDIA_BASE_URL` for a **self-hosted NIM container** — no key needed there).
* Default model: `meta/llama-3.3-70b-instruct`. `"auto"` picks it for you.
* Full tool/function calling, streaming, retries with backoff.

### Google Gemini — chat **and** live voice

```bash
jarvis config set providers.gemini.api_key AIza-...
jarvis config set providers.gemini.model gemini-2.0-flash
jarvis config set providers.gemini.live_model gemini-2.0-flash-live-001
jarvis live
```

One key powers both plain chat and realtime voice mode.

### Anything OpenAI-compatible

Ollama, OpenRouter, Groq, LM Studio, Together, vLLM, llama.cpp's server:

```bash
# local Ollama
jarvis --provider openai_compat \
       --base-url http://127.0.0.1:11434/v1 \
       --model qwen2.5:7b ask "hello"

# or persist it
jarvis config set providers.openai_compat.base_url http://127.0.0.1:11434/v1
jarvis config set providers.openai_compat.model qwen2.5:7b
```

### Write your own provider

Create `jarvis/providers/my_backend.py`:

```python
from jarvis.providers.base import Message, Provider, Response, ToolSpec
from jarvis.providers.registry import register

@register
class MyBackend(Provider):
    name = "my_backend"
    label = "My private model"
    required_env = ("MY_API_KEY",)

    @classmethod
    def default_model(cls) -> str:
        return "my-model-1"

    def chat(self, messages, tools=None, *, temperature=0.7, max_tokens=None) -> Response:
        # call your API here; return Response(text=..., tool_calls=[...])
        ...
```

Import it in `jarvis/providers/__init__.py`, or load it as a plugin at runtime —
see `jarvis/providers/mock.py` for a complete, dependency-free example. That
file is also a working offline demo: `jarvis --provider mock ask "shell: uname -a"`.

---

## Commands

```
jarvis                        interactive chat (default)
jarvis ask "..."              one-shot question (clean output when piped)
jarvis live                   realtime voice conversation (Gemini Live)
jarvis voice                  push-to-talk loop (record → answer → speak)
jarvis say "text"             speak something now
jarvis tools [--json] [-v]    list every tool with its schema
jarvis providers [--models]   list backends, configuration and models
jarvis config [show|path|set] inspect or edit configuration
jarvis sessions               list saved conversations
jarvis doctor [--online]      environment, audio and connectivity diagnostics
jarvis plugins                show the plugin directory
```

Global flags: `--provider`, `--model`, `--live-model`, `--base-url`,
`--temperature`, `--tools a,b,c`, `--yes`, `--no-stream`, `--plain`.

Inside the interactive shell:

```
/help      /tools     /model <id>     /provider <name>
/voice on  /yolo on   /cost          /clear    /session
!df -h     → runs a shell command directly
```

---

## Voice

Three levels, pick what your setup supports:

1. **Spoken replies in chat** — `jarvis --provider nvidia_nim chat --voice tts`
   (or `voice = "tts"` in the config). Long replies are queued and spoken
   sentence-by-sentence while the model is still streaming.
2. **`jarvis voice`** — press Enter, speak for N seconds, Jarvis transcribes
   (`--stt-provider gemini` by default, or any Whisper-compatible endpoint with
   `openai_compat`), answers, and speaks back. Works with slow/limited hardware.
3. **`jarvis live`** — realtime, hands-free, interruptible. Gemini streams audio
   both ways; Jarvis executes tools as the conversation happens.

Audio backends are auto-detected: `sounddevice` if it is installed (lowest
latency), otherwise `termux-microphone-record` for capture and `sox`/`ffplay`
for playback, with a `termux-media-player` fallback. Use headphones in live mode,
otherwise Jarvis hears itself.

---

## Tools

Run `jarvis tools` for the live list. The built-ins:

| Category | Tools |
|---|---|
| **shell** | `run_shell`, `python_exec`, `which`, `command_help` |
| **files** | `read_file`, `write_file`, `list_dir`, `make_dir`, `move_path`, `copy_path`, `delete_path`, `find_files`, `search_text`, `file_info` |
| **web** | `web_fetch`, `web_search`, `open_url` |
| **device** | `say`, `stop_speaking`, `notify_user`, `toast`, `vibrate`, `clipboard_get`, `clipboard_set`, `battery_status`, `get_location`, `wifi_info`, `send_sms`, `make_call`, `take_photo`, `torch`, `device_info` |
| **system** | `current_datetime`, `utc_now`, `system_info`, `get_env`, `list_processes`, `open_app`, `sleep` |
| **memory** | `remember`, `recall`, `forget` |

### Add a tool

Nothing but a decorated function — the JSON Schema is generated from the
signature and docstring:

```python
from jarvis.tools import tool

@tool(category="media", confirm=True)
def play_music(path: str, volume: int = 70) -> str:
    """Play an audio file on the device.

    Args:
        path: File to play.
        volume: 0-100.
    """
    ...
```

Save it as `~/.config/jarvis/tools/music.py` and it is picked up automatically
(`jarvis plugins` lists what loaded).

### Safety

`run_shell`, `python_exec`, `write_file`, `delete_path`, `send_sms`, `make_call`
and anything tagged `confirm=True` ask before running. Commands matching
destructive patterns (`rm -rf /`, `dd of=/dev/block/…`, fork bombs, `mkfs`) are
*always* confirmed, even with `--yes` unset. Add `deny_tools = ["run_shell"]` to
the config to remove a capability entirely.

---

## Configuration

`jarvis config` prints the effective settings (keys redacted); `jarvis config
path` shows the file. Environment variables win over the file, so you can also
just export:

```bash
export NVIDIA_API_KEY=nvapi-...
export GEMINI_API_KEY=AIza-...
export JARVIS_PROVIDER=gemini
```

```toml
default_provider = "nvidia_nim"
voice = "none"          # none | tts
yolo = false            # auto-approve confirmations (careful)
max_steps = 12          # tool-call rounds per turn
deny_tools = []         # capabilities to remove
confirm_tools = ["run_shell", "python_exec", "write_file", "delete_path"]

[providers.nvidia_nim]
api_key = "nvapi-..."
model = "meta/llama-3.3-70b-instruct"

[providers.gemini]
api_key = "AIza-..."
model = "gemini-2.0-flash"
live_model = "gemini-2.0-flash-live-001"

[providers.openai_compat]
base_url = "http://127.0.0.1:11434/v1"
model = "qwen2.5:7b"
```

Sessions and memories live in `~/.local/share/jarvis/` (`sessions/*.json`,
`memory.json`); the config is chmod 600 because it holds keys.

---

## How it fits together

```
jarvis/
├── cli.py            argparse entry point + every subcommand
├── agent.py          the think → call tool → observe → answer loop
├── config.py         TOML + environment resolution
├── session.py        conversation persistence
├── termux.py         termux-api wrappers (mic, TTS, notifications, ...)
├── prompts.py        system prompt + runtime description
├── ui.py             colours, banners, streaming printer
├── providers/
│   ├── base.py       Message / ToolCall / Response / Provider
│   ├── registry.py   name → class registry (add yours here)
│   ├── nvidia_nim.py NVIDIA NIM
│   ├── gemini.py     Gemini (text + tools + Live)
│   ├── openai_compat.py   Ollama/OpenRouter/Groq/LM Studio/...
│   └── mock.py       offline demo + template
├── tools/
│   ├── base.py       @tool decorator + schema generation
│   ├── shell.py  files.py  web.py  termux.py  system.py  memory.py
│   └── registry_loader.py   built-ins + user plugins
└── voice/
    ├── audio.py      PCM capture/playback + resampling
    ├── tts.py        speech cleaning, sentence queueing
    ├── stt.py        Gemini / Whisper transcription
    └── live.py       Gemini Live realtime session with tool calling
```

The agent only knows about `Message`, `ToolCall` and `Response` — that is why a
new AI backend never touches the tool loop, and a new tool never touches a
provider.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `termux-microphone-record not found` | `pkg install termux-api` **and** install the Termux:API app. |
| Voice commands hang, then time out | The Termux:API app is missing or denied microphone permission (Android Settings → Apps → Termux:API). |
| No sound in live mode | Install `sox` (`pkg install sox`) or `ffmpeg`. Buffered fallback needs `termux-media-player`. |
| `401`/`403` from a provider | `jarvis config set providers.<name>.api_key ...` then `jarvis doctor --online`. |
| `404` from a provider | Wrong model id — `jarvis providers --models` lists valid ones. |
| Live mode says the model is unsupported | Try another `--live-model`; Live availability varies per key/region. |
| Jarvis hears itself and loops | Use headphones, or `--silence-ms` via `LiveOptions` in `jarvis live`. |
| Device tools say "unavailable" inside PRoot | Jarvis bridges to `/data/data/com.termux/files/usr/bin` automatically; check `jarvis doctor`. If that path is blocked on your ROM, run Jarvis from Termux itself. |

---

## Tests

```bash
python -m unittest discover -s tests -t . -v
```

46 tests cover schema generation, tool execution, the agent loop (tool calls,
declines, step limits), config round-trips, session persistence, PCM/WAV
handling, Gemini history conversion, the Live connect config, the PRoot→Termux
binary bridge, and the CLI —
including full **HTTP integration tests** against `tests/fake_nim.py`, a fake
OpenAI-compatible server that exercises streaming SSE, streamed tool-call
fragments and error handling without any API key.

## License

MIT — see [LICENSE](LICENSE).
