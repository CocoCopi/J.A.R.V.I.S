#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Jarvis installer for Termux (and PRoot distros on Android).
#
#   ./install.sh                 # install with Gemini + voice support
#   ./install.sh --no-voice      # no extra audio packages
#   ./install.sh --dev           # editable install (for hacking on Jarvis)
#   ./install.sh --extras "gemini,rich,audio"
#
# It is safe to re-run: every step is idempotent.
# ---------------------------------------------------------------------------
set -euo pipefail

WITH_VOICE=1
EXTRAS="gemini,rich"
DEV=0
SKIP_SYSTEM=0

while [ $# -gt 0 ]; do
  case "$1" in
    --no-voice) WITH_VOICE=0 ;;
    --dev) DEV=1 ;;
    --extras) EXTRAS="$2"; shift ;;
    --skip-system) SKIP_SYSTEM=1 ;;
    -h|--help)
      sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

bold() { printf '\033[1m%s\033[0m\n' "$1"; }
info() { printf '  %s\n' "$1"; }
warn() { printf '\033[33m  %s\033[0m\n' "$1"; }
die()  { printf '\033[31m%s\033[0m\n' "$1" >&2; exit 1; }

TERMUX_PREFIX="${PREFIX:-}"
termux_like() { [ -n "$TERMUX_PREFIX" ] && [ -d "$TERMUX_PREFIX/bin" ]; }
is_proot() { [ -n "${PROOT_TMP_DIR:-}${PROOT_LOADER:-}" ]; }

bold "Jarvis installer"
if termux_like; then
  info "environment: Termux ($TERMUX_PREFIX)"
elif is_proot; then
  info "environment: PRoot distro on Android"
else
  info "environment: $(uname -s) (regular Linux - device tools will be limited)"
fi

# --- 1. system packages -----------------------------------------------------
if [ "$SKIP_SYSTEM" -eq 0 ]; then
  bold "1/4 system packages"
  if termux_like; then
    command -v pkg >/dev/null 2>&1 || die "pkg not found; is this really Termux?"
    pkg update -y >/dev/null 2>&1 || warn "pkg update had problems (continuing)"
    pkg install -y python termux-api git >/dev/null
    info "installed: python, git, termux-api"
    if [ "$WITH_VOICE" -eq 1 ]; then
      # sox gives low-latency playback; ffmpeg is a fallback + decoder
      pkg install -y sox ffmpeg >/dev/null 2>&1 || warn "sox/ffmpeg not installed - live audio will be buffered"
      info "installed: sox, ffmpeg (voice playback)"
      if ! command -v termux-microphone-record >/dev/null 2>&1; then
        warn "termux-microphone-record missing; install the Termux:API app from F-Droid"
      fi
    fi
  elif command -v apt-get >/dev/null 2>&1; then
    as_root=""
    [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1 && as_root="sudo"
    $as_root apt-get update -y >/dev/null 2>&1 || warn "apt-get update failed (continuing)"
    $as_root apt-get install -y python3 python3-pip python3-venv git >/dev/null 2>&1 \
      || warn "could not install python3 packages automatically"
    if [ "$WITH_VOICE" -eq 1 ]; then
      $as_root apt-get install -y sox ffmpeg >/dev/null 2>&1 || warn "sox/ffmpeg not installed"
    fi
    info "installed: python3, pip, git"
  else
    warn "no pkg/apt-get found; skipping system packages"
  fi
else
  bold "1/4 system packages (skipped)"
fi

# --- 2. python --------------------------------------------------------------
bold "2/4 python"
PYTHON_BIN=""
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then PYTHON_BIN="$candidate"; break; fi
done
[ -n "$PYTHON_BIN" ] || die "python is not installed"
PY_VERSION="$($PYTHON_BIN -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
info "using $PYTHON_BIN ($PY_VERSION)"
case "$PY_VERSION" in
  3.[0-8]) die "Python 3.9+ is required (found $PY_VERSION)" ;;
esac

if ! $PYTHON_BIN -m pip --version >/dev/null 2>&1; then
  info "bootstrapping pip"
  $PYTHON_BIN -m ensurepip --upgrade >/dev/null 2>&1 || die "could not bootstrap pip"
fi

PIP_FLAGS=""
if ls /usr/lib/python3*/EXTERNALLY-MANAGED >/dev/null 2>&1; then
  PIP_FLAGS="--break-system-packages"
fi

# --- 3. jarvis --------------------------------------------------------------
bold "3/4 jarvis"
$PYTHON_BIN -m pip install --upgrade pip setuptools wheel $PIP_FLAGS >/dev/null 2>&1 || true
if [ "$DEV" -eq 1 ]; then
  $PYTHON_BIN -m pip install -e ".[$EXTRAS]" $PIP_FLAGS
else
  $PYTHON_BIN -m pip install ".[$EXTRAS]" $PIP_FLAGS
fi
info "installed jarvis $($PYTHON_BIN -c 'import jarvis; print(jarvis.__version__)' 2>/dev/null || echo '?')"
info "command available at: $(command -v jarvis || echo "$TERMUX_PREFIX/bin/jarvis")"

# --- 4. configuration -------------------------------------------------------
bold "4/4 configuration"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/jarvis"
mkdir -p "$CONFIG_DIR"
if [ ! -f "$CONFIG_DIR/config.toml" ]; then
  cat > "$CONFIG_DIR/config.toml" <<'TOML'
# Jarvis configuration. Edit with `jarvis config set <key> <value>`.
default_provider = "nvidia_nim"
voice = "none"
yolo = false
max_steps = 12

# --- NVIDIA NIM (chat + tools) ---------------------------------------------
# Get a key at https://build.nvidia.com  (starts with nvapi-)
[providers.nvidia_nim]
api_key = ""
model = "meta/llama-3.3-70b-instruct"

# --- Google Gemini (chat + LIVE voice mode) --------------------------------
# Get a key at https://aistudio.google.com/apikey
[providers.gemini]
api_key = ""
model = "gemini-2.0-flash"
live_model = "gemini-2.0-flash-live-001"

# --- Any OpenAI-compatible endpoint (Ollama, OpenRouter, Groq, LM Studio) ---
# [providers.openai_compat]
# base_url = "http://127.0.0.1:11434/v1"
# api_key = "ollama"
# model = "qwen2.5:7b"
TOML
  chmod 600 "$CONFIG_DIR/config.toml"
  info "wrote config template: $CONFIG_DIR/config.toml"
else
  info "config already exists: $CONFIG_DIR/config.toml"
fi

cat <<'NEXT'

Next steps
----------
1. Add your API keys (one is enough to start):
      jarvis config set providers.nvidia_nim.api_key nvapi-xxxxxxxx
      jarvis config set providers.gemini.api_key AIza-xxxxxxxx

2. Check everything:
      jarvis doctor

3. Talk to it:
      jarvis                        # interactive chat
      jarvis ask "install ripgrep and find my largest pdf"
      jarvis live                   # realtime voice (needs the Gemini key)

Storage access (optional, lets Jarvis read /sdcard):
      termux-setup-storage

Text to speech uses termux-api. Install the "Termux:API" app from F-Droid,
then:  jarvis say "systems online"
NEXT
