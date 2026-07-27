#!/usr/bin/env bash
# Installs the voice stack on a Runpod Pod.
#
# WHERE THINGS GO, and why -- this split was measured, not guessed:
#
#   venv        -> CONTAINER DISK (/opt/voice-venv)
#     Installing it on the network volume took over 90 minutes and had not
#     finished; pip writes tens of thousands of small files and MooseFS is
#     ~15x slower at that than the local overlay. Local disk does it in
#     minutes, and torch imports at runtime are fast too.
#
#   pip cache   -> /workspace/.cache/pip
#   ollama      -> /workspace/ollama-models
#   HF cache    -> /workspace/.cache/huggingface
#     These are big sequential DOWNLOADS, which the network volume handles
#     fine. Keeping them there is what makes a rebuild cheap: the container
#     disk is wiped on every Pod stop, so re-running this script afterwards
#     reinstalls from cache in minutes with nothing re-downloaded.
#
# So: after a Pod stop, just run this script again.
set -euo pipefail

WS=/workspace
SRC="$WS/voice-ai"
VENV=/opt/voice-venv          # container disk, deliberately
LOG="$WS/voice-install.log"

export OLLAMA_MODELS="$WS/ollama-models"
export HF_HOME="$WS/.cache/huggingface"
export PIP_CACHE_DIR="$WS/.cache/pip"

mkdir -p "$OLLAMA_MODELS" "$HF_HOME" "$PIP_CACHE_DIR" "$WS/logs"
exec > >(tee -a "$LOG") 2>&1
say() { printf '\n=== %s | %s ===\n' "$(date +%H:%M:%S)" "$1"; }

say "venv at $VENV (container disk)"
[ -d "$VENV" ] || python3 -m venv "$VENV"
PY="$VENV/bin/python"
"$PY" -m pip install --quiet --upgrade pip

say "base deps"
"$PY" -m pip install --quiet -r "$SRC/requirements.txt"

say "torch + whisper + chatterbox"
"$PY" -m pip install torch torchaudio faster-whisper chatterbox-tts 2>&1 | tail -5

# chatterbox-tts pins torch back (2.8 -> 2.6) but leaves the cu128 torchvision
# in place; the mismatch aborts at import with "operator torchvision::nms does
# not exist". Nothing here imports torchvision, so drop it rather than hunting
# for a matching build.
if "$PY" -c "import torchvision" 2>/dev/null; then
  say "removing mismatched torchvision"
  "$PY" -m pip uninstall -y --quiet torchvision || true
fi

say "torch check"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

say "ollama"
# The binary sits on the container disk (reinstalls in seconds); the models
# are what must persist, hence OLLAMA_MODELS on /workspace.
command -v ollama >/dev/null || curl -fsSL https://ollama.com/install.sh | sh
if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  OLLAMA_HOST=0.0.0.0:11434 OLLAMA_MODELS="$OLLAMA_MODELS" \
    nohup ollama serve > "$WS/logs/ollama.log" 2>&1 &
  for _ in $(seq 1 60); do
    curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break
    sleep 2
  done
fi
curl -s http://127.0.0.1:11434/api/tags | grep -q 'qwen2.5:7b' || ollama pull qwen2.5:7b

say "DONE - deps and model in place"
