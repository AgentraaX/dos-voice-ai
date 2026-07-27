#!/usr/bin/env bash
# Installs Sesame CSM into its OWN venv, separate from the chatterbox one.
#
# They cannot share a venv. CSM pins torch==2.4.0 and transformers==4.49.0;
# chatterbox needs torch 2.6. Installing CSM over chatterbox silently
# downgrades torch and breaks chatterbox at startup with
# "Model.__init__() missing 1 required positional argument: 'config'" --
# which is exactly how production voice went down once already.
#
# Run:  bash pod-install-sesame.sh
# Then: TTS_ENGINE=sesame VENV=/opt/sesame-venv bash pod-start.sh
set -euo pipefail

WS=/workspace
SRC="$WS/voice-ai"
CSM="$SRC/csm_repo"
VENV=/opt/sesame-venv          # container disk: fast to write, fast to import
LOG="$WS/sesame-install.log"

export HF_HOME="$WS/.cache/huggingface"
export PIP_CACHE_DIR="$WS/.cache/pip"

mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$WS/logs"
exec > >(tee -a "$LOG") 2>&1
say() { printf '\n=== %s | %s ===\n' "$(date +%H:%M:%S)" "$1"; }

say "csm_repo"
if [ -d "$CSM/.git" ]; then
  echo "already cloned"
else
  git clone https://github.com/SesameAILabs/csm.git "$CSM"
fi

say "venv at $VENV"
[ -d "$VENV" ] || python3 -m venv "$VENV"
PY="$VENV/bin/python"
"$PY" -m pip install --quiet --upgrade pip
# Same pkg_resources trap as the chatterbox venv: a 3.12 venv ships no
# setuptools, and 81+ dropped pkg_resources entirely.
"$PY" -m pip install --quiet "setuptools<81"

say "voice-ai base deps"
"$PY" -m pip install --quiet -r "$SRC/requirements.txt"

say "CSM deps (pins torch 2.4 -- why this venv is separate)"
"$PY" -m pip install -r "$CSM/requirements.txt" 2>&1 | tail -4

say "faster-whisper (STT runs in this venv too)"
"$PY" -m pip install --quiet faster-whisper

say "versions"
"$PY" - <<'PY'
import torch, transformers
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| transformers", transformers.__version__)
PY

say "tokenizer reachability"
# The gated meta-llama repo supplies only the tokenizer; we use an ungated
# mirror instead. Checked here so a missing mirror fails now, loudly, rather
# than at first speech.
"$PY" - <<'PY'
import os
repo = os.environ.get("CSM_TOKENIZER_REPO", "unsloth/Llama-3.2-1B")
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained(repo)
print(f"tokenizer OK from {repo} (vocab {len(t)})")
PY

say "DONE - start with: TTS_ENGINE=sesame VENV=$VENV bash pod-start.sh"
