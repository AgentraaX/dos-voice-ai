#!/usr/bin/env bash
# Brings the Rumi voice stack up on a fresh or just-migrated Runpod pod.
#
# This is the setup that actually ran live on 2026-07-25 -- Ollama for the
# LLM plus this service running natively -- NOT the Docker + vLLM path in
# README.md. vLLM was never needed: the pod already had Ollama, and a Runpod
# pod is itself a container, so there is nothing to `docker run` inside it.
#
# Migration wipes the container disk (only /workspace survives), so after a
# migrate everything below has to happen again from scratch. Re-running when
# things are already in place is safe -- every step checks first.
#
# Usage, as root on the pod:
#   export VOICE_AI_TOKEN=<same value the backend sends>
#   bash runpod-bringup.sh
#
# Prints the exact env values the backend needs at the end.
set -euo pipefail

WORKDIR=${WORKDIR:-/workspace}
REPO_DIR="$WORKDIR/voice-ai"
SPEECH_PORT=${SPEECH_PORT:-8010}
OLLAMA_PORT=${OLLAMA_PORT:-11434}
QWEN_MODEL=${QWEN_MODEL:-qwen2.5:7b}
LOG_DIR="$WORKDIR/logs"

# No apostrophes in this message: bash parses ${VAR:?word} for quotes, so a
# stray ' makes the whole script a syntax error.
: "${VOICE_AI_TOKEN:?Set VOICE_AI_TOKEN first -- it must match VOICE_AI_TOKEN in the backend env}"

mkdir -p "$LOG_DIR"
say() { printf '\n=== %s ===\n' "$1"; }

say "GPU"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || {
  echo "No GPU visible -- this pod cannot serve Whisper/Chatterbox. Stop here."
  exit 1
}

say "SSH"
# Direct-TCP sshd is not running on a fresh pod, and Runpod's proxy ssh is
# interactive-only (it mangles piped input), so scripted control needs this.
if pgrep -x sshd >/dev/null; then
  echo "sshd already running"
else
  ssh-keygen -A
  /usr/sbin/sshd -p 22
  echo "sshd started on port 22"
fi
mkdir -p /root/.ssh && chmod 700 /root/.ssh
PUBKEY="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICdjEP6EHZMT/xd8qg/erljoFLfKQgpl8HptOl0aD9wW dos-voice-runpod"
grep -qF "AAAAC3NzaC1lZDI1NTE5AAAAICdjEP6EHZMT" /root/.ssh/authorized_keys 2>/dev/null \
  || echo "$PUBKEY" >> /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys

say "System packages"
# ffmpeg is not optional: faster-whisper shells out to it to decode the
# browser's webm/opus recording, so without it every /stt call fails.
if command -v ffmpeg >/dev/null; then
  echo "ffmpeg present"
else
  apt-get update -qq
  apt-get install -y -qq ffmpeg curl git
fi

say "Ollama + $QWEN_MODEL"
command -v ollama >/dev/null || curl -fsSL https://ollama.com/install.sh | sh
if ! curl -sf "http://127.0.0.1:$OLLAMA_PORT/api/tags" >/dev/null 2>&1; then
  # Bind 0.0.0.0, not localhost, or Runpod's proxy cannot reach it.
  OLLAMA_HOST=0.0.0.0:$OLLAMA_PORT nohup ollama serve > "$LOG_DIR/ollama.log" 2>&1 &
  for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$OLLAMA_PORT/api/tags" >/dev/null 2>&1 && break
    sleep 2
  done
fi
# ~4.7GB, and gone after a migrate since Ollama keeps models on the
# container disk (/root/.ollama), not on /workspace.
curl -s "http://127.0.0.1:$OLLAMA_PORT/api/tags" | grep -q "$QWEN_MODEL" \
  || ollama pull "$QWEN_MODEL"
echo "ollama up: $(curl -s "http://127.0.0.1:$OLLAMA_PORT/api/tags" | head -c 160)"

say "voice-ai source"
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull --ff-only
else
  git clone https://github.com/AgentraaX/dos-voice-ai.git "$REPO_DIR"
fi
cd "$REPO_DIR"

say "Python dependencies"
# --break-system-packages: the pod's Python is externally managed (PEP 668)
# and refuses a plain `pip install` without it.
PIP="pip install --quiet --break-system-packages"
$PIP -r requirements.txt
$PIP torch torchaudio faster-whisper chatterbox-tts
# chatterbox-tts pins torch back (2.8 -> 2.6) but leaves the older cu128
# torchvision in place; the mismatch crashes at import with
# "operator torchvision::nms does not exist". Nothing here imports
# torchvision, so removing it is the fix -- do not try to version-match.
if python -c "import torchvision" 2>/dev/null; then
  echo "removing mismatched torchvision"
  pip uninstall -y --break-system-packages torchvision || true
fi

say "Speech service on :$SPEECH_PORT"
pkill -f "uvicorn server:app" 2>/dev/null || true
sleep 1
MOCK_MODE=false PRELOAD=true VOICE_AI_TOKEN="$VOICE_AI_TOKEN" \
  nohup python -m uvicorn server:app --host 0.0.0.0 --port "$SPEECH_PORT" \
  > "$LOG_DIR/voice-ai.log" 2>&1 &
# PRELOAD pulls Whisper large-v3-turbo and Chatterbox into VRAM before the
# service answers, so a cold start genuinely takes minutes.
echo "waiting for models to load (several minutes on a cold pod)..."
for _ in $(seq 1 150); do
  curl -sf "http://127.0.0.1:$SPEECH_PORT/health" >/dev/null 2>&1 && break
  sleep 4
done
curl -sf "http://127.0.0.1:$SPEECH_PORT/health" || {
  echo "service never came up. Last log lines:"
  tail -40 "$LOG_DIR/voice-ai.log"
  exit 1
}
echo

say "cloudflared tunnels"
# Runpod only exposes ports declared when the Pod was created. 8010 almost
# never is, and on a freshly deployed Pod 11434 may not be either -- so
# tunnel BOTH rather than depending on the template being right. Quick
# tunnels need no account, but every hostname is NEW on each run, which is
# exactly why voice silently breaks after a restart.
if ! command -v cloudflared >/dev/null; then
  curl -fsSL -o /usr/local/bin/cloudflared \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x /usr/local/bin/cloudflared
fi
pkill -f "cloudflared tunnel" 2>/dev/null || true
sleep 1

# start_tunnel <local-port> <log-name> -> echoes the public https URL
start_tunnel() {
  local port="$1" name="$2" log="$LOG_DIR/cloudflared-$2.log" url=""
  nohup cloudflared tunnel --url "http://127.0.0.1:$port" > "$log" 2>&1 &
  for _ in $(seq 1 45); do
    url=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$log" | head -1 || true)
    [ -n "$url" ] && break
    sleep 2
  done
  if [ -z "$url" ]; then
    echo "no tunnel URL for $name. Last log lines:" >&2
    tail -20 "$log" >&2
    return 1
  fi
  echo "$url"
}

SPEECH_TUNNEL=$(start_tunnel "$SPEECH_PORT" speech)
LLM_TUNNEL=$(start_tunnel "$OLLAMA_PORT" llm)

POD_ID=$(hostname)
cat <<EOF

=== Put these in the backend .env AND the HF Space variables ===

VOICE_AI_BASE_URL=$LLM_TUNNEL
QWEN_MODEL=$QWEN_MODEL
VOICE_SPEECH_BASE_URL=$SPEECH_TUNNEL
VOICE_AI_TOKEN=(unchanged -- must match on both sides)

If you DID expose port $OLLAMA_PORT on this Pod, prefer the Runpod proxy for
the LLM instead -- it is stable across restarts, unlike the tunnel:
  VOICE_AI_BASE_URL=https://${POD_ID}-${OLLAMA_PORT}.proxy.runpod.net

Both tunnel hostnames are new on every run. They are what silently break
voice after a restart, and the failure looks like a working console that
just never answers.

Sanity-check from your own machine:
  curl $SPEECH_TUNNEL/health
  curl $LLM_TUNNEL/api/tags

/health must report "mock_mode": false -- if it says true, the service came
up without real models and every reply will be the canned mock.
EOF
