#!/usr/bin/env bash
# Starts the voice stack on the Pod and prints the URLs the backend needs.
# Run after pod-install.sh (or on its own if the deps are already in place).
#
#   export VOICE_AI_TOKEN=<same value the backend sends>
#   bash pod-start.sh
set -euo pipefail

WS=/workspace
SRC="$WS/voice-ai"
# Each TTS engine has its own venv -- their torch pins conflict. Override both
# together, or let TTS_ENGINE pick the matching one.
TTS_ENGINE=${TTS_ENGINE:-sesame}
if [ "$TTS_ENGINE" = "sesame" ]; then
  VENV=${VENV:-/opt/sesame-venv}
  export PYTHONPATH="$SRC/csm_repo${PYTHONPATH:+:$PYTHONPATH}"
  # CSM's torch.compile path is slow to warm and flaky on first call.
  export NO_TORCH_COMPILE=${NO_TORCH_COMPILE:-1}
else
  VENV=${VENV:-/opt/voice-venv}
fi
export TTS_ENGINE
SPEECH_PORT=${SPEECH_PORT:-8010}
OLLAMA_PORT=${OLLAMA_PORT:-11434}

export OLLAMA_MODELS="$WS/ollama-models"
# Whisper and Chatterbox weights land here; on the volume so they survive a
# Pod stop -- they are ~2GB of download we do not want to repeat.
export HF_HOME="$WS/.cache/huggingface"
export TORCH_HOME="$WS/.cache/torch"

: "${VOICE_AI_TOKEN:?Set VOICE_AI_TOKEN to the value the backend sends}"
mkdir -p "$WS/logs" "$HF_HOME" "$TORCH_HOME"
say() { printf '\n=== %s ===\n' "$1"; }

say "ollama on :$OLLAMA_PORT"
# KEEP_ALIVE=-1 pins Qwen in VRAM. Ollama's 5-minute default unloads it, and
# the next request then pays a ~15s cold load -- against model_client.py's 20s
# timeout. Too close: a slow cold start silently becomes a Groq answer.
if ! curl -sf "http://127.0.0.1:$OLLAMA_PORT/api/tags" >/dev/null 2>&1; then
  OLLAMA_HOST=0.0.0.0:$OLLAMA_PORT OLLAMA_KEEP_ALIVE=-1 \
    nohup ollama serve > "$WS/logs/ollama.log" 2>&1 &
  for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$OLLAMA_PORT/api/tags" >/dev/null 2>&1 && break
    sleep 2
  done
fi
curl -sf "http://127.0.0.1:$OLLAMA_PORT/api/tags" >/dev/null && echo "ollama ok"

say "speech service on :$SPEECH_PORT (TTS_ENGINE=$TTS_ENGINE, venv=$VENV)"
# Bracketed first letter so this pattern cannot match the running script's own
# command line -- pkill matching its own invocation has killed this shell more
# than once.
pkill -f "[u]vicorn server:app" 2>/dev/null || true
sleep 1
cd "$SRC"
MOCK_MODE=false PRELOAD=true VOICE_AI_TOKEN="$VOICE_AI_TOKEN" \
  TTS_ENGINE="$TTS_ENGINE" HF_HOME="$HF_HOME" TORCH_HOME="$TORCH_HOME" \
  setsid nohup "$VENV/bin/python" -m uvicorn server:app \
  --host 0.0.0.0 --port "$SPEECH_PORT" \
  > "$WS/logs/speech.log" 2>&1 < /dev/null &
# PRELOAD loads Whisper large-v3-turbo and Chatterbox before the service
# answers, so the first start downloads ~2GB and takes minutes.
echo "waiting for models..."
for _ in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$SPEECH_PORT/health" >/dev/null 2>&1 && break
  sleep 5
done
curl -sf "http://127.0.0.1:$SPEECH_PORT/health" || {
  echo "speech service did not come up:"; tail -30 "$WS/logs/speech.log"; exit 1; }
echo

say "tunnels"
# Runpod only proxies ports declared at Pod creation, and neither 8010 nor
# 11434 is exposed on this Pod -- so both go through cloudflared. Quick
# tunnels need no account but get a NEW hostname every run, which is exactly
# what silently breaks voice after a restart.
if ! command -v cloudflared >/dev/null; then
  curl -fsSL -o /usr/local/bin/cloudflared \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x /usr/local/bin/cloudflared
fi
pkill -f "cloudflared tunnel --url" 2>/dev/null || true
sleep 1

start_tunnel() {
  local port="$1" name="$2" log="$WS/logs/tunnel-$2.log" url=""
  setsid nohup cloudflared tunnel --url "http://127.0.0.1:$port" \
    > "$log" 2>&1 < /dev/null &
  for _ in $(seq 1 45); do
    url=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$log" 2>/dev/null | head -1 || true)
    [ -n "$url" ] && break
    sleep 2
  done
  [ -n "$url" ] || { echo "no tunnel for $name:" >&2; tail -20 "$log" >&2; return 1; }
  echo "$url"
}

SPEECH_URL=$(start_tunnel "$SPEECH_PORT" speech)
LLM_URL=$(start_tunnel "$OLLAMA_PORT" llm)

cat > "$WS/CURRENT_URLS.txt" <<EOF
VOICE_AI_BASE_URL=$LLM_URL
QWEN_MODEL=qwen2.5:7b
VOICE_SPEECH_BASE_URL=$SPEECH_URL
EOF

say "URLs for backend .env AND the HF Space variables"
cat "$WS/CURRENT_URLS.txt"
echo
echo "health: $(curl -s "http://127.0.0.1:$SPEECH_PORT/health")"
