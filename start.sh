#!/usr/bin/env bash
# Single-pod entrypoint: launches vLLM (serving Qwen -- the LLM) and this
# STT/TTS service as two processes in one container, waits for vLLM to be
# ready before letting traffic hit the speech service, and forwards SIGTERM
# to both on shutdown. This is what makes "one `docker run`" work end to
# end instead of two manual commands (see README.md).
#
# No-op for local MOCK_MODE runs -- those should just run
# `uvicorn server:app` directly (see README.md "Running locally").
set -euo pipefail

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
VLLM_PORT="${VLLM_PORT:-8000}"
SPEECH_PORT="${SPEECH_PORT:-8001}"
# Leaves headroom on the 24GB 4090 for Whisper + Chatterbox/Sesame to also
# load -- see docs/phase-2/05-infra-runpod.md §3 on VRAM sharing. Tune down
# further if OOM, up if you have the room and want more vLLM throughput.
GPU_MEM_FRACTION="${VLLM_GPU_MEM_FRACTION:-0.55}"

vllm_args=(serve "$QWEN_MODEL" --port "$VLLM_PORT" --gpu-memory-utilization "$GPU_MEM_FRACTION")
if [[ -n "${VOICE_AI_TOKEN:-}" ]]; then
  vllm_args+=(--api-key "$VOICE_AI_TOKEN")
fi

echo "[start.sh] launching vLLM: vllm ${vllm_args[*]}"
vllm "${vllm_args[@]}" &
VLLM_PID=$!

cleanup() {
  echo "[start.sh] shutting down..."
  kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup TERM INT

echo "[start.sh] waiting for vLLM on :${VLLM_PORT}/health..."
for _ in $(seq 1 180); do
  if curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; then
    echo "[start.sh] vLLM is ready."
    break
  fi
  sleep 2
done

echo "[start.sh] starting STT/TTS service on :${SPEECH_PORT}"
uvicorn server:app --host 0.0.0.0 --port "$SPEECH_PORT" &
SPEECH_PID=$!

# Exit (and let the container restart) if either process dies -- a silently
# dead vLLM serving no traffic is worse than a container restart during a
# live test.
wait -n "$VLLM_PID" "$SPEECH_PID"
cleanup
kill "$SPEECH_PID" 2>/dev/null || true
