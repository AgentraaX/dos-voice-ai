# Rumi Voice-AI — Production Deployment Status

_Last updated: 2026-07-25_

## Done (code-complete, verified where possible without a GPU)

### Service (this repo)
- FastAPI service (`server.py`) with `/health`, `/stt`, `/tts` — bearer-token auth
- STT: faster-whisper (`large-v3-turbo`) wrapper (`stt.py`)
- TTS: pluggable — **Chatterbox** (Resemble AI, default) and **Sesame CSM** (alternate), common `synthesize()` interface
- LLM: **Qwen2.5-7B-Instruct** served via vLLM (OpenAI-compatible API)
- `MOCK_MODE` — full STT→LLM→TTS pipeline runs with zero GPU/model weights; used to verify request/response wiring end-to-end
- `PRELOAD` — eager-loads Whisper + TTS engine at container startup (so the *first* real user request isn't also a multi-GB model load)
- `start.sh` — single container launches vLLM + the STT/TTS service together, waits for vLLM's health check before starting the second process, traps SIGTERM to shut both down cleanly
- `Dockerfile` — CUDA 12.4 base image, installs both requirement sets, clones Sesame CSM
- `requirements-gpu.txt` — real `vllm`, `torch`, `torchaudio`, `faster-whisper`, `chatterbox-tts`

### Backend integration (`dos-app-backend`)
- `/v1/console/chat` and `/v1/console/voice` — role-gated, session history, logs every turn as training data
- `model_client.py` — calls Qwen/vLLM first, falls back to Groq if unconfigured
- `voice_client.py` — calls the STT/TTS service (separate env var/port from the LLM — a bug where both shared one URL and would've silently misrouted on the real pod was found and fixed)
- 40/40 backend tests passing against real Postgres

### Frontend (`dos-console-web`)
- Voice composer: mic capture, live waveform reactive to both the user's mic *and* the AI's reply audio, 4-state UI (idle/listening/thinking/speaking)
- Verified against the backend in mock mode

All of the above is pushed to GitHub: `AgentraaX/dos-voice-ai`, `AgentraaX/dos-app-backend`, `AgentraaX/dos-console-web`.

## Left to do (needs a real GPU — none available in the dev sandbox this was built in)

| Item | Why it's still open |
|---|---|
| Deploy this container to a real Runpod GPU pod | Never run outside a GPU-less sandbox — Dockerfile/`start.sh` untested against real hardware |
| Verify real Qwen inference via vLLM | Only exercised via `MOCK_MODE`; real latency/quality/VRAM usage unknown |
| Verify real Whisper STT on actual audio | Same — mock returns a canned transcript |
| Verify real Chatterbox (and Sesame) TTS output | Same — mock returns a canned silent clip |
| Point backend at the real pod | Set `VOICE_AI_BASE_URL` (Qwen, port 8000) and `VOICE_SPEECH_BASE_URL` (STT/TTS, port 8001) to the pod's public address, plus shared `VOICE_AI_TOKEN` |
| Sesame CSM setup | Needs a gated HF model download/token — not tested; Chatterbox is recommended for the first real test instead |
| Latency/load testing under real GPU load | No numbers yet on p50/p95 turn latency, concurrent session capacity |
| Production process supervision beyond `start.sh` | Current script restarts the whole container if either process dies; no finer-grained recovery, no metrics/alerting |
| Networking/TLS for exposing the pod | Currently assumes plain HTTP on Runpod's proxy; no cert/domain setup yet |
| Remaining console v1 scope (deferred earlier) | Feedback thumbs-up/down, tags, scenario picker — not started |

**Bottom line:** all serving code, orchestration, and the full backend/frontend integration are written and unit/mock-verified. What's outstanding is exclusively the *real-GPU* verification pass — nothing left to code before that, just deploy-and-test on Runpod. See `README.md` in this repo for the step-by-step "Tomorrow's Runpod checklist."
