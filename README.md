# voice-ai

STT + TTS serving for the Rumi console's voice feature (and, later, the
in-app driver voice agent). Runs on the Runpod RTX 4090 as **one container,
two processes** (via `start.sh`): vLLM serving **Qwen** as the LLM, and this
FastAPI service for STT/TTS.

| Piece | What | Served by | Port |
|-------|------|-----------|------|
| LLM | Qwen2.5-7B-Instruct | **vLLM's own server** (`vllm serve`, launched by `start.sh`) -- no custom code in this repo | 8000 |
| STT | Whisper (`faster-whisper`) | `POST /stt` in `server.py` | 8001 |
| TTS | Chatterbox (default) or Sesame CSM | `POST /tts` in `server.py` | 8001 |

The backend (`backend/dos-app-backend`) talks to both **independently, on
two separate env vars** -- this matters, they are two different
processes/ports even though they run on the same pod:
- `model_client.py` → `VOICE_AI_BASE_URL` (port 8000, vLLM, LLM replies)
- `voice_client.py` → `VOICE_SPEECH_BASE_URL` (port 8001, this service, STT/TTS)

Both send `VOICE_AI_TOKEN` as the Bearer token (shared secret, same pod).

## Running locally in mock mode (no GPU)

This is what actually runs in CI / any machine without a GPU -- and what's
been verified end-to-end in this repo already. `/stt` returns a fixed
transcript, `/tts` returns a tiny silent WAV -- enough to exercise the
whole console voice pipeline (record → upload → transcript → LLM reply →
synthesized audio → playback → saved interaction) with zero model weights.
`start.sh` is **not** used here -- run `server.py` directly:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
MOCK_MODE=true uvicorn server:app --port 8001 --reload
```

```bash
curl http://localhost:8001/health
curl -X POST http://localhost:8001/tts -H "Content-Type: application/json" -d '{"text":"hi"}' -o out.wav
curl -X POST http://localhost:8001/stt -F "audio=@out.wav" -F "language=en"
```

Then point the backend at it: `VOICE_SPEECH_BASE_URL=http://localhost:8001`
(leave `VOICE_AI_BASE_URL`/`VOICE_AI_TOKEN` unset — text/voice chat both
fall back to Groq for the LLM half in this mode).

## Tomorrow's Runpod checklist (real GPU, real models)

1. **Provision** an RTX 4090 pod (persistent volume for model weights).
2. **Build the image** (installs `ffmpeg`, `requirements-gpu.txt` — which
   now includes `vllm` itself — and clones Sesame CSM into `csm_repo/`):
   ```bash
   docker build -t voice-ai .
   ```
3. **One `docker run`** — starts both vLLM and the STT/TTS service via
   `start.sh`, which waits for vLLM's `/health` before the speech service
   comes up, so you never hit a half-ready pod:
   ```bash
   docker run --gpus all -p 8000:8000 -p 8001:8001 \
     -e MOCK_MODE=false \
     -e TTS_ENGINE=chatterbox \
     -e VOICE_AI_TOKEN=<a long random secret> \
     voice-ai
   ```
   Recommend **`TTS_ENGINE=chatterbox`** for this first test — fewer
   moving parts than Sesame's gated-model setup (below). Switch to
   `TTS_ENGINE=sesame` once chatterbox is confirmed working end-to-end.
   `PRELOAD=true` is the default — Whisper and the TTS engine load at
   startup, not on the first real request; watch the container logs for
   `Preload complete -- ready for real traffic.` before testing.
4. **Verify both processes came up:**
   ```bash
   curl http://<pod-host>:8000/health   # vLLM
   curl http://<pod-host>:8001/health   # this service -- mock_mode should read false
   ```
5. **Point the backend at both** (two separate env vars — see above):
   ```
   VOICE_AI_BASE_URL=http://<pod-host>:8000
   VOICE_SPEECH_BASE_URL=http://<pod-host>:8001
   VOICE_AI_TOKEN=<same secret as step 3>
   ```
6. **One real end-to-end curl** against the backend (not this service
   directly) to confirm the whole chain — STT → Qwen reply → TTS → saved —
   works with real models, the same shape already verified in mock mode:
   ```bash
   curl -X POST http://<backend-host>/v1/console/voice \
     -H "Authorization: Bearer <rumi's access token>" \
     -F "audio=@sample.wav" -F "language=en"
   ```
7. **Sesame CSM setup** (only when you switch to `TTS_ENGINE=sesame`): its
   backbone (Llama-3.2-1B) is gated on Hugging Face — `huggingface-cli
   login` with an account that accepted Meta's license, *before* startup
   (PRELOAD will otherwise fail loudly at boot rather than on first use,
   which is the point — fail fast, not mid-demo).

If 24 GB gets tight running vLLM + Whisper + Chatterbox/Sesame together,
lower `VLLM_GPU_MEM_FRACTION` (default `0.55`) or switch to an AWQ/GPTQ-
quantized Qwen variant — see `docs/phase-2/05-infra-runpod.md` §3.

## Endpoints

- `GET /health` -- `{status, mock_mode, tts_engine, whisper_model}`, unauthenticated. (vLLM has its own `/health` on port 8000.)
- `POST /stt` -- multipart `audio` file + `language` form field → `{text, language}`. Bearer-auth if `VOICE_AI_TOKEN` is set.
- `POST /tts` -- JSON `{text}` → `audio/wav` bytes. Bearer-auth if `VOICE_AI_TOKEN` is set.

## Config reference

| Env var | Default | Meaning |
|---|---|---|
| `MOCK_MODE` | `false` | `true` = canned STT/TTS, no GPU/model deps imported at all. |
| `PRELOAD` | `true` | Eager-load Whisper + the active TTS engine at startup instead of on first request. Ignored under `MOCK_MODE`. |
| `TTS_ENGINE` | `chatterbox` | `chatterbox` or `sesame` — see "Swapping TTS engines" below. |
| `WHISPER_MODEL` | `large-v3-turbo` | Any faster-whisper model name. |
| `VOICE_AI_TOKEN` | unset | Shared bearer secret for `/stt`, `/tts`, and vLLM's `--api-key`. Unset = unauthenticated (local mock-mode only). |
| `QWEN_MODEL` (start.sh) | `Qwen/Qwen2.5-7B-Instruct` | Passed to `vllm serve`. |
| `VLLM_GPU_MEM_FRACTION` (start.sh) | `0.55` | vLLM's `--gpu-memory-utilization`, leaving headroom for STT/TTS. |

## Swapping TTS engines

`TTS_ENGINE=chatterbox` (default) or `TTS_ENGINE=sesame`, selected in
`tts/__init__.py`. Both sit behind the same `synthesize(text: str) ->
bytes` / `preload()` interface (`tts/chatterbox_engine.py`,
`tts/sesame_engine.py`) so adding a third engine later is one new file.
