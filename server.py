"""voice-ai: STT + TTS serving for the Rumi console (and, later, the
in-app driver voice agent). Meant to run on the Runpod RTX 4090 alongside
vLLM's own OpenAI-compatible server, which serves the Qwen LLM directly --
there's no custom LLM code here; see README.md for the `vllm serve` command.
The backend (`backend/dos-app-backend`) calls this service's /stt and /tts,
and vLLM's /v1/chat/completions, independently.

MOCK_MODE=true short-circuits both endpoints with canned responses and
never imports faster-whisper/chatterbox/csm/torch, so the whole pipeline
(console-web -> backend -> here -> backend -> console-web) can be exercised
with no GPU and no model weights. See README.md for the real-GPU cutover.
"""
import base64
import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

log = logging.getLogger("voice-ai")
logging.basicConfig(level=logging.INFO)

MOCK_MODE = os.environ.get("MOCK_MODE", "false").lower() == "true"
# Eager-loads Whisper + the active TTS engine at process startup instead of
# on first request -- default on for real runs so the first real /stt or
# /tts call during a live test isn't also a multi-GB model load. Meaningless
# (and skipped) under MOCK_MODE, which has no models to load.
PRELOAD = os.environ.get("PRELOAD", "true").lower() == "true"
# If unset, the service is unauthenticated -- fine for local mock-mode
# testing behind no public exposure; set this once deployed anywhere with
# a reachable address (the backend sends it as VOICE_AI_TOKEN, see
# backend/dos-app-backend/app/services/voice_client.py and model_client.py).
VOICE_AI_TOKEN = os.environ.get("VOICE_AI_TOKEN")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MOCK_MODE and PRELOAD:
        log.info("Preloading STT (%s)...", os.environ.get("WHISPER_MODEL", "large-v3-turbo"))
        from stt import preload as preload_stt

        preload_stt()

        engine_name = os.environ.get("TTS_ENGINE", "sesame")
        log.info("Preloading TTS engine (%s)...", engine_name)
        from tts import preload_active_engine

        preload_active_engine()
        log.info("Preload complete -- ready for real traffic.")
    yield


app = FastAPI(title="voice-ai", lifespan=lifespan)
_bearer = HTTPBearer(auto_error=False)


def require_token(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> None:
    if VOICE_AI_TOKEN is None:
        return
    if creds is None or creds.credentials != VOICE_AI_TOKEN:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing token")

# A ~0.1s silent WAV -- just enough to be a real, playable audio file for
# pipeline testing. Real audio comes from the TTS engine once MOCK_MODE is
# off.
_MOCK_WAV_BYTES = base64.b64decode(
    "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="
)
_MOCK_TRANSCRIPT = "Hello, can you hear me okay?"


class SttResponse(BaseModel):
    text: str
    language: str


class TtsRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mock_mode": MOCK_MODE,
        "tts_engine": os.environ.get("TTS_ENGINE", "sesame"),
        "whisper_model": os.environ.get("WHISPER_MODEL", "large-v3-turbo"),
    }


@app.post("/stt", response_model=SttResponse, dependencies=[Depends(require_token)])
async def stt(audio: UploadFile = File(...), language: str = Form("en")):
    audio_bytes = await audio.read()

    if MOCK_MODE:
        return SttResponse(text=_MOCK_TRANSCRIPT, language=language)

    from stt import transcribe

    text = transcribe(audio_bytes, language=language)
    return SttResponse(text=text, language=language)


@app.post("/tts", dependencies=[Depends(require_token)])
async def tts(payload: TtsRequest):
    if MOCK_MODE:
        audio_bytes = _MOCK_WAV_BYTES
    else:
        from tts import get_engine

        synthesize = get_engine()
        audio_bytes = synthesize(payload.text)

    return Response(content=audio_bytes, media_type="audio/wav")
