"""Speech-to-text via faster-whisper. Only imported by server.py when
MOCK_MODE is off -- torch/faster-whisper and the model weights are heavy
and GPU-oriented, so nothing here loads until a real transcription is
actually requested.
"""
from __future__ import annotations

import io
import os

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3-turbo")

_model = None


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _load_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        device = "cuda" if _has_cuda() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        _model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute_type)
    return _model


def preload() -> None:
    """Eagerly loads the Whisper model -- called at process startup (see
    server.py's PRELOAD lifespan hook) so the first real request isn't also
    a multi-GB model load."""
    _load_model()


def transcribe(audio_bytes: bytes, language: str | None = None) -> str:
    """Transcribes a short audio clip. Accepts whatever format ffmpeg can
    decode (faster-whisper/ctranslate2 shells out to it), e.g. the
    webm/opus a browser's MediaRecorder produces."""
    model = _load_model()
    segments, _info = model.transcribe(io.BytesIO(audio_bytes), language=language)
    return " ".join(segment.text.strip() for segment in segments).strip()
