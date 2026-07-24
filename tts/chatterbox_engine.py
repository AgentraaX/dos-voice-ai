"""Resemble AI's Chatterbox TTS (open-source, MIT) -- the default voice
engine (this is what docs/phase-2 calls "ChatterLab"). Natural, low-latency,
GPU-accelerated speech synthesis.

Install on the GPU box (see ../requirements-gpu.txt):
    pip install chatterbox-tts

Reference: https://github.com/resemble-ai/chatterbox
"""
from __future__ import annotations

import io

_model = None


def _load_model():
    global _model
    if _model is None:
        import torch
        from chatterbox.tts import ChatterboxTTS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = ChatterboxTTS.from_pretrained(device=device)
    return _model


def preload() -> None:
    """Eagerly loads the Chatterbox model -- see server.py's PRELOAD hook."""
    _load_model()


def synthesize(text: str) -> bytes:
    """Returns WAV bytes for `text`, spoken in Chatterbox's default voice."""
    import torchaudio

    model = _load_model()
    wav = model.generate(text)
    buf = io.BytesIO()
    torchaudio.save(buf, wav, model.sr, format="wav")
    return buf.getvalue()
