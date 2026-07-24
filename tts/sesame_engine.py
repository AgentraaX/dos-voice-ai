"""Sesame CSM (Conversational Speech Model, SesameAILabs) -- the
alternate voice engine (TTS_ENGINE=sesame), the other half of docs/phase-2's
"ChatterLab + SeSame". Known for very natural, human-sounding turn-taking
speech; conditions generation on a Llama-3.2-1B backbone + Mimi codec.

Not a pip package -- clone it alongside this file on the GPU box (see
../requirements-gpu.txt "Sesame CSM setup"):
    git clone https://github.com/SesameAILabs/csm.git csm_repo
and add csm_repo to PYTHONPATH (the Dockerfile does this). The backbone
model is gated on Hugging Face; `huggingface-cli login` with an account
that accepted Meta's Llama license is required before first use.

Reference: https://github.com/SesameAILabs/csm
"""
from __future__ import annotations

import io

_generator = None


def _load_generator():
    global _generator
    if _generator is None:
        # Comes from the cloned csm_repo (see module docstring), not PyPI.
        from generator import load_csm_1b  # type: ignore[import-not-found]

        _generator = load_csm_1b(device="cuda")
    return _generator


def preload() -> None:
    """Eagerly loads the CSM generator -- see server.py's PRELOAD hook."""
    _load_generator()


def synthesize(text: str) -> bytes:
    """Returns WAV bytes for `text`, spoken in CSM's default voice."""
    import torchaudio

    generator = _load_generator()
    audio = generator.generate(
        text=text,
        speaker=0,
        context=[],
        max_audio_length_ms=20_000,
    )
    buf = io.BytesIO()
    torchaudio.save(
        buf, audio.unsqueeze(0).cpu() if audio.dim() == 1 else audio.cpu(),
        generator.sample_rate, format="wav",
    )
    return buf.getvalue()
