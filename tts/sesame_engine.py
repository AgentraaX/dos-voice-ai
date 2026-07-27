"""Sesame CSM (Conversational Speech Model, SesameAILabs) -- the voice engine
(TTS_ENGINE=sesame), the SeSame half of docs/phase-2's "ChatterLab + SeSame".
Very natural, human-sounding turn-taking speech; conditions generation on a
Llama-3.2-1B backbone + Mimi codec.

Not a pip package -- clone it alongside this file on the GPU box:
    git clone https://github.com/SesameAILabs/csm.git csm_repo
and put csm_repo on PYTHONPATH.

DEPENDENCIES CONFLICT WITH CHATTERBOX. CSM pins torch==2.4.0 and
transformers==4.49.0; chatterbox needs torch 2.6. Installing CSM into the
chatterbox venv silently downgrades torch and breaks chatterbox at startup with
"Model.__init__() missing 1 required positional argument". They need separate
venvs -- see pod-install-sesame.sh.

THE GATED MODEL IS NOT ACTUALLY NEEDED. Upstream loads its tokenizer from
meta-llama/Llama-3.2-1B, which is gated on Hugging Face and 403s without an
approved account. But the gated repo supplies ONLY the tokenizer: the model
architecture is built locally by torchtune (csm_repo/models.py), the weights
come from sesame/csm-1b and the codec from kyutai/mimi, both ungated. The
ungated community mirrors carry a byte-identical tokenizer -- same 128256
vocab, same token ids -- so we point at one instead of waiting on Meta's
approval queue. Override with CSM_TOKENIZER_REPO if that mirror disappears.
"""
from __future__ import annotations

import io
import logging
import os

log = logging.getLogger("dos.voice")

_TOKENIZER_REPO = os.environ.get("CSM_TOKENIZER_REPO", "unsloth/Llama-3.2-1B")

_generator = None


def _patched_tokenizer():
    """Upstream's load_llama3_tokenizer, but from an ungated mirror.

    The post-processor setup is copied deliberately rather than called through:
    CSM depends on the bos/eos template, and silently losing it would produce
    subtly wrong conditioning rather than an error.
    """
    from tokenizers.processors import TemplateProcessing
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_REPO)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[
            (f"{bos}", tokenizer.bos_token_id),
            (f"{eos}", tokenizer.eos_token_id),
        ],
    )
    return tokenizer


def _load_generator():
    global _generator
    if _generator is None:
        # Comes from the cloned csm_repo (see module docstring), not PyPI.
        import generator as csm_generator  # type: ignore[import-not-found]

        # Swap the tokenizer source before load_csm_1b builds the Generator,
        # which calls this by module-level name. Patching here keeps the
        # third-party checkout pristine, so re-cloning it cannot revert the fix.
        csm_generator.load_llama3_tokenizer = _patched_tokenizer
        log.info("CSM tokenizer source: %s", _TOKENIZER_REPO)

        _generator = csm_generator.load_csm_1b(device="cuda")
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
        buf,
        audio.unsqueeze(0).cpu() if audio.dim() == 1 else audio.cpu(),
        generator.sample_rate,
        format="wav",
    )
    return buf.getvalue()
