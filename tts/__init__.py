"""Pluggable TTS engines behind one `synthesize(text) -> bytes` interface.
Both real engines are heavy (GPU, multi-GB weights), so they're imported
lazily -- only when `get_engine()`/`preload_active_engine()` is actually
called, which server.py only does when MOCK_MODE is off.
"""
from __future__ import annotations

import os
from typing import Callable

_ENGINES: dict[str, str] = {
    "chatterbox": "chatterbox_engine",
    "sesame": "sesame_engine",
}


def _active_module():
    name = os.environ.get("TTS_ENGINE", "sesame").lower()
    module_name = _ENGINES.get(name)
    if module_name is None:
        raise ValueError(
            f"Unknown TTS_ENGINE: {name!r} (expected one of {sorted(_ENGINES)})"
        )
    import importlib

    return importlib.import_module(f".{module_name}", package=__name__)


def get_engine() -> Callable[[str], bytes]:
    """Returns the active engine's `synthesize(text: str) -> bytes`,
    selected by the TTS_ENGINE env var (default "sesame")."""
    return _active_module().synthesize


def preload_active_engine() -> None:
    """Eagerly loads the active engine's model weights -- see server.py's
    PRELOAD lifespan hook."""
    _active_module().preload()
