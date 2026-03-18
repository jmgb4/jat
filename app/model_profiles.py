"""
Model tuning profiles — pattern-matched per-model default params.

Profiles are loaded from a JSON file (default: data/model_profiles.json).
Keys are matched as case-insensitive substrings of the model name (after
stripping the provider prefix: ollama:, gguf:, hf:, deepseek:).

Step-level params always take priority over profile params.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("jat.model_profiles")

_CACHE: dict[str, dict] = {}
_CACHE_PATH: str = ""


def load_profiles(path: str) -> dict[str, Any]:
    """Load and cache model profiles from a JSON file. Returns {} on any error."""
    global _CACHE, _CACHE_PATH
    if path == _CACHE_PATH and _CACHE:
        return _CACHE
    try:
        p = Path(path)
        if not p.exists():
            return {}
        raw = p.read_text(encoding="utf-8")
        data = json.loads(raw)
        # Strip comment keys
        profiles = {k: v for k, v in data.items() if not k.startswith("_")}
        _CACHE = profiles
        _CACHE_PATH = path
        return profiles
    except Exception as exc:
        logger.warning("Could not load model profiles from %s: %s", path, exc)
        return {}


def _strip_prefix(model_id: str) -> str:
    """Remove provider prefix (ollama:, gguf:, hf:, deepseek:) from model ID."""
    for prefix in ("ollama:", "gguf:", "hf:", "deepseek:"):
        if model_id.startswith(prefix):
            return model_id[len(prefix):]
    return model_id


def get_params_for_model(model_id: str, profiles: dict[str, Any]) -> dict[str, Any]:
    """
    Return merged params for a model by substring-matching profile keys.

    Multiple keys may match (e.g. "qwen" and "qwen3" both match "qwen3:14b").
    More-specific (longer) keys win over shorter ones.
    """
    if not profiles or not model_id:
        return {}
    name = _strip_prefix(model_id).lower()
    # Collect all matching profile entries, sorted by key length (shorter first).
    # Longer key = more specific = applied last (wins).
    matches = sorted(
        [(k, v) for k, v in profiles.items() if k.lower() in name],
        key=lambda kv: len(kv[0]),
    )
    merged: dict[str, Any] = {}
    for _, params in matches:
        if isinstance(params, dict):
            merged.update(params)
    return merged


def invalidate_cache() -> None:
    """Force reload on next load_profiles() call (useful after the JSON file is edited)."""
    global _CACHE, _CACHE_PATH
    _CACHE = {}
    _CACHE_PATH = ""
