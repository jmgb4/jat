"""VRAM-aware advice for Ollama steps (avoid GPU->RAM spill on Windows/4090).

Goal: keep the pipeline GPU-first by preventing oversized models / contexts from
overflowing VRAM and spilling into host RAM (which looks like "brutally slow").

This module uses best-effort heuristics:
- model "size" is parsed from `ollama list` output (disk size, approximate)
- GPU VRAM totals come from `nvidia-smi`
- KV-cache overhead is approximated as ~512MB per 4096 ctx
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Optional

from app.config import Settings


_SIZE_RE = re.compile(r"^\s*(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>KB|MB|GB|TB)\s*$", re.IGNORECASE)


def _size_to_mb(size_str: str) -> Optional[float]:
    m = _SIZE_RE.match((size_str or "").strip())
    if not m:
        return None
    num = float(m.group("num"))
    unit = m.group("unit").upper()
    mul = {"KB": 1 / 1024, "MB": 1.0, "GB": 1024.0, "TB": 1024.0 * 1024.0}.get(unit)
    if mul is None:
        return None
    return num * mul


def _kv_overhead_mb(num_ctx: int) -> int:
    """
    Heuristic KV-cache overhead.

    Roughly: 4096 ctx ~ 512MB, 8192 ~ 1024MB, 16384 ~ 2048MB.
    This is conservative enough to reduce spill probability without needing model internals.
    """

    n = max(512, int(512 * (max(1024, int(num_ctx)) / 4096)))
    return min(n, 4096)


def _query_vram_mb() -> tuple[Optional[int], Optional[int]]:
    """
    Return (total_mb, free_mb) from nvidia-smi. Values are in MiB from nvidia-smi.
    """

    try:
        p = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2.5,
        )
        if p.returncode != 0:
            return None, None
        line = (p.stdout or "").strip().splitlines()[0]
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2:
            return None, None
        total = int(float(parts[0]))
        free = int(float(parts[1]))
        return total, free
    except Exception:
        return None, None


def _parse_ollama_list(stdout: str) -> dict[str, float]:
    """
    Parse `ollama list` output into {model_name: size_mb}.

    Expected columns: NAME ID SIZE MODIFIED
    Example line: qwen3:14b  bdbd...  9.3 GB  2 hours ago
    """

    sizes: dict[str, float] = {}
    lines = (stdout or "").splitlines()
    for ln in lines:
        ln = (ln or "").strip()
        if not ln:
            continue
        if ln.lower().startswith("name"):
            continue
        parts = ln.split()
        if len(parts) < 4:
            continue
        name = parts[0].strip()
        size_val = parts[2].strip()
        size_unit = parts[3].strip()
        mb = _size_to_mb(f"{size_val} {size_unit}")
        if name and mb is not None:
            sizes[name] = mb
    return sizes


@dataclass
class OllamaCatalog:
    """Cached view of local Ollama model sizes."""

    sizes_mb: dict[str, float]
    fetched_at: float


_CATALOG: Optional[OllamaCatalog] = None


def get_ollama_sizes_mb(ttl_sec: float = 60.0) -> dict[str, float]:
    global _CATALOG
    now = time.monotonic()
    if _CATALOG and (now - _CATALOG.fetched_at) < ttl_sec:
        return dict(_CATALOG.sizes_mb)
    try:
        p = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=3.0)
        sizes = _parse_ollama_list(p.stdout if p.returncode == 0 else "")
    except Exception:
        sizes = {}
    _CATALOG = OllamaCatalog(sizes_mb=sizes, fetched_at=now)
    return dict(sizes)


def advise_ollama_step(
    *,
    model_id: str,
    step_params: dict[str, Any],
    config: Settings,
) -> tuple[str, dict[str, Any], list[str]]:
    """
    Return (advised_model_id, advised_params, advisory_logs).

    - If the requested model+ctx is likely to spill VRAM, reduce ctx (down to 4096)
      or substitute to a fallback model.
    """

    logs: list[str] = []
    mid = (model_id or "").strip()
    if not mid.startswith("ollama:"):
        return mid, step_params, logs

    requested_name = mid.split(":", 1)[1].strip()
    if not requested_name:
        return mid, step_params, logs

    total_mb, free_mb = _query_vram_mb()
    if total_mb is None:
        # Can't read VRAM; do nothing.
        return mid, step_params, logs

    headroom_mb = int(getattr(config, "OLLAMA_VRAM_HEADROOM_MB", 2500))
    min_ctx = int(getattr(config, "OLLAMA_MIN_NUM_CTX", 4096))

    sizes = get_ollama_sizes_mb(ttl_sec=60.0)
    model_mb = sizes.get(requested_name)
    # If unknown size, still apply a free-VRAM heuristic.
    ctx_current = int(step_params.get("num_ctx") or getattr(config, "OLLAMA_NUM_CTX", 8192))

    def fits(model_size_mb: Optional[float], ctx: int) -> bool:
        kv_mb = _kv_overhead_mb(ctx)
        if model_size_mb is None:
            # If we don't know model size, just ensure free VRAM isn't critically low.
            return (free_mb or 0) > (headroom_mb + kv_mb)
        return (model_size_mb + kv_mb + headroom_mb) < total_mb

    # Step 1: reduce context if needed
    ctx_candidates = []
    for c in (ctx_current, 8192, 6144, 4096):
        c = int(c)
        if c < min_ctx:
            continue
        if c not in ctx_candidates:
            ctx_candidates.append(c)

    chosen_ctx = None
    for c in ctx_candidates:
        if fits(model_mb, c):
            chosen_ctx = c
            break

    if chosen_ctx is not None and chosen_ctx != ctx_current:
        new_params = dict(step_params)
        new_params["num_ctx"] = chosen_ctx
        logs.append(f"VRAM guardrail: reducing ctx {ctx_current}→{chosen_ctx} for ollama:{requested_name} to avoid spill")
        return mid, new_params, logs

    if chosen_ctx is not None:
        # Fits as-is
        return mid, step_params, logs

    # Step 2: substitute fallback model if the requested model is too large
    fallback_mid = str(getattr(config, "OLLAMA_FALLBACK_MODEL_ID", "ollama:qwen3:14b") or "").strip()
    if not fallback_mid.startswith("ollama:"):
        fallback_mid = f"ollama:{fallback_mid}" if fallback_mid else ""
    if fallback_mid and fallback_mid != mid:
        fb_name = fallback_mid.split(":", 1)[1]
        fb_mb = sizes.get(fb_name)
        # Try ctx candidates for fallback too
        for c in ctx_candidates:
            if fits(fb_mb, c):
                new_params = dict(step_params)
                new_params["num_ctx"] = c
                logs.append(
                    f"VRAM guardrail: substituting {mid}→{fallback_mid} (ctx={c}) to stay GPU-first"
                )
                return fallback_mid, new_params, logs

        # As a last resort, force minimal ctx for fallback and hope for the best.
        new_params = dict(step_params)
        new_params["num_ctx"] = max(min_ctx, 4096)
        logs.append(
            f"VRAM guardrail: substituting {mid}→{fallback_mid} (ctx={new_params['num_ctx']}) to reduce spill risk"
        )
        return fallback_mid, new_params, logs

    # No fallback configured; do nothing.
    logs.append(f"VRAM guardrail: {mid} may overflow VRAM (no fallback configured)")
    return mid, step_params, logs

