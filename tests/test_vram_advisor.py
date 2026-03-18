import types

import pytest


def test_advise_ollama_step_substitutes_when_too_large(monkeypatch):
    from app import vram_advisor

    # Force deterministic VRAM and model catalog.
    monkeypatch.setattr(vram_advisor, "_query_vram_mb", lambda: (24564, 12000))
    monkeypatch.setattr(
        vram_advisor,
        "get_ollama_sizes_mb",
        lambda ttl_sec=60.0: {"glm-4.7-flash-neo-code": 22000.0, "qwen3:14b": 9300.0},
    )

    cfg = types.SimpleNamespace(
        OLLAMA_VRAM_HEADROOM_MB=2500,
        OLLAMA_MIN_NUM_CTX=4096,
        OLLAMA_NUM_CTX=8192,
        OLLAMA_FALLBACK_MODEL_ID="ollama:qwen3:14b",
    )

    mid, params, logs = vram_advisor.advise_ollama_step(
        model_id="ollama:glm-4.7-flash-neo-code",
        step_params={"num_ctx": 8192},
        config=cfg,
    )

    assert mid == "ollama:qwen3:14b"
    assert int(params["num_ctx"]) in (8192, 6144, 4096)
    assert logs and any("substituting" in x.lower() for x in logs)


def test_advise_ollama_step_reduces_ctx_when_possible(monkeypatch):
    from app import vram_advisor

    # Choose a size where our heuristic overflows at 8192 but fits at 4096:
    # 8192: model + 1024 + 2500 > 24564  => model > 21040
    # 4096: model +  512 + 2500 < 24564  => model < 21552
    monkeypatch.setattr(vram_advisor, "_query_vram_mb", lambda: (24564, 12000))
    monkeypatch.setattr(
        vram_advisor,
        "get_ollama_sizes_mb",
        lambda ttl_sec=60.0: {"big:latest": 21200.0},
    )

    cfg = types.SimpleNamespace(
        OLLAMA_VRAM_HEADROOM_MB=2500,
        OLLAMA_MIN_NUM_CTX=4096,
        OLLAMA_NUM_CTX=8192,
        OLLAMA_FALLBACK_MODEL_ID="ollama:qwen3:14b",
    )

    mid, params, logs = vram_advisor.advise_ollama_step(
        model_id="ollama:big:latest",
        step_params={"num_ctx": 8192},
        config=cfg,
    )

    assert mid == "ollama:big:latest"
    assert int(params["num_ctx"]) in (6144, 4096)
    assert logs and any("reducing ctx" in x.lower() for x in logs)

