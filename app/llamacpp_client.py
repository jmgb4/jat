"""llama.cpp runtime client for GGUF models."""

from __future__ import annotations

import asyncio
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from app.ai_client import AIClient, _estimate_tokens
from app.config import Settings


def _register_cuda_dll_dirs() -> None:
    """
    On Windows, add torch's bundled CUDA 12 DLL directory to the DLL search path
    so that llama.dll (cu124 build) can find cublas64_12.dll / cudart64_12.dll.

    This must be called before any llama_cpp import.
    The CUDA DLLs ship inside the torch package when installing torch+cu124.
    """
    if sys.platform != "win32":
        return
    try:
        import torch as _torch  # noqa: PLC0415
        torch_lib = Path(_torch.__file__).parent / "lib"
        if torch_lib.is_dir():
            # os.add_dll_directory is available on Python 3.8+ Windows
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(torch_lib))
    except Exception:
        pass  # torch may not be installed; llama.cpp may still work without it


_register_cuda_dll_dirs()


@dataclass(frozen=True)
class LlamaCppModelSpec:
    gguf_path: str


_LLAMA_CACHE: dict[str, "Llama"] = {}


def _cache_key(gguf_path: str, n_ctx: int, n_threads: int, n_gpu_layers: int) -> str:
    base = {"p": str(Path(gguf_path)), "n_ctx": n_ctx, "n_threads": n_threads, "n_gpu_layers": n_gpu_layers}
    return json.dumps(base, sort_keys=True)


def _load_llama(path: str, config: Settings, overrides: Optional[dict[str, Any]] = None) -> "Llama":
    # Lazy import to keep optional dependency isolated.
    try:
        from llama_cpp import Llama
        from llama_cpp import llama_cpp as _lc
    except Exception as e:
        # On Windows, a partially-installed wheel or missing VC++ runtime can surface as a DLL load failure.
        msg = (
            "Failed to import llama-cpp-python runtime.\n\n"
            "Most likely cause on Windows: missing 'llama.dll' in your venv or a missing dependency.\n"
            "Expected path:\n"
            "  venv\\Lib\\site-packages\\llama_cpp\\lib\\llama.dll\n\n"
            "Fix:\n"
            "  1) From the project root, run: .\\setup.ps1\n"
            "  2) Or reinstall the package:\n"
            "     .\\venv\\Scripts\\pip.exe uninstall -y llama-cpp-python\n"
            "     .\\venv\\Scripts\\pip.exe install -r requirements.txt\n\n"
            "If it worked previously, check whether antivirus quarantined the DLL or the install was interrupted."
        )
        raise RuntimeError(msg) from e

    # GPU-only enforcement (fail fast if user asked for GPU).
    require_gpu = bool(getattr(config, "LLAMACPP_REQUIRE_GPU", False))
    if require_gpu:
        supports = False
        try:
            fn = getattr(_lc, "llama_supports_gpu_offload", None)
            supports = bool(fn() if callable(fn) else False)
        except Exception:
            supports = False
        if not supports:
            raise RuntimeError(
                "GGUF runtime is configured as GPU-only but llama.cpp reports no GPU offload support.\n\n"
                "What this means:\n"
                "- You have a CPU-only llama-cpp-python build installed.\n"
                "- Large GGUF models will be extremely slow on CPU.\n\n"
                "Fix:\n"
                "1) Reinstall CUDA-enabled llama-cpp-python (cu121) from requirements.txt\n"
                "2) Ensure CUDA 12.x runtime DLLs are available (common missing DLLs: cublas64_12.dll, cublasLt64_12.dll, cudart64_12.dll)\n"
                "3) Re-run .\\setup.ps1, then verify with:\n"
                "   .\\venv\\Scripts\\python.exe -c \"from llama_cpp import llama_cpp as lc; print(lc.llama_supports_gpu_offload())\""
            )

    gguf = str(Path(path))
    overrides = overrides or {}
    n_ctx = int(overrides.get("n_ctx") or getattr(config, "LLAMACPP_N_CTX", 8192))
    n_threads = int(overrides.get("n_threads") or getattr(config, "LLAMACPP_N_THREADS", 0) or 0)
    n_gpu_layers = int(overrides.get("n_gpu_layers") if overrides.get("n_gpu_layers") is not None else getattr(config, "LLAMACPP_N_GPU_LAYERS", -1))

    key = _cache_key(gguf, n_ctx=n_ctx, n_threads=n_threads, n_gpu_layers=n_gpu_layers)
    if key in _LLAMA_CACHE:
        return _LLAMA_CACHE[key]

    llama = Llama(
        model_path=gguf,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
    _LLAMA_CACHE[key] = llama
    return llama


class LlamaCppClient(AIClient):
    def __init__(self, config: Settings, gguf_path: str, params: Optional[dict[str, Any]] = None):
        self._config = config
        self._gguf_path = gguf_path
        self._params = params or {}

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        on_token: Optional[Callable[[int], None]] = None,
        on_complete: Optional[Callable[[int, float], None]] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Use llama.cpp chat completion with streaming.
        """
        loop = asyncio.get_running_loop()
        merged = {**(self._params or {}), **(params or {})}
        llama = _load_llama(self._gguf_path, self._config, overrides=merged)
        factor = getattr(self._config, "TOKEN_ESTIMATION_FACTOR", 4)
        temp = float(merged.get("temperature") if merged.get("temperature") is not None else getattr(self._config, "LLAMACPP_TEMPERATURE", 0.3))
        top_p = merged.get("top_p")
        top_k = merged.get("top_k")
        repeat_penalty = merged.get("repeat_penalty")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.monotonic()
        parts: list[str] = []
        total_tokens = 0

        def _run_blocking() -> tuple[str, int, float]:
            nonlocal total_tokens
            kwargs: dict[str, Any] = {
                "messages": messages,
                "temperature": temp,
                "max_tokens": max_tokens or 8192,
                "stream": True,
            }
            if top_p is not None:
                kwargs["top_p"] = float(top_p)
            if top_k is not None:
                kwargs["top_k"] = int(top_k)
            if repeat_penalty is not None:
                kwargs["repeat_penalty"] = float(repeat_penalty)

            stream = llama.create_chat_completion(**kwargs)
            for chunk in stream:
                try:
                    delta = chunk["choices"][0].get("delta") or {}
                    content = delta.get("content") or ""
                except Exception:
                    content = ""
                if not content:
                    continue
                parts.append(content)
                total_tokens += _estimate_tokens(content, factor)
                if on_token:
                    loop.call_soon_threadsafe(on_token, total_tokens)
            elapsed = time.monotonic() - start
            return ("".join(parts).strip(), total_tokens, elapsed)

        text, toks, elapsed = await asyncio.to_thread(_run_blocking)
        if on_complete:
            on_complete(toks, elapsed)
        return text


def evict_all_llama_cache() -> None:
    """
    Best-effort eviction to reduce long-running memory growth.
    Clearing cache only drops Python references; GPU/CPU memory release depends on
    llama-cpp-python + GC timing.
    """
    _LLAMA_CACHE.clear()
    gc.collect()

