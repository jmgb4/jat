"""Local Transformers runtime (optional dependency)."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Callable, Optional

from app.ai_client import AIClient, _estimate_tokens
from app.config import Settings

_HF_CACHE: dict[str, tuple[Any, Any]] = {}


def evict_all_transformers_cache() -> None:
    _HF_CACHE.clear()
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class TransformersClient(AIClient):
    def __init__(self, config: Settings, hf_path: str, params: Optional[dict[str, Any]] = None):
        self._config = config
        self._path = str(Path(hf_path))
        self._params = params or {}

    def _load(self):
        if self._path in _HF_CACHE:
            return _HF_CACHE[self._path]

        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Transformers runtime not installed. Install: pip install torch transformers"
            ) from e

        # Best-effort load. Keep it conservative to avoid surprises.
        trust_remote_code = bool(self._params.get("trust_remote_code", False))
        tok = AutoTokenizer.from_pretrained(self._path, local_files_only=True, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            self._path,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
            device_map="auto",
            torch_dtype="auto",
        )
        model.eval()
        _HF_CACHE[self._path] = (tok, model)
        return tok, model

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        on_token: Optional[Callable[[int], None]] = None,
        on_complete: Optional[Callable[[int, float], None]] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> str:
        params = {**(self._params or {}), **(params or {})}
        factor = getattr(self._config, "TOKEN_ESTIMATION_FACTOR", 4)
        tok, model = self._load()

        # Build messages; use chat template if available.
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        text_in: str
        if hasattr(tok, "apply_chat_template"):
            try:
                text_in = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                text_in = (system_prompt + "\n\n" if system_prompt else "") + prompt
        else:
            text_in = (system_prompt + "\n\n" if system_prompt else "") + prompt

        # Lazy imports for streaming
        try:
            import torch  # type: ignore
            from transformers import TextIteratorStreamer  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Transformers runtime not installed. Install: pip install torch transformers"
            ) from e

        inputs = tok(text_in, return_tensors="pt")
        # device_map may shard; move only if tensor has device attribute
        try:
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        except Exception:
            pass

        temperature = float(params.get("temperature", 0.7))
        top_p = params.get("top_p")
        top_k = params.get("top_k")
        repetition_penalty = params.get("repeat_penalty") or params.get("repetition_penalty")
        max_new_tokens = int(max_tokens or params.get("max_new_tokens") or 8192)

        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature,
            "streamer": streamer,
        }
        if top_p is not None:
            gen_kwargs["top_p"] = float(top_p)
        if top_k is not None:
            gen_kwargs["top_k"] = int(top_k)
        if repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = float(repetition_penalty)

        start = time.monotonic()
        parts: list[str] = []
        total_tokens = 0

        def _run_generate():
            with torch.inference_mode():
                model.generate(**gen_kwargs)

        # Run generation in a worker thread to allow streaming iteration.
        task = asyncio.create_task(asyncio.to_thread(_run_generate))
        try:
            for piece in streamer:
                if not piece:
                    continue
                parts.append(piece)
                total_tokens += _estimate_tokens(piece, factor)
                if on_token:
                    on_token(total_tokens)
                # Yield to event loop occasionally
                await asyncio.sleep(0)
        finally:
            await task

        elapsed = time.monotonic() - start
        if on_complete:
            on_complete(total_tokens, elapsed)
        return "".join(parts).strip()

