"""Unified AI client with Ollama and DeepSeek providers, streaming and fallback."""

import asyncio
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional

from openai import AsyncOpenAI

from app.config import Settings
from app.model_registry import ModelRegistry

logger = logging.getLogger("jat.ai_client")


def _estimate_tokens(text: str, factor: int = 4) -> int:
    return max(1, len(text) // factor) if text else 0


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models (qwen3, deepseek-r1)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class AIClient(ABC):
    """Abstract base for AI providers."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        on_token: Optional[Callable[[int], None]] = None,
        on_complete: Optional[Callable[[int, float], None]] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> str:
        """Return generated text. When on_token/on_complete are set, may use streaming."""
        pass


class SmokeTestClient(AIClient):
    """
    Fast deterministic client for smoke tests.

    Enabled by `Settings.SMOKE_TEST_MODE`. Avoids network calls and large model loads.
    """

    def __init__(self, config: Settings, label: str):
        self._config = config
        self._label = label or "smoke"

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        on_token: Optional[Callable[[int], None]] = None,
        on_complete: Optional[Callable[[int, float], None]] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> str:
        _ = (system_prompt, max_tokens, params)
        await asyncio.sleep(0)
        text = f"[SMOKE:{self._label}] " + ((prompt or "").strip()[:800])
        factor = getattr(self._config, "TOKEN_ESTIMATION_FACTOR", 4)
        toks = _estimate_tokens(text, factor)
        if on_token:
            on_token(toks)
        if on_complete:
            on_complete(toks, 0.0)
        return text


class OllamaClient(AIClient):
    """Local Ollama API client with optional streaming."""

    def __init__(self, config: Settings, model_override: Optional[str] = None):
        self._config = config
        self._model = model_override or config.OLLAMA_MODEL

    def _get_client(self):
        import ollama
        return ollama.AsyncClient(host=self._config.OLLAMA_URL)

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        on_token: Optional[Callable[[int], None]] = None,
        on_complete: Optional[Callable[[int, float], None]] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> str:
        # NOTE: ollama python client raises ResponseError for HTTP failures.
        # We translate "model not found" into a helpful actionable message.
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        params = params or {}
        # Ollama options: force GPU and set sensible context defaults, allow per-call overrides.
        # num_gpu=999 tells Ollama to offload all layers to GPU (RTX 4090 has plenty of VRAM).
        cfg_num_gpu = int(getattr(self._config, "OLLAMA_NUM_GPU", 999))
        cfg_num_ctx = int(getattr(self._config, "OLLAMA_NUM_CTX", 16384))
        options: dict[str, Any] = {
            "temperature": float(params.get("temperature", 0.3)),
            "num_gpu": int(params.get("num_gpu", cfg_num_gpu)),
            "num_ctx": int(params.get("num_ctx", cfg_num_ctx)),
        }
        for k in ("top_p", "top_k", "repeat_penalty", "seed"):
            if k in params and params.get(k) is not None:
                options[k] = params[k]
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        client = self._get_client()
        factor = getattr(self._config, "TOKEN_ESTIMATION_FACTOR", 4)

        # keep_alive controls how long Ollama keeps the model loaded after the request.
        # "5m" = default (keep for 5 minutes), 0 = unload immediately (frees VRAM/RAM),
        # -1 = keep forever. Pass as a param override from the caller.
        keep_alive = params.get("keep_alive", None)  # None → Ollama uses its own default

        try:
            if on_token is not None or on_complete is not None:
                start = time.monotonic()
                total_tokens = 0
                content_parts = []
                chat_kwargs: dict[str, Any] = dict(
                    model=self._model, messages=messages, options=options, stream=True
                )
                if keep_alive is not None:
                    chat_kwargs["keep_alive"] = keep_alive
                stream = await client.chat(**chat_kwargs)
                async for chunk in stream:
                    part = (chunk.get("message") or {}).get("content") or ""
                    if part:
                        content_parts.append(part)
                        tok = _estimate_tokens(part, factor)
                        total_tokens += tok
                        if on_token:
                            on_token(total_tokens)
                elapsed = time.monotonic() - start
                if on_complete:
                    on_complete(total_tokens, elapsed)
                return _strip_thinking("".join(content_parts))

            chat_kwargs = dict(model=self._model, messages=messages, options=options)
            if keep_alive is not None:
                chat_kwargs["keep_alive"] = keep_alive
            response = await client.chat(**chat_kwargs)
            return _strip_thinking((response.get("message") or {}).get("content", ""))
        except Exception as e:
            # Detect missing model (common: status code 404).
            status_code = getattr(e, "status_code", None)
            msg = str(e or "")
            if status_code == 404 or ("status code: 404" in msg.lower() and "not found" in msg.lower()):
                # Auto-pull the model then retry once instead of immediately failing.
                logger.warning(
                    "Ollama model '%s' not found — attempting auto-pull (this may take a while)...",
                    self._model,
                )
                try:
                    pull_client = self._get_client()
                    async for progress in await pull_client.pull(self._model, stream=True):
                        pull_status = (progress or {}).get("status", "")
                        if pull_status:
                            logger.info("ollama pull %s: %s", self._model, pull_status)
                    logger.info("Auto-pull of '%s' complete — retrying generation.", self._model)
                    # Retry once; if it fails again let the exception propagate naturally.
                    return await self.generate(
                        prompt,
                        system_prompt=system_prompt,
                        max_tokens=max_tokens,
                        on_token=on_token,
                        on_complete=on_complete,
                        params=params,
                    )
                except Exception as pull_err:
                    raise RuntimeError(
                        f"Ollama model '{self._model}' not found and auto-pull failed: {pull_err}\n"
                        "Fix manually:\n"
                        f"  ollama pull {self._model}\n"
                        f"  ollama list\n"
                    ) from pull_err
            raise


class DeepSeekClient(AIClient):
    """DeepSeek API client (OpenAI-compatible) with optional streaming."""

    def __init__(self, config: Settings, model_override: Optional[str] = None):
        self._config = config
        self._model = model_override or getattr(config, "DEFAULT_DEEPSEEK_MODEL", "deepseek-chat")
        timeout = 120.0 if config.QUALITY_OVER_SPEED else 60.0
        self._client = AsyncOpenAI(
            api_key=config.DEEPSEEK_API_KEY or "dummy",
            # Use the OpenAI-compatible base path.
            # (DeepSeek also serves from /, but /v1 is the canonical OpenAI SDK path.)
            base_url="https://api.deepseek.com/v1",
            timeout=timeout,
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        on_token: Optional[Callable[[int], None]] = None,
        on_complete: Optional[Callable[[int, float], None]] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> str:
        if not (self._config.DEEPSEEK_API_KEY or "").strip():
            raise RuntimeError(
                "DeepSeek is not configured (DEEPSEEK_API_KEY is missing). "
                "Add it to your .env file and restart the app."
            )
        if getattr(self._config, "REDACT_PII_FOR_DEEPSEEK", True):
            try:
                from app.privacy import redact_pii

                prompt = redact_pii(prompt)
                if system_prompt:
                    system_prompt = redact_pii(system_prompt)
            except Exception:
                # Never fail a request due to redaction.
                pass

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        params = params or {}
        kwargs = {
            "model": self._model,
            "messages": messages,
            "temperature": float(params.get("temperature", 0.3)),
            "stream": on_token is not None or on_complete is not None,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        if kwargs["stream"]:
            start = time.monotonic()
            total_tokens = 0
            content_parts = []
            stream = await self._client.chat.completions.create(**kwargs)
            async for chunk in stream:
                delta = (chunk.choices[0].delta if chunk.choices else None) or None
                if not delta:
                    continue
                # Standard models: delta.content. Reasoner (deepseek-reasoner): delta.reasoning_content (thinking) and delta.content (answer).
                part = (getattr(delta, "content", None) or "") + (getattr(delta, "reasoning_content", None) or "")
                if part:
                    content_parts.append(part)
                    total_tokens += max(1, _estimate_tokens(part))
                    if on_token:
                        on_token(total_tokens)
            elapsed = time.monotonic() - start
            if on_complete:
                on_complete(total_tokens, elapsed)
            full = "".join(content_parts).strip()
            # Reasoner may include thinking in stream; strip <think> blocks for pipeline (Ollama does the same).
            if "<think>" in full or "</think>" in full:
                full = _strip_thinking(full)
            return full

        response = await self._client.chat.completions.create(**kwargs)
        return (response.choices[0].message.content or "").strip()


def get_ai_client(provider_name: str, config: Settings, model_override: Optional[str] = None) -> AIClient:
    """Return the appropriate client for provider_name (ollama or deepseek)."""
    name = (provider_name or "").lower()
    if getattr(config, "SMOKE_TEST_MODE", False):
        label = f"{name}:{(model_override or '').strip()}" if model_override else name
        return SmokeTestClient(config, label=label)
    if name == "deepseek":
        return DeepSeekClient(config, model_override=model_override)
    return OllamaClient(config, model_override=model_override)


def get_ai_client_for_model_id(model_id: str, config: Settings) -> tuple[AIClient, str]:
    """
    Return AIClient and a display name for a provider-agnostic model_id.

    Supported ids:
    - ollama:<model_name>
    - gguf:<something> (resolved via ModelRegistry)
    - hf:<something> (resolved via ModelRegistry)
    """
    mid = (model_id or "").strip()
    if not mid:
        raise ValueError("model_id is required")

    if getattr(config, "SMOKE_TEST_MODE", False):
        # Keep display name stable for UI/assertions.
        return SmokeTestClient(config, label=mid), mid

    if ":" not in mid:
        # Back-compat: treat as an Ollama model name.
        client = OllamaClient(config, model_override=mid)
        return client, mid

    prefix, rest = mid.split(":", 1)
    prefix = prefix.lower().strip()
    rest = rest.strip()

    if prefix == "ollama":
        client = OllamaClient(config, model_override=rest)
        return client, rest

    if prefix == "gguf":
        reg = ModelRegistry(config)
        for m in reg.list():
            if m.id == mid:
                if not m.gguf_path:
                    break
                from app.llamacpp_client import LlamaCppClient

                return LlamaCppClient(config, gguf_path=m.gguf_path, params=m.llamacpp_params), m.display_name
        raise FileNotFoundError(f"Installed model not found: {mid}")

    if prefix == "hf":
        reg = ModelRegistry(config)
        for m in reg.list():
            if m.id == mid:
                if not m.hf_path:
                    break
                from app.transformers_client import TransformersClient

                return TransformersClient(config, hf_path=m.hf_path, params=m.llamacpp_params), m.display_name
        raise FileNotFoundError(f"Installed model not found: {mid}")

    if prefix == "deepseek":
        # rest is the DeepSeek model name (e.g. "deepseek-chat", "deepseek-reasoner")
        client = DeepSeekClient(config, model_override=rest)
        return client, f"DeepSeek/{rest}"

    raise ValueError(f"Unsupported model_id prefix: {prefix}")


async def call_with_fallback(
    prompt: str,
    system_prompt: Optional[str],
    providers: List[str],
    config: Settings,
    max_tokens: Optional[int] = None,
    on_token: Optional[Callable[[int], None]] = None,
    on_complete: Optional[Callable[[int, float], None]] = None,
    params: Optional[dict[str, Any]] = None,
    ollama_model_override: Optional[str] = None,
    deepseek_model_override: Optional[str] = None,
    on_fallback: Optional[Callable[[str, str], None]] = None,
) -> str:
    """Try each provider in order; return first successful response. Pass callbacks for streaming."""
    errors = []
    for i, name in enumerate(providers):
        try:
            model_override = ollama_model_override if name == "ollama" else deepseek_model_override
            client = get_ai_client(name, config, model_override=model_override)
            out = await client.generate(
                prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                on_token=on_token,
                on_complete=on_complete,
                params=params,
            )
            if out:
                return out
        except Exception as e:
            errors.append(f"{name}: {e}")
            if i + 1 < len(providers) and on_fallback:
                next_name = providers[i + 1]
                try:
                    on_fallback(name, next_name)
                except Exception:
                    pass
            continue
    raise RuntimeError("All providers failed: " + "; ".join(errors))
