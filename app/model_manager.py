"""Ollama model management (local list, pull, show, delete)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

import aiohttp

from app.config import Settings


@dataclass(frozen=True)
class OllamaModel:
    name: str
    size: Optional[int] = None
    digest: Optional[str] = None
    modified_at: Optional[str] = None
    details: Optional[dict[str, Any]] = None


class ModelManager:
    def __init__(self, config: Settings):
        self._config = config
        self._base = (config.OLLAMA_API_BASE or "").rstrip("/")

    async def list_local_models(self) -> list[dict[str, Any]]:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self._base}/tags") as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data.get("models", []) or []

    async def show_model(self, model: str, verbose: bool = False) -> dict[str, Any]:
        payload = {"model": model, "verbose": bool(verbose)}
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self._base}/show", json=payload) as resp:
                resp.raise_for_status()
                return await resp.json()

    async def delete_model(self, model: str) -> None:
        # Ollama uses DELETE with a JSON body.
        payload = {"model": model}
        async with aiohttp.ClientSession() as session:
            async with session.delete(f"{self._base}/delete", json=payload) as resp:
                resp.raise_for_status()

    async def pull_events(self, model: str) -> AsyncIterator[dict[str, Any]]:
        """
        Stream pull progress events from Ollama.
        Each line is a JSON object (ndjson).
        """
        payload = {"model": model, "stream": True}
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self._base}/pull", json=payload) as resp:
                resp.raise_for_status()
                async for raw in resp.content:
                    if not raw:
                        continue
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except Exception:
                        yield {"status": "error", "error": "Failed to parse Ollama pull event", "raw": line}

