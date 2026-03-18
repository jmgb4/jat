"""Provider-agnostic model search + GGUF download/installation."""

from __future__ import annotations

import os
import re
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import aiohttp
from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url

from app.config import Settings


def _safe_dirname(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("\\", "/")
    s = re.sub(r"[^a-z0-9._/-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "model"


@dataclass(frozen=True)
class CatalogResult:
    repo_id: str
    title: str
    gguf_files: list[str]
    downloads: int | None = None
    likes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "title": self.title,
            "gguf_files": self.gguf_files,
            "downloads": self.downloads,
            "likes": self.likes,
        }


class HubManager:
    """
    GGUF-first manager:
    - Search repositories
    - List GGUF files in a repo
    - Download a single GGUF file into HUB_CACHE_DIR
    """

    def __init__(self, config: Settings):
        self._config = config
        self._token = (getattr(config, "HF_TOKEN", "") or "").strip() or None
        self._api = HfApi(token=self._token)

        base = Path(config.HUB_CACHE_DIR)
        self._cache_dir = base if base.is_absolute() else (Path(__file__).parent.parent / base)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _list_gguf_files(self, repo_id: str) -> list[str]:
        # This hits the Hub API and returns all filenames in the repo.
        files = self._api.list_repo_files(repo_id=repo_id, repo_type="model")
        ggufs = [f for f in files if f.lower().endswith(".gguf")]
        return sorted(ggufs)

    def search(self, query: str, limit: int = 15) -> list[CatalogResult]:
        q = (query or "").strip()
        if not q:
            return []

        # Broad search; we will filter to repos that actually contain GGUF files.
        results = []
        it = self._api.list_models(search=q, sort="downloads", direction=-1, limit=limit * 3)
        for m in it:
            repo_id = getattr(m, "modelId", None) or getattr(m, "id", None)
            if not repo_id:
                continue
            try:
                ggufs = self._list_gguf_files(repo_id)
                if not ggufs:
                    continue
                results.append(
                    CatalogResult(
                        repo_id=repo_id,
                        title=repo_id,
                        gguf_files=ggufs[:20],
                        downloads=getattr(m, "downloads", None),
                        likes=getattr(m, "likes", None),
                    )
                )
                if len(results) >= limit:
                    break
            except Exception:
                continue

        return results

    def download_gguf(self, repo_id: str, filename: str) -> Path:
        raise NotImplementedError("Use download_gguf_stream for progress-capable downloads.")

    async def download_gguf_stream(
        self,
        repo_id: str,
        filename: str,
        chunk_size: int = 4 * 1024 * 1024,
        *,
        cancel_check: Optional[callable] = None,
        max_retries: int = 4,
    ):
        """
        Download GGUF into HUB_CACHE_DIR/<repo_id_sanitized>/<filename> and yield progress events.

        Yields dicts compatible with our SSE progress stream:
        - {status, total, completed} while downloading
        - {status: 'success', path: '...'} on completion
        - {status: 'error', error: '...'} on failure
        """
        target_dir = self._cache_dir / _safe_dirname(repo_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / filename
        partial_path = target_dir / (filename + ".partial")

        # Use resolved URL + metadata to get size and handle redirects/CDN.
        url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type="model")
        try:
            meta = get_hf_file_metadata(url=url, token=self._token)
        except Exception as e:
            yield {"status": "error", "error": f"Failed to locate file on Hugging Face: {e}"}
            return

        total = int(getattr(meta, "size", 0) or 0)
        download_url = getattr(meta, "location", None) or url

        def _is_cancelled() -> bool:
            try:
                return bool(cancel_check()) if callable(cancel_check) else False
            except Exception:
                return False

        completed = 0
        if partial_path.exists():
            try:
                completed = int(partial_path.stat().st_size)
            except Exception:
                completed = 0

        # If a full file already exists and matches expected size, short-circuit.
        if target_path.exists() and total and target_path.stat().st_size == total:
            yield {"status": "success", "path": str(target_path)}
            return

        # Ensure we always write to .partial, then rename on success.
        for attempt in range(max_retries):
            if _is_cancelled():
                yield {"status": "cancelled", "total": total, "completed": completed}
                return

            headers: dict[str, str] = {}
            if completed > 0:
                headers["Range"] = f"bytes={completed}-"

            try:
                timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=30)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(download_url, headers=headers) as resp:
                        # 200 = full download, 206 = resume download
                        if resp.status not in (200, 206):
                            text = await resp.text()
                            raise RuntimeError(f"HTTP {resp.status}: {text[:300]}")

                        mode = "ab" if completed > 0 and resp.status == 206 else "wb"
                        if mode == "wb":
                            completed = 0
                        with open(partial_path, mode) as f:
                            async for chunk in resp.content.iter_chunked(chunk_size):
                                if _is_cancelled():
                                    yield {"status": "cancelled", "total": total, "completed": completed}
                                    return
                                if not chunk:
                                    continue
                                f.write(chunk)
                                completed += len(chunk)
                                yield {"status": "downloading", "total": total, "completed": completed}

                # If the Hub reports a size and we didn't reach it, treat as transient failure and retry.
                if total and completed < total:
                    raise RuntimeError(f"Download incomplete (got {completed} of {total} bytes)")

                # Atomic-ish finalize.
                try:
                    if target_path.exists():
                        target_path.unlink()
                except Exception:
                    pass
                partial_path.replace(target_path)
                yield {"status": "success", "path": str(target_path), "total": total, "completed": completed}
                return
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # Keep partial for resume.
                err = str(e)
                yield {
                    "status": "retrying" if attempt + 1 < max_retries else "error",
                    "error": err,
                    "attempt": attempt + 1,
                    "max_retries": max_retries,
                    "total": total,
                    "completed": completed,
                }
                if attempt + 1 >= max_retries:
                    return
                # Exponential-ish backoff
                await asyncio.sleep(min(8.0, 1.0 + (attempt * 2.0)))

