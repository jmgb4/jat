"""Download and register pass models for the pipeline."""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.config import get_settings
from app.hub_manager import HubManager, _safe_dirname
from app.model_registry import ModelRegistry


cfg = get_settings()
hub = HubManager(cfg)
reg = ModelRegistry(cfg)

# display_name, repo_id, filename, llamacpp_params
ITEMS = [
    (
        "GLM-4.7-Flash-NEO-CODE",
        "DavidAU/GLM-4.7-Flash-NEO-CODE-Imatrix-MAX-GGUF",
        "GLM-4.7-Flash-NEO-CODE-MAX-imat-D_AU-Q5_K_M.gguf",
        {"n_ctx": 16384, "temperature": 0.80, "top_p": 0.60, "top_k": 2, "repeat_penalty": 1.0},
    ),
    (
        "Magistral-Small-2509",
        "bartowski/mistralai_Magistral-Small-2509-GGUF",
        "mistralai_Magistral-Small-2509-Q5_K_M.gguf",
        {"n_ctx": 40960, "temperature": 0.70, "top_p": 0.95},
    ),
    (
        "Mistral Small 3.1",
        "Triangle104/Mistral-Small-3.1-24B-Instruct-2503-Q5_K_M-GGUF",
        "mistral-small-3.1-24b-instruct-2503-q5_k_m.gguf",
        {"n_ctx": 32768, "temperature": 0.70, "top_p": 0.95},
    ),
]


def existing_target_path(repo_id: str, filename: str) -> Path:
    return Path(hub._cache_dir) / _safe_dirname(repo_id) / filename


def reconcile_root_file(repo_id: str, filename: str) -> Path:
    """Move root cached gguf file into expected subfolder layout."""
    target = existing_target_path(repo_id, filename)
    root = Path(hub._cache_dir) / filename
    try:
        if root.exists() and (not target.exists()):
            target.parent.mkdir(parents=True, exist_ok=True)
            root.replace(target)
            return target
    except Exception:
        pass
    return target


async def download_one(display: str, repo_id: str, filename: str, params: dict) -> None:
    target = reconcile_root_file(repo_id, filename)
    if target.exists() and target.stat().st_size > 0:
        print(f"==> {display}: already downloaded")
        print(f"    path: {target}")
        reg.add_llamacpp(display_name=display, gguf_path=str(target), source_icon="⬡", llamacpp_params=params)
        print("    registered")
        return

    print(f"==> {display}: downloading")
    print(f"    {repo_id}/{filename}")
    last_pct = -1
    async for evt in hub.download_gguf_stream(repo_id, filename):
        st = evt.get("status")
        if st == "downloading":
            tot = int(evt.get("total") or 0)
            done = int(evt.get("completed") or 0)
            if tot > 0:
                pct = int((done / tot) * 100)
                if pct != last_pct:
                    print(f"    {pct:3d}% ({done}/{tot})", end="\r")
                    last_pct = pct
        elif st == "success":
            path = evt.get("path")
            print()
            print(f"    saved: {path}")
            reg.add_llamacpp(display_name=display, gguf_path=str(path), source_icon="⬡", llamacpp_params=params)
            print("    registered")
            return
        elif st == "error":
            raise RuntimeError(evt.get("error") or "download failed")


async def main() -> None:
    print(f"Cache dir: {hub._cache_dir}")
    for item in ITEMS:
        await download_one(*item)
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
