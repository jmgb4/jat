"""Registry for installed models (non-Ollama runtime)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from app.config import Settings


Runtime = Literal["ollama", "llamacpp", "transformers"]


def _safe_id(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("\\", "/")
    s = re.sub(r"[^a-z0-9._/-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "model"


@dataclass(frozen=True)
class InstalledModel:
    id: str
    display_name: str
    runtime: Runtime
    source_icon: str
    gguf_path: Optional[str] = None
    llamacpp_params: Optional[dict[str, Any]] = None
    hf_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "runtime": self.runtime,
            "source_icon": self.source_icon,
            "gguf_path": self.gguf_path,
            "llamacpp_params": self.llamacpp_params,
            "hf_path": self.hf_path,
        }


class ModelRegistry:
    def __init__(self, config: Settings):
        self._config = config
        p = Path(config.MODEL_REGISTRY_PATH)
        self._path = p if p.is_absolute() else (Path(__file__).parent.parent / p)

    def _read(self) -> list[dict[str, Any]]:
        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
            return data if isinstance(data, list) else []
        except FileNotFoundError:
            return []
        except Exception:
            return []

    def _write(self, items: list[dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(items, indent=2), encoding="utf-8")

    def list(self) -> list[InstalledModel]:
        out: list[InstalledModel] = []
        for it in self._read():
            try:
                runtime = it.get("runtime")
                if runtime not in ("llamacpp", "ollama", "transformers"):
                    continue
                out.append(
                    InstalledModel(
                        id=str(it.get("id") or ""),
                        display_name=str(it.get("display_name") or it.get("id") or ""),
                        runtime=runtime,  # type: ignore[arg-type]
                        source_icon=str(it.get("source_icon") or "⬡"),
                        gguf_path=(str(it.get("gguf_path")) if it.get("gguf_path") else None),
                        llamacpp_params=(it.get("llamacpp_params") if isinstance(it.get("llamacpp_params"), dict) else None),
                        hf_path=(str(it.get("hf_path")) if it.get("hf_path") else None),
                    )
                )
            except Exception:
                continue
        return [m for m in out if m.id]

    def add_llamacpp(
        self,
        display_name: str,
        gguf_path: str,
        source_icon: str = "⬡",
        llamacpp_params: Optional[dict[str, Any]] = None,
    ) -> InstalledModel:
        mid = f"gguf:{_safe_id(display_name)}"
        model = InstalledModel(
            id=mid,
            display_name=display_name.strip() or mid,
            runtime="llamacpp",
            source_icon=source_icon,
            gguf_path=str(gguf_path),
            llamacpp_params=llamacpp_params or None,
        )
        items = self._read()
        items = [x for x in items if str(x.get("id")) != model.id]
        items.append(model.to_dict())
        self._write(items)
        return model

    def add_transformers(
        self,
        display_name: str,
        hf_path: str,
        source_icon: str = "◆",
        params: Optional[dict[str, Any]] = None,
    ) -> InstalledModel:
        mid = f"hf:{_safe_id(display_name)}"
        model = InstalledModel(
            id=mid,
            display_name=display_name.strip() or mid,
            runtime="transformers",
            source_icon=source_icon,
            hf_path=str(hf_path),
            llamacpp_params=(params or None),
        )
        items = self._read()
        items = [x for x in items if str(x.get("id")) != model.id]
        items.append(model.to_dict())
        self._write(items)
        return model

    def remove(self, model_id: str) -> bool:
        items = self._read()
        before = len(items)
        items = [x for x in items if str(x.get("id")) != model_id]
        self._write(items)
        return len(items) != before

    def update_llamacpp_params(self, model_id: str, params: dict[str, Any]) -> bool:
        """Update generation params for a registered model (llamacpp or ollama runtime)."""
        items = self._read()
        changed = False
        for it in items:
            if str(it.get("id")) != model_id:
                continue
            runtime = str(it.get("runtime") or "")
            if runtime not in ("llamacpp", "ollama"):
                continue
            it["llamacpp_params"] = params
            changed = True
            break
        if changed:
            self._write(items)
        return changed

