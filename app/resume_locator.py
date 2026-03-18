"""Base resume discovery and selection persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from app.config import Settings

Status = Literal["ok", "missing", "multiple"]


@dataclass(frozen=True)
class BaseResumeInfo:
    status: Status
    candidates: list[str]
    selected: str | None
    selected_path: Path | None
    # When True, both leadership and engineering paths are available; worker will choose by role.
    role_based: bool = False
    leadership_path: Path | None = None
    engineering_path: Path | None = None


def _candidates(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        return []
    paths = list(base_dir.glob("*.docx")) + list(base_dir.glob("*.txt"))
    # stable ordering for UI
    return sorted(paths, key=lambda p: p.name.lower())


def _selected_file(base_dir: Path) -> Path:
    return base_dir / ".selected"


def read_selected(base_dir: Path) -> str | None:
    try:
        value = _selected_file(base_dir).read_text(encoding="utf-8").strip()
        return value or None
    except OSError:
        return None


def write_selected(base_dir: Path, filename: str) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    _selected_file(base_dir).write_text(filename.strip(), encoding="utf-8")


def _find_by_convention(base_dir: Path) -> tuple[Path | None, Path | None]:
    """Find leadership and engineering resume paths by filename convention in base_dir."""
    paths = _candidates(base_dir)
    lead_path: Path | None = None
    eng_path: Path | None = None
    for p in paths:
        name_lower = p.name.lower()
        if "leadership" in name_lower and lead_path is None:
            lead_path = p
        if ("engineering" in name_lower or "vm" in name_lower) and eng_path is None:
            eng_path = p
    return lead_path, eng_path


def resolve_role_based_resume_paths(
    base_dir: Path,
    project_root: Path,
    config: "Settings",
) -> tuple[Path | None, Path | None]:
    """
    Return (leadership_path, engineering_path) for role-based resume selection.
    Uses config paths if both set, else filename convention in base_dir.
    """
    lead_cfg = (getattr(config, "BASE_RESUME_LEADERSHIP_PATH", None) or "").strip()
    eng_cfg = (getattr(config, "BASE_RESUME_ENGINEERING_PATH", None) or "").strip()
    if lead_cfg and eng_cfg:
        p_lead = Path(lead_cfg)
        p_eng = Path(eng_cfg)
        if not p_lead.is_absolute():
            p_lead = (project_root / p_lead).resolve()
        if not p_eng.is_absolute():
            p_eng = (project_root / p_eng).resolve()
        if p_lead.exists() and p_eng.exists():
            return p_lead, p_eng
    return _find_by_convention(base_dir)


def resolve_base_resume(base_dir: Path, project_root: Path | None = None, config: "Settings | None" = None) -> BaseResumeInfo:
    paths = _candidates(base_dir)
    candidates = [p.name for p in paths]
    if not candidates:
        return BaseResumeInfo(status="missing", candidates=[], selected=None, selected_path=None)

    # Check for role-based mode: two variant paths (config or convention).
    role_based = False
    lead_path: Path | None = None
    eng_path: Path | None = None
    if project_root is not None and config is not None:
        lead_path, eng_path = resolve_role_based_resume_paths(base_dir, project_root, config)
        role_based = lead_path is not None and eng_path is not None

    selected = read_selected(base_dir)
    if selected and selected in candidates:
        return BaseResumeInfo(
            status="ok",
            candidates=candidates,
            selected=selected,
            selected_path=base_dir / selected,
            role_based=role_based,
            leadership_path=lead_path,
            engineering_path=eng_path,
        )

    if len(candidates) == 1 and not role_based:
        only = candidates[0]
        return BaseResumeInfo(
            status="ok",
            candidates=candidates,
            selected=only,
            selected_path=base_dir / only,
            role_based=False,
            leadership_path=None,
            engineering_path=None,
        )

    # Multiple candidates, or role-based: if role_based we're ok without .selected
    if role_based:
        return BaseResumeInfo(
            status="ok",
            candidates=candidates,
            selected=selected if selected and selected in candidates else None,
            selected_path=base_dir / selected if selected and selected in candidates else None,
            role_based=True,
            leadership_path=lead_path,
            engineering_path=eng_path,
        )

    return BaseResumeInfo(
        status="multiple",
        candidates=candidates,
        selected=None,
        selected_path=None,
        role_based=False,
        leadership_path=None,
        engineering_path=None,
    )

