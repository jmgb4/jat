"""Classify job role as leadership, engineering, or combined for resume selection."""

from __future__ import annotations

import json
import re
from typing import Literal

RoleKind = Literal["leadership", "engineering", "combined"]

# Keywords (case-insensitive) that suggest leadership/management focus.
LEADERSHIP_KEYWORDS = [
    "lead", "leader", "leadership", "manager", "management", "director",
    "head of", "head of team", "people lead", "people manager",
    "managing team", "managing engineers", "team lead", "principal",
]

# Keywords that suggest engineering/technical/VM focus.
ENGINEERING_KEYWORDS = [
    "engineer", "engineering", "developer", "sre", "devops",
    "vm", "virtualization", "virtual machine", "vsphere", "vmware",
    "technical", "infrastructure", "systems", "software",
    "architect", "implementation", "hands-on", "coding", "infra",
]


def _score_text(text: str, keywords: list[str]) -> int:
    """Return number of keyword matches (by word or phrase) in text."""
    if not (text or text.strip()):
        return 0
    lower = text.lower()
    count = 0
    for kw in keywords:
        # Prefer whole-word/phrase: use word boundary for single words.
        if " " in kw:
            if kw in lower:
                count += 2  # phrase match
        else:
            if re.search(r"\b" + re.escape(kw) + r"\b", lower):
                count += 1
    return count


def classify_role(
    job_title: str,
    job_description: str,
    parsed_job_json: str | None = None,
) -> RoleKind:
    """
    Classify the job as primarily leadership, engineering, or combined.

    Uses keyword scoring on title, description, and optionally parsed job JSON.
    Returns "combined" when the role is unclear or mixed so the app can use
    both leadership and engineering resume content.
    """
    title = (job_title or "").strip()
    desc = (job_description or "").strip()
    combined = f"{title}\n{desc}"

    if parsed_job_json and parsed_job_json.strip():
        try:
            data = json.loads(parsed_job_json)
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, str):
                        combined += "\n" + v
                    elif isinstance(v, list):
                        for item in v:
                            if isinstance(item, str):
                                combined += "\n" + item
        except (json.JSONDecodeError, TypeError):
            pass

    lead_score = _score_text(combined, LEADERSHIP_KEYWORDS)
    eng_score = _score_text(combined, ENGINEERING_KEYWORDS)

    # Clear winner: one score is strictly higher.
    if lead_score > eng_score:
        return "leadership"
    if eng_score > lead_score:
        return "engineering"

    # Tie or both zero: use combined resume.
    return "combined"
