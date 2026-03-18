"""File I/O, sanitisation, token estimation, and context loading."""

import os
import re
from pathlib import Path
from typing import Dict, Optional


def sanitise_filename(name: str) -> str:
    """Replace unsafe characters for use in filenames."""
    safe = re.sub(r'[<>:"/\\|?*]', "_", name)
    return safe.strip() or "unnamed"


def make_download_filename(job_title: Optional[str], suffix: str) -> str:
    """Build a friendly download filename: SafeJobTitle_Year_suffix.md (e.g. resume or cover_letter)."""
    from datetime import datetime
    year = datetime.now().year
    safe = (job_title or "").strip()
    safe = re.sub(r"[^a-zA-Z0-9 ]", "", safe).strip().replace(" ", "_")[:45]
    if not safe:
        safe = "result"
    return f"{safe}_{year}_{suffix}.md"


def ensure_dir(path: str | Path) -> None:
    """Create directory and parents if they do not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def load_text_file(path: str | Path, encoding: str = "utf-8") -> str:
    """Read a text file and return its contents."""
    with open(path, encoding=encoding) as f:
        return f.read()


def save_text_file(path: str | Path, content: str, encoding: str = "utf-8") -> None:
    """Write content to a text file and flush to disk."""
    path = Path(path).resolve()
    ensure_dir(path.parent)
    with open(path, "w", encoding=encoding) as f:
        f.write(content or "")
        f.flush()
        try:
            os.fsync(f.fileno())
        except (OSError, AttributeError):
            pass


def load_context_files(context_dir: str | Path) -> Dict[str, str]:
    """Read all .txt files from context_dir. Keys are filenames without extension."""
    context_dir = Path(context_dir)
    if not context_dir.is_dir():
        return {}
    result: Dict[str, str] = {}
    for p in sorted(context_dir.glob("*.txt")):
        try:
            result[p.stem] = load_text_file(p)
        except OSError:
            continue
    return result


def load_context_files_concatenated(context_dir: str | Path) -> str:
    """Read all .txt files and concatenate with clear separators."""
    files = load_context_files(context_dir)
    if not files:
        return ""
    parts = [f"--- {name} ---\n{content}" for name, content in files.items()]
    return "\n\n".join(parts)


def estimate_tokens(text: str) -> int:
    """Rough token count for English: ~1 token per 4 characters."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def truncate_text(text: str, max_tokens: int) -> str:
    """Truncate text at estimated token boundary."""
    if not text or max_tokens <= 0:
        return ""
    estimated = estimate_tokens(text)
    if estimated <= max_tokens:
        return text
    target_chars = max_tokens * 4
    return text[:target_chars].rsplit(maxsplit=1)[0] if target_chars < len(text) else text[:target_chars]


def format_number(num: int | float) -> str:
    """Format number for display (e.g. 1234 -> 1,234)."""
    if isinstance(num, int):
        return f"{num:,}"
    if isinstance(num, float):
        if num == int(num):
            return f"{int(num):,}"
        return f"{int(num):,}.{str(num).split('.', 1)[1]}"
    return str(num)


# Replace percentage phrases with generic wording (resume/cover). Match common metric patterns.
_PERCENT_GENERIC_PAIRS = [
    (re.compile(r"\b(?:reduced|decreased|cut|lowered)\s+by\s+\d+(?:\.\d+)?\s*%", re.IGNORECASE), "reduced significantly"),
    (re.compile(r"\b(?:increased|improved|raised)\s+by\s+\d+(?:\.\d+)?\s*%", re.IGNORECASE), "increased significantly"),
    (re.compile(r"\bby\s+\d+(?:\.\d+)?\s*%\b", re.IGNORECASE), "significantly"),
    (re.compile(r"\b\d+(?:\.\d+)?\s*%\s*(improvement|increase|growth)\b", re.IGNORECASE), r"significant \1"),
    (re.compile(r"\b\d+(?:\.\d+)?\s*%\s*(reduction|decrease)\b", re.IGNORECASE), r"significant \1"),
    (re.compile(r"\b(?:improvement|increase|growth)\s+of\s+\d+(?:\.\d+)?\s*%", re.IGNORECASE), "significant improvement"),
    (re.compile(r"\b(?:reduction|decrease)\s+of\s+\d+(?:\.\d+)?\s*%", re.IGNORECASE), "significant reduction"),
    (re.compile(r"\b(?:up\s+to|by)\s+\d+(?:\.\d+)?\s*%\s+(faster|better|more|less)\b", re.IGNORECASE), r"substantially \1"),
    (re.compile(r"\b\d+(?:\.\d+)?\s*%\s+(faster|better|more|less)\b", re.IGNORECASE), r"substantially \1"),
]
# Fallback: standalone metric percentage (avoid "100% remote", "50% travel")
_PERCENT_STANDALONE = re.compile(r"\b\d+(?:\.\d+)?\s*%\b(?!\s*(?:remote|travel|distributed|on-site|hybrid))")


def _replace_percentages_with_generic(text: str) -> str:
    """Replace percentage-based metrics with generic wording (significantly, substantially). Preserves work-arrangement terms like 100% remote."""
    out = text
    for pattern, repl in _PERCENT_GENERIC_PAIRS:
        out = pattern.sub(repl, out)
    # Standalone percentage that looks like a metric (not remote/travel etc.): replace with "significantly"
    out = _PERCENT_STANDALONE.sub("significantly", out)
    return out


def sanitise_resume_style(text: str) -> str:
    """Post-process resume/cover text: reduce percentages to generic wording, Oxford commas, double dashes, horizontal rules, bullet markers."""
    if not text or not text.strip():
        return text
    out = _replace_percentages_with_generic(text)
    # Oxford comma: ", and " -> " and "; also ", and" at word boundary (end of line or before punctuation)
    out = re.sub(r",\s+and\s+", " and ", out)
    out = re.sub(r",\s+and\b", " and", out)
    # Double dashes: " -- " -> " — "; "---" (horizontal) handled below; normalize en-dash/em-dash between words
    out = re.sub(r"\s+--\s+", " — ", out)
    out = re.sub(r"(?<=[^\s-])--(?=[^\s-])", "—", out)
    out = re.sub(r"\s+[–—]\s+", " — ", out)  # Unicode en-dash (U+2013), em-dash (U+2014)
    # Horizontal rules: line of only --- or *** or ___ -> blank line (do not touch lines with other content)
    lines = out.splitlines()
    new_lines = []
    for line in lines:
        s = line.strip()
        if re.match(r"^[-]{3,}$", s) or re.match(r"^[*]{3,}$", s) or re.match(r"^[_]{3,}$", s):
            new_lines.append("")
        else:
            # Normalise bullet lines: "* " or "- " (with optional leading space) -> "* " + content, single space, aligned left
            bullet_match = re.match(r"^(\s*)([-*])\s+(.*)$", line)
            if bullet_match:
                # Line starts with - or * followed by at least one space: treat as bullet (avoids matching **Bold**)
                rest = bullet_match.group(3).strip()
                rest = re.sub(r"\s+", " ", rest)  # collapse multiple spaces to one
                new_lines.append("* " + rest if rest else "* ")
            else:
                new_lines.append(line)
    # Remove a single blank line between job title (or company) and the first bullet.
    result = []
    for i, line in enumerate(new_lines):
        if (
            line.strip() == ""
            and result
            and i + 1 < len(new_lines)
            and result[-1].strip()
            and not result[-1].strip().startswith("* ")
            and new_lines[i + 1].strip().startswith("* ")
        ):
            continue
        result.append(line)
    return "\n".join(result)


def calculate_speed(start_time: float, tokens_generated: int) -> float:
    """Return tokens per second; 0 if no time elapsed or no tokens."""
    import time
    elapsed = time.monotonic() - start_time
    if elapsed <= 0 or tokens_generated <= 0:
        return 0.0
    return tokens_generated / elapsed


def extract_job_title_from_text(text: str) -> Optional[str]:
    """Simple heuristic: first non-empty line, trimmed."""
    if not text:
        return None
    for line in text.strip().splitlines():
        line = line.strip()
        if line and len(line) < 200:
            return line
    return None
