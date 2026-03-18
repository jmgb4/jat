"""Load personal placeholders from a secure local folder for substitution into resume/cover.

LOCAL ONLY — No private information from this folder is ever sent to any API.
Substitution runs only after all AI/API calls, on the final text, on this machine.
"""

import re
from pathlib import Path
from typing import Dict

from app.utils import load_text_file


# Only allow placeholder keys that are alphanumeric and underscore (no path traversal).
_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


def load_personal_vars(config) -> Dict[str, str]:
    """
    Read key=value pairs from data/personal/*.txt (including template.txt).
    Returns a dict of placeholder name -> value. Keys must match [A-Za-z0-9_]+.
    Values are stripped; empty values are skipped. If the folder or file is missing, returns {}.
    Load order: template.txt first, then other .txt files (e.g. personal_info.txt), so user files override template.
    """
    personal_dir = getattr(config, "PERSONAL_DIR", "data/personal/")
    base = Path(__file__).resolve().parent.parent
    folder = base / personal_dir if not Path(personal_dir).is_absolute() else Path(personal_dir)
    if not folder.is_dir():
        return {}
    result: Dict[str, str] = {}
    # Load template.txt first, then all other .txt files (sorted), so personal_info.txt etc. override template.
    files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() == ".txt"]
    files.sort(key=lambda f: (0 if f.name.lower() == "template.txt" else 1, f.name))
    for fname in files:
        try:
            content = load_text_file(fname)
        except OSError:
            continue
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                if _KEY_PATTERN.match(key):
                    result[key] = value
    return result


def apply_personal_fill(text: str, vars: Dict[str, str]) -> str:
    """Replace {{KEY}} in text with vars.get(KEY, ''). Also replace [COLLEGE NAME] with COLLEGE if set. Leaves placeholders unchanged if key not in vars."""
    if not text or not vars:
        return text
    out = text
    for key, value in vars.items():
        out = out.replace("{{" + key + "}}", value)
    # Legacy placeholders: if model outputs [COLLEGE NAME] or [YOUR NAME], fill from vars when set
    if "COLLEGE" in vars and vars["COLLEGE"]:
        out = out.replace("[COLLEGE NAME]", vars["COLLEGE"])
        out = out.replace("[UNIVERSITY NAME]", vars["COLLEGE"])
    if "HEADER_LINE2" in vars and vars["HEADER_LINE2"]:
        out = out.replace("[YOUR NAME]", vars["HEADER_LINE2"])
    return out
