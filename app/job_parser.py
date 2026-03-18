"""Job description parser for shared pipeline context."""

from __future__ import annotations

import json
import re
from typing import Any

from app.ai_client import get_ai_client_for_model_id
from app.config import Settings


_JSON_RE = re.compile(r"\{[\s\S]*\}|\[[\s\S]*\]")


def _extract_json(text: str) -> Any:
    """
    Best-effort JSON extraction from a model response.
    Returns parsed Python object on success, otherwise raises.
    """
    s = (text or "").strip()
    if not s:
        raise ValueError("empty")
    # Prefer direct parse first
    try:
        return json.loads(s)
    except Exception:
        pass
    m = _JSON_RE.search(s)
    if not m:
        raise ValueError("no json found")
    return json.loads(m.group(0))


async def parse_job(job_description: str, config: Settings) -> str:
    """
    Parse job description into a compact JSON blob for reuse across resume + cover steps.
    Returns JSON text (pretty-printed).
    """
    jd = (job_description or "").strip()
    if not jd:
        return json.dumps({"required_skills": [], "preferred_skills": [], "responsibilities": [], "raw": ""}, indent=2)

    client, _ = get_ai_client_for_model_id(getattr(config, "OLLAMA_FALLBACK_MODEL_ID", "ollama:qwen3:14b"), config)
    system = (
        "You are a job description parser.\n"
        "Extract:\n"
        "1) required_skills (must-have; include both technical and soft skills)\n"
        "2) preferred_skills (nice-to-have)\n"
        "3) responsibilities (5-10 key responsibilities)\n"
        "4) experience_years (number or null)\n"
        "5) certifications (list)\n"
        "6) industry_or_domain (string or null)\n"
        "7) keywords (list)\n\n"
        "Output ONLY valid JSON. No commentary.\n"
    )
    prompt = f"JOB DESCRIPTION:\n{jd}\n"
    out = await client.generate(
        prompt,
        system_prompt=system,
        max_tokens=1200,
        params={"temperature": 0.2, "top_p": 0.9, "num_ctx": int(getattr(config, "OLLAMA_NUM_CTX", 8192))},
    )
    try:
        parsed = _extract_json(out)
        # Ensure object-like JSON for downstream templates.
        if not isinstance(parsed, (dict, list)):
            parsed = {"parsed": parsed}
        return json.dumps(parsed, indent=2)
    except Exception:
        return json.dumps({"raw": (out or "").strip()[:20000]}, indent=2)

