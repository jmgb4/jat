"""Context window management: token estimation, truncation, and summarisation.

Optional in-memory cache for prepare_context (TTL 300s, max 10 entries).

Priority tiers
--------------
Tier 1 — PROTECTED (included first, at full length, never AI-summarised):
    dos_and_donts, style_guide

    These files contain hard rules the model must follow. Dropping or compressing
    them silently would undermine the entire generation quality. They are always
    included in full; if they alone exceed the budget the excess is hard-truncated
    (still better than omitting the rules entirely).

Tier 2 — COMPRESSIBLE (fill remaining budget after Tier 1):
    story, role_adaptation, achievements, keywords, and anything else

    Processed in that order for AI summarisation: story is the longest/most
    narrative, so it benefits most from compression. The rest fill remaining
    space; if a file still doesn't fit after summarisation it is dropped.
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from app.tokenizer import estimate_tokens

logger = logging.getLogger(__name__)
from app.utils import (
    load_context_files,
    truncate_text,
)

# Keys (filename stems, lowercase) that are always included at full length first.
_PROTECTED: frozenset[str] = frozenset({"dos_and_donts", "style_guide"})

# Preferred compression order for Tier-2 files (story first — most verbose).
# career_context_vault: sole source of truth — identity, raw stories 1.1–3.5, tech list, framing, prohibitions.
# professional_summary_and_stories: narrative summary + Stories 22–27 (What I Did / Why It Mattered / Added Context).
# translation_guide: recontextualize-without-fabricate rules, SOC translation table, honesty filters.
# story_mapping_guide: which stories fit which role types/themes, keyword-to-story mapping.
# cover_letter_framework: four-paragraph structure, tone, default stories by job type, hook/proof/bridge/close templates.
# resume_formatting: layout (no horizontal rules, two-column header), summary, technical expertise, experience bullets, education, cover letter format.
# yahoo_resume_outline: preferred base resume structure and content to build from when tailoring.
# human_tone: no percentages, buzzword replacements, colleague test, human voice (canonical for tone).
# certifications_deep_dive: GCIH, GPEN, GWAPT, GSEC official knowledge areas; when to pull from each (incident response, pen test, web app, cloud, AI).
# final_polish: humble confidence, anti-arrogant, conciseness (human_tone is canonical for buzzwords/no %).
_TIER2_ORDER = ("story", "professional_summary_and_stories", "career_context_vault", "role_adaptation", "translation_guide", "story_mapping_guide", "cover_letter_framework", "resume_formatting", "yahoo_resume_outline", "human_tone", "certifications_deep_dive", "final_polish", "achievements", "keywords")

_CONTEXT_CACHE_TTL = 300
_CONTEXT_CACHE_MAX = 10
_context_cache: Dict[tuple, tuple[str, float]] = {}


async def prepare_context(
    base_resume: str,
    context_files: Dict[str, str],
    job_description: str,
    ai_client: Any,
    max_tokens: int,
    model_id: Optional[str] = None,
) -> str:
    """
    Combine base resume, context files, and job description into a single string
    that fits within *max_tokens*.

    Tier 1 (dos_and_donts, style_guide) are always written in full.
    Tier 2 fills the remaining budget; individual files are AI-summarised or
    dropped if they don't fit.     When model_id is provided (e.g. deepseek:...),
    token counts use tiktoken when available for better budgeting.
    """
    # Optional cache (key by resume + job snippet, max_tokens, model_id)
    cache_key = (hash((base_resume[:4000], job_description[:4000])), max_tokens, model_id or "")
    now = time.monotonic()
    if cache_key in _context_cache:
        cached, ts = _context_cache[cache_key]
        if now - ts < _CONTEXT_CACHE_TTL:
            logger.info("prepare_context: cache hit")
            return cached
        del _context_cache[cache_key]
    build_start = time.monotonic()
    # Evict oldest if at cap
    while len(_context_cache) >= _CONTEXT_CACHE_MAX and _context_cache:
        oldest = min(_context_cache.items(), key=lambda x: x[1][1])
        del _context_cache[oldest[0]]

    def tok(s: str) -> int:
        return estimate_tokens(s, model_id)

    # ── budget split: give base resume and job description plenty of room so we don't truncate material the model needs for a long, complete resume ──────────────────────────────────────────────────────────
    resume_tokens = min(tok(base_resume), max_tokens // 2)
    job_tokens = min(tok(job_description), max_tokens // 4)
    overhead = 500
    context_budget = max(max_tokens - resume_tokens - job_tokens - overhead, 0)

    # ── Tier 1: protected files ───────────────────────────────────────────────
    tier1_parts: list[str] = []
    used_tier1 = 0
    for key in sorted(_PROTECTED):
        content = context_files.get(key, "")
        if not content:
            continue
        token_count = tok(content)
        if used_tier1 + token_count > context_budget and context_budget > 0:
            allowed = max(context_budget - used_tier1, 50)
            content = truncate_text(content, allowed)
            token_count = tok(content)
        tier1_parts.append(f"[CONTEXT_{key.upper()}]\n{content}")
        used_tier1 += token_count

    # ── Tier 2: compressible files (parallel summarisation where needed) ──────
    remaining = max(context_budget - used_tier1, 0)
    tier2_keys = _tier2_order(context_files)
    # First pass: decide for each key whether to include full, summarise, or drop; collect summarise tasks.
    to_summarise: list[tuple[str, str, int]] = []
    full_parts: list[tuple[str, str, int]] = []
    used_after_full = used_tier1
    for key in tier2_keys:
        if remaining - used_after_full <= 0:
            break
        content = context_files.get(key, "")
        if not content:
            continue
        token_count = tok(content)
        allowance = remaining - used_after_full
        if token_count <= allowance:
            full_parts.append((key, content, token_count))
            used_after_full += token_count
        elif allowance > 100:
            to_summarise.append((key, content, allowance))

    # Run all summarisations in parallel.
    summaries: dict[str, str] = {}
    if to_summarise:
        tasks = [_summarise_text(content, allowance, ai_client) for (_, content, allowance) in to_summarise]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for (key, _, _), result in zip(to_summarise, results):
            if isinstance(result, str) and result:
                summaries[key] = result

    # Assemble Tier 2 in order: full content then summaries (by original key order).
    tier2_parts: list[str] = []
    used_tier2 = 0
    for key, content, token_count in full_parts:
        tier2_parts.append(f"[CONTEXT_{key.upper()}]\n{content}")
        used_tier2 += token_count
    for key, content, allowance in to_summarise:
        summary = summaries.get(key)
        if summary:
            stok = tok(summary)
            if used_tier2 + stok <= remaining:
                tier2_parts.append(f"[CONTEXT_{key.upper()}]\n{summary}")
                used_tier2 += stok

    # ── Assemble final string ─────────────────────────────────────────────────
    parts = [
        f"[BASE_RESUME]\n{truncate_text(base_resume, resume_tokens)}",
        f"[JOB_DESCRIPTION]\n{truncate_text(job_description, job_tokens)}",
        *tier1_parts,
        *tier2_parts,
    ]
    result = "\n\n---\n\n".join(parts)
    _context_cache[cache_key] = (result, now)
    elapsed = time.monotonic() - build_start
    logger.info("prepare_context: built in %.1fs", elapsed)
    return result


def _tier2_order(context_files: Dict[str, str]) -> list[str]:
    """
    Return Tier-2 keys in preferred compression order.

    Named files in _TIER2_ORDER come first (story → role_adaptation → …),
    followed by any other non-protected keys in alphabetical order.
    """
    ordered: list[str] = []
    seen: set[str] = set()
    for key in _TIER2_ORDER:
        if key in context_files and key not in _PROTECTED:
            ordered.append(key)
            seen.add(key)
    for key in sorted(context_files.keys()):
        if key not in _PROTECTED and key not in seen:
            ordered.append(key)
    return ordered


async def _summarise_text(text: str, target_tokens: int, ai_client: Any) -> str:
    """Use AI to condense *text* to roughly *target_tokens*. Falls back to truncation."""
    if estimate_tokens(text) <= target_tokens:
        return text
    prompt = (
        f"Summarise the following text concisely, preserving key facts and achievements. "
        f"Keep the summary under {target_tokens * 4} characters.\n\nText:\n{text[:20000]}\n\nSummary:"
    )
    try:
        summary = await ai_client.generate(
            prompt,
            system_prompt="You are a concise summariser. Output only the summary, no preamble.",
        )
        return summary or truncate_text(text, target_tokens)
    except Exception as e:
        logger.warning("summarise_text failed, falling back to truncation: %s", e)
        return truncate_text(text, target_tokens)
