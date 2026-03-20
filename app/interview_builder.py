"""Two-pass interview preparation generator.

Pass 1 — Reasoner:
    Analyzes the job description against the candidate's full context.
    Maps requirements to evidence, identifies gaps, predicts 10-15 likely
    interview questions, and proposes STAR stories for each.

Pass 2 — Chat / Polisher:
    Takes the Pass 1 analysis and produces the final, formatted interview
    prep document: narrative threads, full STAR answers, opening/closing
    statements, and questions to ask the interviewer.

Public API:
    run_pass1(...)  → analysis string
    run_pass2(...)  → final prep markdown string
    generate_interview_prep(...)  → convenience wrapper (both passes)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional  # Any needed for on_chunk return type

from app.ai_client import get_ai_client_for_model_id
from app.config import Settings, get_settings

logger = logging.getLogger("jat.interview_builder")

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_REASONER_SYSTEM = """\
You are an expert career coach and senior hiring manager.
Your job is to analyze a job description against a candidate's background and produce a \
structured analysis that will be used to write comprehensive interview preparation materials.

Output ONLY the structured analysis — no preamble, no sign-off.

FORMAT YOUR OUTPUT AS:

## Requirements Map
For each key requirement in the job description, one row:
| Requirement | Candidate Evidence | Strength (Strong/Partial/Gap) |

## Gap Bridges
For each Gap or Partial from the map, one entry:
**Gap: [requirement]**
Bridge: [adjacent experience that proves capability + how to frame it]

## Narrative Threads
List 4-6 consistent career threads that run through the candidate's background.
These will anchor every interview answer.
Example thread: "Builder — built X from scratch, built Y from scratch"

## Predicted Questions (10-15)
For each question:
**Q: [Interview question]**
Type: [Behavioral / Technical / Situational / Culture]
Story to use: [reference from achievements/story context]
Key points to hit: [2-3 bullet points]
Watch-out: [one thing to avoid saying]

## Company Language to Mirror
List 5-8 exact phrases from the job description to weave into answers naturally.
"""

_CHAT_SYSTEM = """\
You are a senior career coach writing final interview preparation materials.
You have been given a structured analysis. Turn it into a polished, complete prep guide.

Write in a warm, direct, second-person voice ("You led...", "Your answer here...").
Every STAR answer must follow: Situation → Task → Action → Result → Lesson.

Each STAR answer must be 75-120 words — that is 30-60 seconds when spoken aloud.
Tight, punchy, and memorisable. Cut filler. Every sentence must earn its place.
Structure: Situation (1 sentence) → Task (1 sentence) → Action (2-3 sentences) →
Result (1 sentence) → Lesson (1 sentence). That is your budget. Do not exceed it.

Maximum 2 stories per question. Only include a second story when it covers a clearly
different skill or context. If one story is sufficient, omit Story 2 entirely.

Output ONLY the formatted prep document — no meta-commentary.
"""

_CHAT_PROMPT_TEMPLATE = """\
## Candidate Analysis (from reasoning step)
{analysis}

---

## Job Description
{job_description}

---

## Candidate Context
{context}

---

Using the analysis above, write the complete interview prep document with these sections:

### 1. Your 5 Narrative Threads
Short paragraph for each thread. Tell the candidate what they are and why they matter.

### 2. Opening Statement
A 90-second "tell me about yourself" answer tailored to this specific role.
Use their language, hit 2 threads.

### 3. Predicted Questions & STAR Answers
For every predicted question from the analysis, use this exact layout:

**Q[N]: [question]**
- Resume bullets this proves: [list the exact bullet(s) from the resume that this answer demonstrates]
- Job posting requirements this answers: [list the exact requirement(s) from the JD this directly addresses]

**Story** (75-120 words, STAR format):
[Tight STAR answer — Situation 1 sentence, Task 1 sentence, Action 2-3 sentences, Result 1 sentence, Lesson 1 sentence. Humble-confident tone. No percentages.]

**How to use this story:** [1-2 sentences of delivery coaching: what to emphasise, how to open the answer, what watch-out to avoid per the analysis]

*(Only add Story 2 below if there is a meaningfully different skill or context — otherwise omit it entirely)*
**Story 2 — alternate angle** (75-120 words, optional):
[Second STAR answer using a different project or skill dimension, same structure]

**How to use Story 2:** [1-2 sentences delivery coaching for this alternate angle]

### 4. Closing Statement
A 2-3 sentence closing when asked "Do you have any questions or anything to add?"

### 5. Questions to Ask the Interviewer
5 thoughtful questions that show preparation and genuine curiosity.
One should reference a specific detail from the job description.

### 6. Quick Reference Card
A bullet-point cheat sheet: key phrases to use, phrases to avoid, 3 numbers/facts to \
remember, and one story per thread.
"""


# ---------------------------------------------------------------------------
# Public API — individual passes
# ---------------------------------------------------------------------------

async def run_pass1(
    *,
    job_description: str,
    base_resume: str,
    context_text: str,
    model_id: str,
    on_chunk: Optional[Callable[[str], Any]] = None,
    extra_context: str = "",
) -> str:
    """Pass 1: Reasoner — structural analysis.

    Returns the raw analysis markdown that Pass 2 consumes.
    on_chunk(text) is called with each text fragment as it arrives (optional).
    extra_context is appended to the prompt when the user provides custom focus areas.
    """
    config = get_settings()
    client, _ = get_ai_client_for_model_id(model_id, config)
    prompt = (
        "## Job Description\n"
        f"{job_description.strip()}\n\n"
        "---\n\n"
        "## Candidate's Base Resume\n"
        f"{base_resume.strip()}\n\n"
        "---\n\n"
        "## Candidate's Career Context\n"
        f"{context_text.strip()}\n\n"
    )
    if extra_context.strip():
        prompt += (
            "---\n\n"
            "## Candidate Notes / Focus Areas\n"
            f"{extra_context.strip()}\n\n"
        )
    prompt += "---\n\nProduce the structured interview analysis as specified."
    return await client.generate(
        prompt=prompt,
        system_prompt=_REASONER_SYSTEM,
        params={"temperature": 0.2},
        on_chunk=on_chunk,
    )


async def run_pass2(
    *,
    analysis: str,
    job_description: str,
    context_text: str,
    model_id: str,
    on_chunk: Optional[Callable[[str], Any]] = None,
) -> str:
    """Pass 2: Chat / Polish — final prep document.

    Takes the Pass 1 analysis and returns the formatted prep markdown.
    on_chunk(text) is called with each text fragment as it arrives (optional).
    """
    config = get_settings()
    client, _ = get_ai_client_for_model_id(model_id, config)
    prompt = _CHAT_PROMPT_TEMPLATE.format(
        analysis=analysis.strip(),
        job_description=job_description.strip()[:6000],
        context=context_text.strip()[:8000],
    )
    result = await client.generate(
        prompt=prompt,
        system_prompt=_CHAT_SYSTEM,
        params={"temperature": 0.4},
        on_chunk=on_chunk,
    )
    return result.strip()


# ---------------------------------------------------------------------------
# Public API — convenience wrapper
# ---------------------------------------------------------------------------

async def generate_interview_prep(
    *,
    job_description: str,
    base_resume: str,
    context_text: str,
    model_sequence: List[str],
    progress_cb: Optional[Callable[[str], None]] = None,
) -> str:
    """Run the two-pass interview prep pipeline and return the final markdown document.

    Args:
        job_description: Full scraped job description text.
        base_resume: The candidate's base resume text.
        context_text: Pre-joined context (story, achievements, keywords, etc.).
        model_sequence: List of model IDs; index 0 = reasoner, index 1 = chat/polisher.
            If only one model is provided it is used for both passes.
        progress_cb: Optional callback(message) for progress reporting.

    Returns:
        Final interview prep markdown string.
    """
    config = get_settings()
    if not model_sequence:
        model_sequence = [config.OLLAMA_MODEL, config.OLLAMA_MODEL]

    reasoner_model = model_sequence[0]
    chat_model = model_sequence[1] if len(model_sequence) > 1 else model_sequence[0]

    if progress_cb:
        progress_cb(f"Pass 1 — Reasoning with {reasoner_model}…")
    logger.info("interview_prep pass1 model=%s", reasoner_model)

    analysis = await run_pass1(
        job_description=job_description,
        base_resume=base_resume,
        context_text=context_text,
        model_id=reasoner_model,
    )

    if progress_cb:
        progress_cb(f"Pass 2 — Polishing with {chat_model}…")
    logger.info("interview_prep pass2 model=%s", chat_model)

    prep_doc = await run_pass2(
        analysis=analysis,
        job_description=job_description,
        context_text=context_text,
        model_id=chat_model,
    )

    if progress_cb:
        progress_cb("Interview prep generation complete.")
    return prep_doc
