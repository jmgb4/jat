"""Shared prompt rules used by resume/cover builders and pipeline JSON loader.

Single source of truth for placeholder rule, style (Oxford commas, dashes),
resume order/structure, and banned phrases. Used to reduce duplication and
keep Python prompts and pipeline JSON aligned.
"""

# Use placeholders only; never put real names/schools/companies in the output. They are filled locally after generation; no private information is sent to any API.
PLACEHOLDER_RULE = (
    "Do not invent or guess candidate name, school/college name, or company names. "
    "Use only these placeholders so the app can fill them locally after generation (no private data is ever sent to any API): "
    "{{JOB1_COMPANY}}, {{JOB1_START}}, {{JOB1_END}} (and JOB2, JOB3) for each role, format Company | Start – End with a pipe; {{COLLEGE}} for school. Do not output contact/name at the top; user adds their own. "
    "Never output the user's real name, school, or company names — only the placeholders."
)

# One-line style rule for resumes and cover letters.
STYLE_NO_OXFORD_DASHES = (
    'No Oxford commas (write "A, B and C" not "A, B, and C"). '
    "No double dashes; use a single dash or rephrase."
)

# Summary at top: short (2–3 lines).
SUMMARY_THREE_LINES = "2–3 sentences (about 3 lines) only"

# Length: aim for 2 full pages; minimum bullet counts.
RESUME_LENGTH_RULE = (
    "Aim for 2 full pages. If under 2 pages, add relevant bullets from the base resume to reach 2 pages. "
    "Do not exceed 2 pages; if over 2 pages, trim the Technical Expertise section or tighten bullet wording, not remove bullets. "
    "Do not truncate or shorten below the minimum bullet counts. "
    "Technical Expertise: be concise. Do not list every tool or skill; highlight what is most relevant to the job. "
    "Prefer 2–4 categories with a short list each, not an exhaustive dump. "
    "Bullets: first job at least 10, second at least 6, third at least 6; quality over quantity but meet minimums and aim for 2 pages."
)

# Resume output order and structure. The app replaces these placeholders from a local file after generation; no private data is ever sent to any API.
RESUME_ORDER_RULE = (
    "Do not include contact lines or name at the top; the user will add their own. Start with: summary paragraph (" + SUMMARY_THREE_LINES + ") → blank line → "
    "**Technical Expertise:** (concise, 2–4 categories) → **Experience** (job section) with bullet points only → Education, Certifications. "
    "Under Experience, each role block: (1) **Role title** in bold on its own line — use {{JOB1_TITLE}}, {{JOB2_TITLE}}, {{JOB3_TITLE}} when provided, else from base resume; (2) next line: Company | Start – End using a pipe between company and dates, e.g. {{JOB1_COMPANY}} | {{JOB1_START}} – {{JOB1_END}} (and JOB2, JOB3); (3) then bullets. "
    "Education: use {{COLLEGE}} for the school name. Never put real names, companies, or schools in the output — only these placeholders; they are replaced locally after generation and no private information is sent to any API."
)

# Banned phrases for resume (replace with plain English).
BANNED_PHRASES_RESUME = (
    "leveraged→used, optimized→improved, architected→designed, orchestrated→coordinated, spearheaded→led, "
    "robust→solid, holistic→overall, proven track record→(show work), results-driven→(never), "
    "passionate about→(describe what you did), cutting-edge→new. "
    "Avoid \"in order to\", \"due to the fact that\", \"demonstrated the ability to\", \"tasked with\". "
    "Do not use: \"just what this position needs\", \"what this position needs\", \"exactly what this role needs\" or similar — delete or rephrase. "
    "Also flag: thought leadership→(never), deep expertise→(list skills instead)."
)

# Banned phrases for cover letter.
BANNED_PHRASES_COVER = (
    "leveraged→used, utilized→used, optimized→improved, architected→designed, orchestrated→coordinated, "
    "spearheaded→led, robust→solid, holistic→overall, actionable→useful, proven track record→(show work), "
    "thought leadership→(never), results-driven→(never), passionate about→(describe what you did), "
    "I am excited to apply→(be specific). Do not use: \"just what this position needs\", \"what this position needs\", \"exactly what this role needs\" — delete or rephrase. "
    "Avoid: \"in order to\" (use \"to\"), \"due to the fact that\" (use \"because\")."
)

# Summary paragraph spec (for injection into prompts); kept for backward compatibility in JSON.
SUMMARY_FIVE_SENTENCES = "about five sentences"

# Map of marker names to snippet strings (for pipeline JSON loader).
SNIPPET_MAP = {
    "PLACEHOLDER_RULE": PLACEHOLDER_RULE,
    "STYLE_NO_OXFORD_DASHES": STYLE_NO_OXFORD_DASHES,
    "RESUME_ORDER_RULE": RESUME_ORDER_RULE,
    "RESUME_LENGTH_RULE": RESUME_LENGTH_RULE,
    "BANNED_PHRASES_RESUME": BANNED_PHRASES_RESUME,
    "BANNED_PHRASES_COVER": BANNED_PHRASES_COVER,
    "SUMMARY_FIVE_SENTENCES": SUMMARY_FIVE_SENTENCES,
    "SUMMARY_THREE_LINES": SUMMARY_THREE_LINES,
}


def inject_snippets_into_string(text: str) -> str:
    """Replace {{SNIPPET_NAME}} markers in text with values from SNIPPET_MAP."""
    if not text or "{{" not in text:
        return text
    result = text
    for key, value in SNIPPET_MAP.items():
        result = result.replace("{{" + key + "}}", value)
    return result


def inject_snippets_into_pipeline(pipeline: dict) -> dict:
    """In-place inject snippet values into every step's system_prompt in resume_steps and cover_steps."""
    if not pipeline:
        return pipeline
    for key in ("resume_steps", "cover_steps"):
        steps = pipeline.get(key)
        if not isinstance(steps, list):
            continue
        for step in steps:
            if isinstance(step, dict) and "system_prompt" in step and isinstance(step["system_prompt"], str):
                step["system_prompt"] = inject_snippets_into_string(step["system_prompt"])
    return pipeline
