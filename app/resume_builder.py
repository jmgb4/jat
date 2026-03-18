"""Four-pass resume generation with Ollama and DeepSeek."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional

from app.ai_client import get_ai_client_for_model_id
from app.config import Settings, get_context_window
from app.context_manager import prepare_context
from app.pipeline_engine import PipelineStep, run_pipeline, steps_from_pipeline_config
from app.prompt_snippets import (
    BANNED_PHRASES_RESUME,
    PLACEHOLDER_RULE,
    RESUME_LENGTH_RULE,
    RESUME_ORDER_RULE,
    STYLE_NO_OXFORD_DASHES,
    SUMMARY_THREE_LINES,
)
from app.utils import save_text_file, sanitise_resume_style

logger = logging.getLogger(__name__)
_RESUME_EMPTY_PLACEHOLDER = "# Resume\n\n(Pipeline returned no content for this run.)"

RESUME_SYSTEM = f"""You are a resume writer. Output must sound like a real person wrote it. Two non-negotiable rules: (1) there MUST be a short summary at the top after the name ({SUMMARY_THREE_LINES}); (2) the resume must be at most 2 pages — do not exceed 2 pages.

PLACEHOLDERS — {PLACEHOLDER_RULE}

RULE 1 — Do not output contact or name at the top; the user will add their own. Start with a short summary of {SUMMARY_THREE_LINES}, then **Technical Expertise:**. The summary must connect the job posting to the candidate: use the job's required_skills and responsibilities and tie each to the candidate's real experience from the base resume or context. Be concise. No invented claims. {RESUME_ORDER_RULE}

LENGTH — {RESUME_LENGTH_RULE} Address every key point in the job posting without exceeding 2 pages.

HONESTY: Every claim must be traceable to the base resume or context. Never invent skills, tools, certs, roles or outcomes. Never add anything not in your materials. No language fluency or LLM security testing unless present.

HARD FORMAT: No horizontal rules. {STYLE_NO_OXFORD_DASHES} No first person in bullets; past tense for past roles. Section headings bold with colon: **Technical Expertise:**, **Education:**, **Certifications:**. No "Experience" heading — after Technical Expertise each job block: (1) **Role title** in bold on its own line (use {{JOB1_TITLE}}, {{JOB2_TITLE}}, {{JOB3_TITLE}} when provided in personal file, else the title from the base resume); (2) next line: Company | Start – End with a pipe between company and dates, e.g. {{JOB1_COMPANY}} | {{JOB1_START}} – {{JOB1_END}} (and JOB2, JOB3); (3) then bullets. Job descriptions: bullets only under each role; no paragraphs. Every achievement is a bullet starting with "* ". Under each job: no blank lines between bullets (back-to-back); one blank line only before the next company. Technical Expertise: * **Category:** item1, item2. Education and Certifications: plain text, no bullets. Use {{JOB1_COMPANY}} | {{JOB1_START}} – {{JOB1_END}} format (pipe between company and dates) and {{COLLEGE}} for school — these placeholders are filled locally after generation; never put real private info in the output. Do not output contact or name at the top; user adds their own.

REDUCE PERCENTAGES — Do not include specific percentages (e.g. "30%", "reduced by 40%") in the resume. Replace any percentage with generic wording: significantly, greatly, substantially, or similar. If the base resume or context mentions a percentage, still express the impact in words (e.g. "reduced costs significantly" not "reduced costs by 25%"). This keeps the resume professional and avoids overstated or outdated metrics.

BUZZWORDS — replace: {BANNED_PHRASES_RESUME}

STYLE: Vary sentence length and bullet openings. Use "we" for team work; clarify your role. Use "Led" only if you actually led.

CONTENT: Start bullets with strong verbs (Led, Built, Automated, Reduced, Designed). Name tools: Nessus not "vulnerability scanner", Splunk not "SIEM". Show outcome in words. Output only the resume text."""

REVIEW_SYSTEM = f"""You are a strict resume editor. Review the draft against the base resume and the writing rules. Do not rewrite the resume. Output only a bullet list of specific, actionable suggestions.

CHECK FOR EACH BULLET:
1. Banned phrases — flag and replace: {BANNED_PHRASES_RESUME}
2. Weak openers — flag "responsible for", "tasked with", "assisted with", "helped with". Suggest a concrete action verb.
3. Missing specifics — flag bullets with no tool name or scale. Suggest adding e.g. the tool name or scope. Do NOT suggest adding percentages or "X% improvement".
4. Generic duty vs concrete achievement — if the bullet has no outcome, suggest adding impact in words (e.g. "reduced manual effort significantly"). Never suggest adding a percentage.
5. Hallucination — flag any skill, cert, tool or achievement not in the base resume. Flag any invented institution, company or product names; names should be from base resume/context or ALL CAPS placeholders ([YOUR NAME], [COLLEGE NAME], [COMPANY NAME]). Flag language fluency, LLM security testing (prompt injection, etc.) unless present. Flag any invented numbers or percentages; suggest qualitative wording instead.
6. Style — flag {STYLE_NO_OXFORD_DASHES} first person in bullets, and any horizontal rules or full-page lines (use blank lines only). Under Experience, each job must have the role title in bold on its own line, then company and dates (standard), then bullets; flag if role title is not bold or format is wrong. Flag any blank line between two bullets under the same job (bullets within a job must be consecutive with no gaps). Flag any paragraph-style text under a job (job descriptions must be bullets only, no paragraphs). Flag bullets under Education or Certifications (those sections must be plain text, no bullets). Flag Technical Expertise if items are sub-bulleted (only the category line should be a bullet: * **Category:** item1, item2).
7. Bullet counts — first job at least 10 bullets; second at least 6; third at least 6. Do not drop bullets below these minimums. Aim for 2 full pages. Flag if any role is far below. Flag if resume exceeds 2 pages — suggest trimming technical detail or tightening wording, not removing bullets.
8. Job posting coverage — flag if key requirements from the job posting are not addressed; the resume must hit ALL points the base resume can support.
9. Summary at top (critical) — flag as CRITICAL if there is no summary ({SUMMARY_THREE_LINES}) at the top before **Technical Expertise:**. The resume starts with the summary (user adds their own header above); then **Technical Expertise:**. If it goes straight to Technical Expertise with no summary, the summary is missing. The summary must tie job requirements to the candidate's experience; flag if generic or too long.
10. Missing content — flag if bullets or roles from the base resume were dropped or shortened.
11. Sentence variety — flag if three or more consecutive bullets start the same way.

Be specific. Cite the exact phrase. One suggestion per bullet. Never suggest adding percentages."""

FINAL_SYSTEM = f"""Final-pass resume editor. Polish the draft. Output only the final resume text. The resume must be at most 2 pages and MUST have a short summary at the top.

PLACEHOLDERS: {PLACEHOLDER_RULE} Flag and fix any invented person, school or company name.

TWO MUST-DOS: (1) Summary at top: There MUST be {SUMMARY_THREE_LINES} at the start before **Technical Expertise:** (user adds their own contact/name above). If the draft has no such summary or it is too long, add or tighten it. (2) Length: Aim for 2 full pages. Bullets: first job at least 10, second at least 6, third at least 6. Do not truncate below minimums. If over 2 pages, trim technical detail or tighten wording only; do not remove bullets.

HONESTY: Do not add claims not in the base resume. Every claim traceable to the source.

FORMAT: No horizontal rules. {STYLE_NO_OXFORD_DASHES} No first person in bullets. Under Experience, each job: **Role title** (bold) on its own line ({{JOB1_TITLE}}, {{JOB2_TITLE}}, {{JOB3_TITLE}} or from base resume); next line Company | Start – End (pipe between company and dates); then bullets only, no paragraphs; no blank lines between bullets; one blank line before the next company. Technical Expertise: * **Category:** items. Education and Certifications: plain text. Use placeholders for names not in base resume.

Replace buzzwords. REDUCE PERCENTAGES — If the draft contains any percentage (e.g. "by 30%", "40% improvement"), replace it with generic wording: significantly, substantially, greatly. No specific percentages in the final resume. Human tone. Vary sentence openings."""


HUMANIZE_SYSTEM = f"""You are a humanizing editor. Your only job is to make the text read as if a real person wrote it in one sitting — not a language model.

RULES:
- Do NOT add, remove or change any facts, skills, dates, job titles or bullet content. Preserve every claim and section. Only change wording and rhythm. {PLACEHOLDER_RULE} Keep [YOUR NAME], [COLLEGE NAME], [COMPANY NAME] etc. as ALL CAPS placeholders; do not replace with invented names.
- REDUCE PERCENTAGES — If any bullet still contains a specific percentage (e.g. "30%", "reduced by 40%"), replace it with generic wording (significantly, substantially, greatly). Do not leave numeric percentages in the final text.
- Fix AI-like patterns: (1) Too many bullets starting with the same structure (e.g. "Led...", "Built...", "Designed..." in a row). Vary openings: start some with context ("When X, ..."), some with the action, some with the outcome. (2) Overly even sentence length — add a few short punchy lines and a few longer ones. (3) Remove any remaining template phrases: "demonstrated ability to", "in order to", "due to the fact that". (4) Avoid listy, parallel phrasing; break the rhythm occasionally so it feels natural.
- Do not introduce {STYLE_NO_OXFORD_DASHES} or first person in bullets. Keep format: blank lines between sections only; no blank lines between bullets under the same job; no horizontal rules. Job descriptions must remain bullets only, no paragraphs.
- Output only the resume text. No commentary."""



async def generate_resume(
    job_description: str,
    job_folder: str | Path,
    config: Settings,
    base_resume: str,
    context_files: Dict[str, str],
    combined_context: Optional[str] = None,
    ai_clients: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[..., Awaitable[None]]] = None,
    ollama_model_override: Optional[str] = None,
    use_deepseek: bool = True,
    model_sequence: Optional[list[str]] = None,
    parallel_flags: Optional[list[bool]] = None,
    pipeline_steps: Optional[list[dict[str, Any]]] = None,
    parsed_job_json: Optional[str] = None,
) -> tuple[str, list[str]]:
    """
    Run multi-pass resume generation. Saves all intermediate files to job_folder.
    Returns (final_resume_text, suggestions_list).
    suggestions_list contains outputs from review/suggestion steps (not part of the final document).
    """
    job_folder = Path(job_folder)
    job_folder.mkdir(parents=True, exist_ok=True)

    def _normalize(mid: str) -> str:
        mid = (mid or "").strip()
        if not mid:
            return mid
        # Treat known prefixes as already-normalized model ids.
        # IMPORTANT: do NOT wrap "deepseek:*" as an Ollama model (it is a cloud provider).
        if (
            mid.startswith("ollama:")
            or mid.startswith("gguf:")
            or mid.startswith("hf:")
            or mid.startswith("deepseek:")
        ):
            return mid
        # Back-compat: treat as an Ollama model name
        return f"ollama:{mid}"

    # Pipeline override (dynamic steps)
    if pipeline_steps and isinstance(pipeline_steps, list):
        # Use Pass 1 model to help prep context (summarization/truncation)
        first_model = str((pipeline_steps[0] or {}).get("model_id") or "").strip()
        if not first_model:
            raise ValueError("pipeline_steps[0].model_id is required")
        _ctx = combined_context
        if not _ctx:
            ctx_client, _ = get_ai_client_for_model_id(first_model, config)
            _ctx = await prepare_context(
                base_resume, context_files, job_description, ctx_client,
                get_context_window(first_model, config),
                model_id=first_model,
            )

        vars = {
            "context": _ctx,
            "base_resume": base_resume[:40000],
            "job_description": job_description[:15000],
            "parsed_job": (parsed_job_json or "").strip(),
            "input": "",
        }
        steps = steps_from_pipeline_config(pipeline_steps, RESUME_SYSTEM, 8192)
        if not steps:
            raise ValueError("pipeline_steps is empty")
        res = await run_pipeline(config=config, steps=steps, vars=vars, progress_callback=progress_callback)
        # Save clean Markdown outputs + sidecar metadata (no frontmatter).
        for i, out in enumerate(res["outputs"], start=1):
            md_path = job_folder / f"resume_step{i}.md"
            save_text_file(md_path, out)
            meta = {
                "job_id": job_folder.name,
                "type": "resume",
                "step": i,
                "is_review": steps[i - 1].is_review,
                "model": (res.get("models") or [None] * len(res["outputs"]))[i - 1],
                "timestamp": time.time(),
            }
            save_text_file(md_path.with_suffix(".meta.json"), json.dumps(meta, indent=2))
        final_path = job_folder / "resume_final.md"
        final_text = (res["final"] or "").strip()
        if not final_text:
            logger.warning("Resume pipeline returned empty content; writing placeholder to %s", final_path)
            final_text = _RESUME_EMPTY_PLACEHOLDER
        final_text = sanitise_resume_style(final_text)
        if getattr(config, "PERSONAL_FILL_ENABLED", True):
            from app.personal_fill import load_personal_vars, apply_personal_fill
            pvars = load_personal_vars(config)
            if pvars:
                final_text = apply_personal_fill(final_text, pvars)
        save_text_file(final_path, final_text)
        save_text_file(
            final_path.with_suffix(".meta.json"),
            json.dumps({"job_id": job_folder.name, "type": "resume", "final": True, "timestamp": time.time()}, indent=2),
        )
        return final_text, res.get("suggestions") or []

    # Determine pass models (1..N)
    default_seq = getattr(config, "DEFAULT_MODEL_SEQUENCE", []) or []
    if model_sequence and len(model_sequence) >= 1 and all(isinstance(x, str) and x.strip() for x in model_sequence):
        pass_models = [_normalize(m) for m in model_sequence]
    elif not use_deepseek and len(default_seq) == 4:
        pass_models = [_normalize(m) for m in default_seq]
    else:
        # Build a properly-prefixed 4-pass sequence so the pipeline engine always handles it.
        base_ollama_id = _normalize(ollama_model_override or getattr(config, "DEFAULT_OLLAMA_MODEL", config.OLLAMA_MODEL))
        deepseek_id = f"deepseek:{getattr(config, 'DEFAULT_DEEPSEEK_MODEL', 'deepseek-chat')}"
        pass_models = [base_ollama_id, deepseek_id, base_ollama_id, deepseek_id]

    # If we are given a full local sequence (model ids), run per-pass with the selected runtime.
    is_model_id_sequence = all(
        isinstance(x, str)
        and (
            x.startswith("ollama:")
            or x.startswith("gguf:")
            or x.startswith("hf:")
            or x.startswith("deepseek:")
        )
        for x in pass_models
    )
    if is_model_id_sequence:
        pass1_client, pass1_name = get_ai_client_for_model_id(pass_models[0], config)
        _ctx = combined_context
        if not _ctx:
            _ctx = await prepare_context(
                base_resume, context_files, job_description, pass1_client,
                get_context_window(pass_models[0], config),
                model_id=pass_models[0],
            )

        # Pipeline: variable number of steps, with template variables.
        vars = {
            "context": _ctx,
            "base_resume": base_resume[:40000],
            "job_description": job_description[:15000],
            "parsed_job": (parsed_job_json or "").strip(),
            "input": "",
        }

        steps: list[PipelineStep] = []
        n = len(pass_models)

        def _ctx_param(override_ctx: int) -> dict:
            return {"num_ctx": override_ctx} if override_ctx > 0 else {}

        def _draft_step(model_id: str) -> PipelineStep:
            return PipelineStep(
                name="Draft",
                model_id=model_id,
                system_prompt=RESUME_SYSTEM,
                max_tokens=8192,
                params={
                    "temperature": getattr(config, "DRAFT_TEMPERATURE", 0.7),
                    **_ctx_param(getattr(config, "DRAFT_NUM_CTX", 0)),
                },
                prompt_template=(
                    "You have been given the full base resume and context (stories, achievements). Use them: include every relevant bullet and every story that matches the job requirements. Do not summarize, omit or shorten.\n\n"
                    "Do not add contact or name at the top; user adds their own. Order: 1) SUMMARY (required): " + SUMMARY_THREE_LINES + " tying the job to the candidate's experience. 2) **Technical Expertise:** (concise, 2–4 categories). 3) Experience: each job = **Role title** (bold) on one line — use {{JOB1_TITLE}}, {{JOB2_TITLE}}, {{JOB3_TITLE}} when provided, else from base resume; next line {{JOB1_COMPANY}} | {{JOB1_START}} – {{JOB1_END}} (pipe between company and dates); then bullets. 4) Education, Certifications.\n\n"
                    "Length: Aim for 2 full pages. Bullets: first job at least 10, second at least 6, third at least 6. Do not truncate below minimums; if over 2 pages, trim technical section or tighten wording only. Do not add anything not in the base resume or context.\n\n"
                    "Job requirements (extracted from posting):\n{{parsed_job}}\n\n"
                    "{{context}}\n\n"
                    "Write the full resume (start with summary paragraph — no contact/name at top; then rest):"
                ),
            )

        def _review_step(model_id: str) -> PipelineStep:
            return PipelineStep(
                name="Review",
                model_id=model_id,
                system_prompt=REVIEW_SYSTEM,
                max_tokens=2000,
                params={
                    "temperature": getattr(config, "REVIEW_TEMPERATURE", 0.2),
                    **_ctx_param(getattr(config, "REVIEW_NUM_CTX", 0)),
                },
                prompt_template=(
                    "Review the following resume draft against the original base resume, context rules and job description.\n"
                    "Identify any exaggerations, missing achievements or deviations from the rules.\n"
                    "Provide specific improvement suggestions only. Do not rewrite the resume.\n\n"
                    "Base Resume:\n{{base_resume}}\n\n"
                    "Job Description:\n{{job_description}}\n\n"
                    "Draft Resume:\n{{input}}\n\n"
                    "Suggestions:"
                ),
            )

        def _apply_step(model_id: str, draft_key: str, suggestions_key: str) -> PipelineStep:
            return PipelineStep(
                name="Apply suggestions",
                model_id=model_id,
                system_prompt=RESUME_SYSTEM,
                max_tokens=8192,
                params={
                    "temperature": getattr(config, "APPLY_TEMPERATURE", 0.4),
                },
                prompt_template=(
                    "Using the original base resume, job description and the following suggestions, produce an improved resume draft.\n"
                    "Keep the summary (" + SUMMARY_THREE_LINES + " at the top before Technical Expertise). Aim for 2 full pages. Bullets: first job at least 10, second at least 6, third at least 6. Do not truncate below minimums. Incorporate the suggestions.\n\n"
                    "Base Resume:\n{{base_resume}}\n\n"
                    "Job Description:\n{{job_description}}\n\n"
                    "Current Draft:\n{{" + draft_key + "}}\n\n"
                    "Suggestions:\n{{" + suggestions_key + "}}\n\n"
                    "Write the improved resume:"
                ),
            )

        def _refine_step(model_id: str) -> PipelineStep:
            return PipelineStep(
                name="Refine",
                model_id=model_id,
                system_prompt=RESUME_SYSTEM,
                max_tokens=8192,
                params={
                    "temperature": getattr(config, "REFINE_TEMPERATURE", 0.5),
                },
                prompt_template=(
                    "Revise the resume draft. At most 2 pages.\n"
                    "1) Ensure there is a short SUMMARY (" + SUMMARY_THREE_LINES + ") right after the name, before **Technical Expertise:**. If missing, add it.\n"
                    "2) Bullets: first job at least 10, second at least 6, third at least 6. Do not truncate below minimums. Aim for 2 full pages. If over 2 pages, trim technical detail or tighten wording only.\n"
                    "3) Preserve all existing content. No blank lines between bullets under the same job; one blank line before the next company. No Oxford commas (use 'A, B and C' not 'A, B, and C').\n\n"
                    "Base Resume:\n{{base_resume}}\n\n"
                    "Job Description:\n{{job_description}}\n\n"
                    "Current Draft:\n{{input}}\n\n"
                    "Output the revised resume only."
                ),
            )

        def _polish_step(model_id: str) -> PipelineStep:
            return PipelineStep(
                name="Final polish",
                model_id=model_id,
                system_prompt=FINAL_SYSTEM,
                max_tokens=8192,
                params={
                    "temperature": getattr(config, "POLISH_TEMPERATURE", 0.7),
                },
                prompt_template=(
                    "Final polish. Two requirements:\n\n"
                    "1) SUMMARY AT TOP: The resume MUST have a short summary (" + SUMMARY_THREE_LINES + ") right after the name, before **Technical Expertise:**. If missing, add it. Do not make it long.\n\n"
                    "2) LENGTH: Aim for 2 full pages. Bullets: first job at least 10, second at least 6, third at least 6. Do not truncate below minimums. If over 2 pages, trim and tighten wording only.\n\n"
                    "Also: Storytelling and strong verbs. No Oxford commas; no double dashes; no blank lines between bullets under the same job. Every claim must be in the base resume.\n\n"
                    "Resume to polish:\n{{input}}\n\n"
                    "Job Description (for context):\n{{job_description}}\n\n"
                    "Output the final resume only."
                ),
            )

        def _humanize_step(model_id: str) -> PipelineStep:
            return PipelineStep(
                name="Humanize",
                model_id=model_id,
                system_prompt=HUMANIZE_SYSTEM,
                max_tokens=8192,
                params={
                    "temperature": getattr(config, "HUMANIZE_TEMPERATURE", 0.6),
                },
                prompt_template=(
                    "Rewrite the following resume so it reads as if a human wrote it. Change only wording and rhythm; do not add, remove or alter any facts, bullets or sections.\n\n"
                    "Resume:\n{{input}}\n\n"
                    "Output the humanized resume only."
                ),
            )

        def _with_parallel(step, idx: int):
            """Return step with parallel_with_prev set from parallel_flags if provided."""
            if parallel_flags and idx < len(parallel_flags) and parallel_flags[idx]:
                # Only deepseek: steps may run in parallel — enforced in pipeline_engine too
                import dataclasses
                return dataclasses.replace(step, parallel_with_prev=True)
            return step

        humanize_enabled = getattr(config, "HUMANIZE_STEP", True)
        if n == 1:
            steps = [_with_parallel(_draft_step(pass_models[0]), 0)]
            if humanize_enabled:
                steps.append(_humanize_step(pass_models[0]))
        elif n == 4:
            # Keep the classic 4-pass behavior.
            steps = [
                _with_parallel(_draft_step(pass_models[0]), 0),
                _with_parallel(_review_step(pass_models[1]), 1),
                _with_parallel(_apply_step(pass_models[2], "step1", "step2"), 2),
                _with_parallel(_polish_step(pass_models[3]), 3),
            ]
            if humanize_enabled:
                steps.append(_humanize_step(pass_models[3]))
        else:
            steps.append(_with_parallel(_draft_step(pass_models[0]), 0))
            # If user provided 4+ steps, keep a review/apply pair early, then refine with the remaining steps.
            if n >= 4:
                steps.append(_with_parallel(_review_step(pass_models[1]), 1))
                steps.append(_with_parallel(_apply_step(pass_models[2], "step1", "step2"), 2))
                # Remaining middle models (excluding last polish model).
                for i, mid in enumerate(pass_models[3:-1], start=3):
                    steps.append(_with_parallel(_refine_step(mid), i))
                steps.append(_with_parallel(_polish_step(pass_models[-1]), n - 1))
            else:
                # For 2-3 steps: draft -> refine(s) -> polish.
                for i, mid in enumerate(pass_models[1:-1], start=1):
                    steps.append(_with_parallel(_refine_step(mid), i))
                steps.append(_with_parallel(_polish_step(pass_models[-1]), n - 1))
            if humanize_enabled:
                steps.append(_humanize_step(pass_models[-1]))
        res = await run_pipeline(config=config, steps=steps, vars=vars, progress_callback=progress_callback)
        outs = res["outputs"]
        if outs:
            for i, out in enumerate(outs, start=1):
                save_text_file(job_folder / f"resume_step{i}.md", out)
        final_resume = (res["final"] or "").strip()
        if not final_resume:
            logger.warning("Resume pipeline returned empty content; writing placeholder to %s", job_folder / "resume_final.md")
            final_resume = _RESUME_EMPTY_PLACEHOLDER
        final_resume = sanitise_resume_style(final_resume)
        if getattr(config, "PERSONAL_FILL_ENABLED", True):
            from app.personal_fill import load_personal_vars, apply_personal_fill
            pvars = load_personal_vars(config)
            if pvars:
                final_resume = apply_personal_fill(final_resume, pvars)
        save_text_file(job_folder / "resume_final.md", final_resume)
        save_text_file(
            job_folder / "resume_pipeline.meta.json",
            json.dumps(
                {
                    "job_id": job_folder.name,
                    "type": "resume",
                    "timestamp": time.time(),
                    "models": res.get("models"),
                },
                indent=2,
            ),
        )
        return final_resume, res.get("suggestions") or []

    # Safety net: should not be reached because all model ID branches above produce properly-prefixed
    # IDs that route through the pipeline engine. If somehow reached, raise a clear error.
    raise RuntimeError(
        f"Could not resolve a valid model sequence for resume generation. "
        f"Models: {pass_models!r}. "
        f"Set DEFAULT_MODEL_SEQUENCE in .env or configure the Sequencer."
    )
