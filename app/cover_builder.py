"""Four-pass cover letter generation with Ollama and DeepSeek."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional

from app.ai_client import get_ai_client_for_model_id
from app.config import Settings, get_context_window
from app.context_manager import prepare_context
from app.pipeline_engine import PipelineStep, run_pipeline, steps_from_pipeline_config
from app.prompt_snippets import BANNED_PHRASES_COVER, PLACEHOLDER_RULE, STYLE_NO_OXFORD_DASHES
from app.utils import save_text_file, sanitise_resume_style

logger = logging.getLogger(__name__)
_COVER_EMPTY_PLACEHOLDER = "# Cover letter\n\n(Pipeline returned no content for this run.)"

COVER_SYSTEM = f"""You are a cover letter writer. Output must sound like a real person wrote it — not a language model. Follow every rule below.

PLACEHOLDERS: {PLACEHOLDER_RULE}

HONESTY: Every claim must be traceable to the base resume or context. Never invent skills, tools, certs, roles or outcomes. If in doubt, leave it out or use the exact wording from the base resume.

ALIGNMENT: Use the job's requirements and keywords to choose which of your experiences and stories to highlight. Match their language where you have real experience—do not add anything not in your materials. Enrich by phrasing what you actually did so the letter speaks directly to the role.

NO PERCENTAGES. Do not write "reduced by 40%", "improved 30%", or any percentage in the letter. Use words only: significantly, greatly, substantially, noticeably. Exception: exact figure explicitly in the source.

BUZZWORDS — replace, do not use: {BANNED_PHRASES_COVER} Contractions are fine: I've, didn't, it's. Colleague test: would you say this to the hiring manager over coffee? If not, rewrite.

STYLE: {STYLE_NO_OXFORD_DASHES} One page, three to four paragraphs. Output only the letter text.

HUMAN-LIKE: Vary sentence length; mix short punchy sentences with longer ones. Do not sound like a template. Write like you are talking to them, not giving a speech.

CONTENT: Open with a specific reason for this role and company. Name tools: Splunk not "SIEM", Nessus not "vulnerability scanner". Include one concrete achievement relevant to the job. Use "I"."""

COVER_REVIEW_SYSTEM = f"""You are a strict cover letter editor. Review the draft against the base resume and the writing rules. Do not rewrite the letter. Output only a bullet list of specific, actionable suggestions.

CHECK FOR:
1. Banned phrases — flag and replace with plain English: {BANNED_PHRASES_COVER} Never suggest adding percentages.
2. Generic opening — if the first sentence could belong to any letter, suggest a specific opener tied to this company or role.
3. Missing specifics — flag claims with no tool name or scale. Do not suggest adding percentages.
4. Hallucination — flag any skill, cert, tool or achievement not in the base resume. Flag invented person, school or company names; they should be from base resume or use placeholders [YOUR NAME], [COLLEGE NAME], [COMPANY NAME]. Flag invented numbers or percentages; suggest qualitative wording instead.
5. Missed alignment — note job requirements not addressed but available in the base resume.
6. Style — flag {STYLE_NO_OXFORD_DASHES}
7. Sentence variety — flag if two or more consecutive sentences are the same length.
8. Length — flag if over one page or under 200 words.

Be specific. Cite the exact phrase. Never suggest adding percentages."""

COVER_FINAL_SYSTEM = f"""Final-pass cover letter editor. Polish the draft. Output only the final letter text.

PLACEHOLDERS: {PLACEHOLDER_RULE}

HONESTY: Do not add or imply any skill, tool, cert or outcome not in the base resume. Every claim must be traceable to the source.

RULES: (1) REDUCE PERCENTAGES — Remove any specific percentage (e.g. "30%", "reduced by 40%"); replace with generic wording: significantly, greatly, substantially. (2) Replace buzzwords per {BANNED_PHRASES_COVER} (3) Vary sentence length; if two in a row are same length, rewrite one. (4) Read aloud; if you would not say it to the hiring manager, simplify. (5) First sentence specific to this company or role. (6) End with a clear call to action. (7) {STYLE_NO_OXFORD_DASHES} (8) Do not invent. One page only."""


COVER_HUMANIZE_SYSTEM = f"""You are a humanizing editor. Your only job is to make the letter read as if a real person wrote it in one sitting — not a language model.

RULES:
- Do NOT add, remove or change any facts, skills, achievements or claims. Preserve every point. Only change wording and rhythm. {PLACEHOLDER_RULE} Keep [YOUR NAME], [COLLEGE NAME], [COMPANY NAME] etc. as ALL CAPS placeholders; do not replace with invented names.
- REDUCE PERCENTAGES — If any sentence still contains a specific percentage, replace it with generic wording (significantly, substantially, greatly).
- Fix AI-like patterns: (1) Overly even sentence length — add a few short punchy sentences and a few longer ones. (2) Template openers like "I am writing to express..." or "I am excited to apply..." — replace with something specific and natural. (3) Listy or parallel phrasing; break the rhythm so it feels like a real letter. (4) Remove any remaining "in order to", "due to the fact that".
- Keep format: {STYLE_NO_OXFORD_DASHES} One page. Use "I" where appropriate.
- Output only the cover letter text. No commentary."""


async def generate_cover_letter(
    job_description: str,
    job_title: str,
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
    Run multi-pass cover letter generation. Saves intermediate files to job_folder.
    Returns (final_cover_text, suggestions_list).
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
        return f"ollama:{mid}"

    # Pipeline override (dynamic steps)
    if pipeline_steps and isinstance(pipeline_steps, list):
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
            "job_title": job_title,
            "input": "",
        }
        steps = steps_from_pipeline_config(pipeline_steps, COVER_SYSTEM, 4096)
        if not steps:
            raise ValueError("pipeline_steps is empty")
        res = await run_pipeline(config=config, steps=steps, vars=vars, progress_callback=progress_callback)
        for i, out in enumerate(res["outputs"], start=1):
            md_path = job_folder / f"cover_step{i}.md"
            save_text_file(md_path, out)
            meta = {
                "job_id": job_folder.name,
                "type": "cover",
                "step": i,
                "is_review": steps[i - 1].is_review,
                "model": (res.get("models") or [None] * len(res["outputs"]))[i - 1],
                "timestamp": time.time(),
            }
            save_text_file(md_path.with_suffix(".meta.json"), json.dumps(meta, indent=2))
        final_path = job_folder / "cover_final.md"
        final_text = (res["final"] or "").strip()
        if not final_text:
            logger.warning("Cover letter pipeline returned empty content; writing placeholder to %s", final_path)
            final_text = _COVER_EMPTY_PLACEHOLDER
        final_text = sanitise_resume_style(final_text)
        if getattr(config, "PERSONAL_FILL_ENABLED", True):
            from app.personal_fill import load_personal_vars, apply_personal_fill
            pvars = load_personal_vars(config)
            if pvars:
                final_text = apply_personal_fill(final_text, pvars)
        save_text_file(final_path, final_text)
        save_text_file(
            final_path.with_suffix(".meta.json"),
            json.dumps({"job_id": job_folder.name, "type": "cover", "final": True, "timestamp": time.time()}, indent=2),
        )
        return final_text, res.get("suggestions") or []

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

        vars = {
            "context": _ctx,
            "base_resume": base_resume[:40000],
            "job_description": job_description[:15000],
            "parsed_job": (parsed_job_json or "").strip(),
            "job_title": job_title,
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
                system_prompt=COVER_SYSTEM,
                max_tokens=4096,
                params={
                    "temperature": getattr(config, "DRAFT_TEMPERATURE", 0.7),
                    **_ctx_param(getattr(config, "DRAFT_NUM_CTX", 0)),
                },
                prompt_template=(
                    "Using the context and job details below, write a tailored cover letter for this job.\n"
                    "Use the job's requirements and keywords to choose which of your experiences (from base resume and context/stories) to highlight. Do not add anything not in your materials—only enrich by matching your real experience to their language.\n\n"
                    "Job Title: {{job_title}}\n\n"
                    "Job requirements (extracted from posting):\n{{parsed_job}}\n\n"
                    "{{context}}\n\n"
                    "Write the tailored cover letter:"
                ),
            )

        def _review_step(model_id: str) -> PipelineStep:
            return PipelineStep(
                name="Review",
                model_id=model_id,
                system_prompt=COVER_REVIEW_SYSTEM,
                max_tokens=1500,
                params={
                    "temperature": getattr(config, "REVIEW_TEMPERATURE", 0.2),
                    **_ctx_param(getattr(config, "REVIEW_NUM_CTX", 0)),
                },
                prompt_template=(
                    "Review the following cover letter draft against the original base resume, context rules and job description.\n"
                    "Identify any exaggerations, invented claims, style violations and missed opportunities to align with the job.\n"
                    "Provide specific improvement suggestions only. Do not rewrite the letter.\n\n"
                    "Base Resume:\n{{base_resume}}\n\n"
                    "Job Description:\n{{job_description}}\n\n"
                    "Draft Cover Letter:\n{{input}}\n\n"
                    "Suggestions:"
                ),
            )

        def _apply_step(model_id: str, draft_key: str, suggestions_key: str) -> PipelineStep:
            return PipelineStep(
                name="Apply suggestions",
                model_id=model_id,
                system_prompt=COVER_SYSTEM,
                max_tokens=4096,
                params={
                    "temperature": getattr(config, "APPLY_TEMPERATURE", 0.4),
                },
                prompt_template=(
                    "Using the original base resume, job description and the following suggestions, produce an improved cover letter draft.\n"
                    "Strictly follow all context rules. Do not exaggerate. Incorporate the suggestions.\n\n"
                    "Job Title: {{job_title}}\n\n"
                    "Base Resume:\n{{base_resume}}\n\n"
                    "Job Description:\n{{job_description}}\n\n"
                    "Current Draft:\n{{" + draft_key + "}}\n\n"
                    "Suggestions:\n{{" + suggestions_key + "}}\n\n"
                    "Write the improved cover letter:"
                ),
            )

        def _refine_step(model_id: str) -> PipelineStep:
            return PipelineStep(
                name="Refine",
                model_id=model_id,
                system_prompt=COVER_SYSTEM,
                max_tokens=4096,
                params={
                    "temperature": getattr(config, "REFINE_TEMPERATURE", 0.5),
                },
                prompt_template=(
                    "Revise the cover letter draft below to better fit the job description while strictly staying within base resume facts and style rules.\n"
                    "Reduce repetition. Keep it to one page.\n\n"
                    "Job Title: {{job_title}}\n\n"
                    "Base Resume:\n{{base_resume}}\n\n"
                    "Job Description:\n{{job_description}}\n\n"
                    "Current Draft:\n{{input}}\n\n"
                    "Output the revised cover letter only."
                ),
            )

        def _polish_step(model_id: str) -> PipelineStep:
            return PipelineStep(
                name="Final polish",
                model_id=model_id,
                system_prompt=COVER_FINAL_SYSTEM,
                max_tokens=4096,
                params={
                    "temperature": getattr(config, "POLISH_TEMPERATURE", 0.7),
                },
                prompt_template=(
                    "Perform a final polish on this cover letter. Focus on:\n"
                    "- Clear narrative and fit with the job description.\n"
                    "- Achievement emphasis and relevance. No Oxford commas, no double dashes.\n"
                    "- No exaggeration; every claim must be supported by the base resume.\n"
                    "- One page only.\n\n"
                    "Cover letter to polish:\n{{input}}\n\n"
                    "Job Description (for context):\n{{job_description}}\n\n"
                    "Output the final cover letter only."
                ),
            )

        def _humanize_step(model_id: str) -> PipelineStep:
            return PipelineStep(
                name="Humanize",
                model_id=model_id,
                system_prompt=COVER_HUMANIZE_SYSTEM,
                max_tokens=4096,
                params={
                    "temperature": getattr(config, "HUMANIZE_TEMPERATURE", 0.6),
                },
                prompt_template=(
                    "Rewrite the following cover letter so it reads as if a human wrote it. Change only wording and rhythm; do not add, remove or alter any facts or claims.\n\n"
                    "Cover letter:\n{{input}}\n\n"
                    "Output the humanized cover letter only."
                ),
            )

        def _with_parallel(step, idx: int):
            if parallel_flags and idx < len(parallel_flags) and parallel_flags[idx]:
                import dataclasses
                return dataclasses.replace(step, parallel_with_prev=True)
            return step

        humanize_enabled = getattr(config, "HUMANIZE_STEP", True)
        if n == 1:
            steps = [_with_parallel(_draft_step(pass_models[0]), 0)]
            if humanize_enabled:
                steps.append(_humanize_step(pass_models[0]))
        elif n == 4:
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
            if n >= 4:
                steps.append(_with_parallel(_review_step(pass_models[1]), 1))
                steps.append(_with_parallel(_apply_step(pass_models[2], "step1", "step2"), 2))
                for i, mid in enumerate(pass_models[3:-1], start=3):
                    steps.append(_with_parallel(_refine_step(mid), i))
                steps.append(_with_parallel(_polish_step(pass_models[-1]), n - 1))
            else:
                for i, mid in enumerate(pass_models[1:-1], start=1):
                    steps.append(_with_parallel(_refine_step(mid), i))
                steps.append(_with_parallel(_polish_step(pass_models[-1]), n - 1))
            if humanize_enabled:
                steps.append(_humanize_step(pass_models[-1]))
        res = await run_pipeline(config=config, steps=steps, vars=vars, progress_callback=progress_callback)
        outs = res["outputs"]
        if outs:
            for i, out in enumerate(outs, start=1):
                save_text_file(job_folder / f"cover_step{i}.md", out)
        final_cover = (res["final"] or "").strip()
        if not final_cover:
            logger.warning("Cover letter pipeline returned empty content; writing placeholder to %s", job_folder / "cover_final.md")
            final_cover = _COVER_EMPTY_PLACEHOLDER
        final_cover = sanitise_resume_style(final_cover)
        if getattr(config, "PERSONAL_FILL_ENABLED", True):
            from app.personal_fill import load_personal_vars, apply_personal_fill
            pvars = load_personal_vars(config)
            if pvars:
                final_cover = apply_personal_fill(final_cover, pvars)
        save_text_file(job_folder / "cover_final.md", final_cover)
        save_text_file(
            job_folder / "cover_pipeline.meta.json",
            json.dumps(
                {
                    "job_id": job_folder.name,
                    "type": "cover",
                    "timestamp": time.time(),
                    "models": res.get("models"),
                },
                indent=2,
            ),
        )
        return final_cover, res.get("suggestions") or []

    # Safety net: should not be reached because all model ID branches above produce properly-prefixed
    # IDs that route through the pipeline engine. If somehow reached, raise a clear error.
    raise RuntimeError(
        f"Could not resolve a valid model sequence for cover letter generation. "
        f"Models: {pass_models!r}. "
        f"Set DEFAULT_MODEL_SEQUENCE in .env or configure the Sequencer."
    )
