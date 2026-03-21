"""FastAPI application: routes and background job processing."""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from app.ai_client import get_ai_client_for_model_id
from app.config import get_context_window, get_settings
from app.interview_builder import generate_interview_prep, run_pass1, run_pass2
from app.context_manager import prepare_context
from app.cover_builder import generate_cover_letter
from app.docx_parser import extract_text_from_docx
from app.models import AnswerRequest, BatchStartRequest, JobResult, JobStatus, StartRequest
from app.model_manager import ModelManager
from app.model_registry import ModelRegistry
from app.hub_manager import HubManager
from app.history import HistoryStore, job_key_for_url
from app.hardware_monitor import get_hardware_stats
from app.job_parser import parse_job
from app.resume_locator import resolve_base_resume, write_selected
from app.resume_builder import generate_resume
from app.role_classifier import classify_role
from app.prompt_snippets import inject_snippets_into_pipeline
from app.spider import scrape_job
from app.utils import ensure_dir, load_context_files, extract_job_title_from_text, sanitise_filename, make_download_filename

# Playwright requires asyncio subprocess support on Windows. Some event loop policies
# (SelectorEventLoop) do not implement subprocesses and will raise NotImplementedError.
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())  # type: ignore[attr-defined]
    except Exception:
        pass

# In-memory job state (job_id -> dict)
jobs: Dict[str, Dict[str, Any]] = {}
# Optional job queue: process jobs sequentially (maxsize 10)
job_queue: asyncio.Queue = asyncio.Queue(maxsize=10)

# ---------------------------------------------------------------------------
# In-process caches — avoids re-reading unchanged files on every job
# ---------------------------------------------------------------------------
_context_files_cache: Dict[str, Any] | None = None
_context_files_cache_dir: str = ""
_resume_text_cache: Dict[str, tuple] = {}  # path -> (mtime, text)


def _get_context_files(context_dir: Path) -> Dict[str, Any]:
    """Return context files dict, re-reading from disk only when the directory path changes."""
    global _context_files_cache, _context_files_cache_dir
    key = str(context_dir)
    if _context_files_cache is None or _context_files_cache_dir != key:
        _context_files_cache = load_context_files(context_dir)
        _context_files_cache_dir = key
    return _context_files_cache


def _get_resume_text(path: Path) -> str:
    """Return extracted DOCX/TXT text, re-parsing only when the file's mtime changes."""
    key = str(path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = -1.0
    cached = _resume_text_cache.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    text = extract_text_from_docx(path)
    _resume_text_cache[key] = (mtime, text)
    return text


def _all_deepseek(steps: list | None) -> bool:
    """Return True only when every step in the list uses the DeepSeek API.

    Local Ollama pipelines use keep_alive=0 (evict_between_steps), so running
    two pipelines concurrently causes the model to be reloaded after every
    single step — significantly slower than sequential execution.  Parallelism
    is therefore only safe when all steps are stateless DeepSeek API calls.
    """
    if not steps:
        return False
    return all((s.get("model_id") or "").startswith("deepseek:") for s in steps)

app = FastAPI(title="Job Application Tailor", version="0.1.0")
config = get_settings()
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

LOG_PATH = Path(__file__).parent.parent / "app.log"
logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("jat")
history = HistoryStore(Path(__file__).parent.parent / "data" / "job_history.db")

#
# In-memory download state (download_id -> state)
# This is separate from jobs. Downloads must keep running even if the user navigates away.
#
downloads: Dict[str, Dict[str, Any]] = {}


def _get_download(download_id: str) -> Dict[str, Any]:
    st = downloads.get(download_id)
    if not isinstance(st, dict):
        raise HTTPException(status_code=404, detail="download_id not found")
    return st


def _public_download_state(st: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "download_id": st.get("download_id"),
        "kind": st.get("kind"),
        "label": st.get("label"),
        "status": st.get("status"),
        "error": st.get("error"),
        "total": st.get("total"),
        "completed": st.get("completed"),
        "started_at": st.get("started_at"),
        "updated_at": st.get("updated_at"),
        "done": bool(st.get("done")),
        "cancelled": bool(st.get("cancelled")),
        "path": st.get("path"),
    }


# Phrases that indicate the job posting is no longer available (checked in first 500 chars of scraped text).
_JOB_UNAVAILABLE_PHRASES = [
    "no longer available",
    "job you are trying to apply for is no longer available",
    "job has expired",
    "position has been filled",
    "this position is no longer open",
    "role has been filled",
]


def _is_job_unavailable(description: str) -> bool:
    """Return True if the description indicates the job posting is unavailable."""
    if not description or not description.strip():
        return False
    sample = (description.strip() + "\n")[:500].lower()
    return any(phrase.lower() in sample for phrase in _JOB_UNAVAILABLE_PHRASES)


def _job_folder(job_id: str) -> Path:
    """Return absolute path to this job's folder under data/jobs (relative to app root)."""
    jobs_base = Path(config.JOBS_DIR)
    if not jobs_base.is_absolute():
        jobs_base = Path(__file__).parent.parent / jobs_base
    path = (jobs_base / job_id).resolve()
    return path


def _pipelines_dir() -> Path:
    return Path(__file__).parent.parent / "data" / "pipelines"


def _pipeline_path(name: str) -> Path:
    fname = sanitise_filename(name).replace(" ", "_")
    if not fname.lower().endswith(".json"):
        fname = fname + ".json"
    return _pipelines_dir() / fname


async def job_queue_worker() -> None:
    """Process jobs from the queue one at a time."""
    while True:
        job_id: str | None = None
        url: str | None = None
        start_req: StartRequest | None = None
        got_item = False
        try:
            job_id, url, start_req = await job_queue.get()
            got_item = True
            job = jobs.get(job_id)
            if not job:
                continue
            stats_queue = job.get("stats_queue")
            if stats_queue:
                await stats_queue.put({"log": "Starting...", "queue_position": 1})
            logger.info("job=%s starting url=%s", job_id, url)
            # Wrap in a Task so it can be cancelled via /api/jobs/{id}/cancel
            task = asyncio.create_task(process_job(job_id, url, start_req, stats_queue))
            jobs[job_id]["_task"] = task
            try:
                await task
            except asyncio.CancelledError:
                logger.info("job=%s cancelled by user", job_id)
                jobs[job_id]["status"] = "cancelled"
                jobs[job_id]["progress"] = None
                jobs[job_id]["message"] = "Stopped by user"
                if stats_queue:
                    await stats_queue.put({"status": "cancelled", "message": "Stopped by user"})
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.exception("job_queue_worker error: %s", e)
            if job_id and job_id in jobs:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["message"] = str(e)
                if jobs[job_id].get("stats_queue"):
                    await jobs[job_id]["stats_queue"].put({"status": "error", "message": str(e)})
        finally:
            if got_item:
                try:
                    job_queue.task_done()
                except ValueError:
                    # Defensive: avoid noisy shutdown crashes if task accounting is off
                    # (e.g., due to reload/cancellation edge cases).
                    pass


async def process_job(
    job_id: str,
    url: str,
    start_req: StartRequest,
    stats_queue: asyncio.Queue,
) -> None:
    async def push_stats(evt: Dict[str, Any]) -> None:
        evt.setdefault("status", jobs[job_id]["status"])
        jobs[job_id]["stats"] = {**jobs[job_id].get("stats", {}), **evt}
        await stats_queue.put(dict(evt))

    job_folder = _job_folder(job_id)
    ensure_dir(job_folder)
    logger.info("job=%s output folder: %s", job_id, job_folder)
    try:
        logger.info("job=%s scrape begin", job_id)
        scraped = None
        if getattr(start_req, "job_description_override", None):
            job_desc = str(start_req.job_description_override)
            title = str(getattr(start_req, "job_title_override", "") or "") or extract_job_title_from_text(job_desc)
            jobs[job_id]["status"] = "generating"
            jobs[job_id]["progress"] = "Using provided job description..."
            await push_stats({"progress": "Using provided job description...", "log": "Using provided job description"})
        else:
            jobs[job_id]["status"] = "scraping"
            jobs[job_id]["progress"] = "Scraping job posting..."
            await push_stats({"progress": "Scraping job posting...", "log": "Scraping..."})
            scraped = await scrape_job(url, job_folder=job_folder, timeout_ms=config.SCRAPE_TIMEOUT * 1000)
            job_desc = scraped["description"]
            title = scraped.get("title") or extract_job_title_from_text(job_desc)
            scrape_source = str(scraped.get("scrape_source") or "unknown")
            scrape_meta = scraped.get("scrape_meta") if isinstance(scraped.get("scrape_meta"), dict) else {}
            jobs[job_id]["scrape_source"] = scrape_source
            jobs[job_id]["scrape_meta"] = scrape_meta
            if _is_job_unavailable(job_desc or ""):
                jobs[job_id]["status"] = "error"
                jobs[job_id]["message"] = "This job posting is no longer available."
                await push_stats({"status": "error", "message": jobs[job_id]["message"]})
                return
            # Avoid keeping large scrape artifacts in RAM. Persist to disk and keep a small pointer.
            try:
                p = (Path(job_folder) / "job_posting.txt").resolve()
                text = job_desc or ""
                if not text.strip():
                    logger.warning("job=%s scraped description is empty; job_posting.txt may be empty", job_id)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text, encoding="utf-8")
                jobs[job_id]["job_posting_file"] = p.name
            except Exception as e:
                logger.warning("job=%s failed to write job_posting.txt: %s", job_id, e)
            jobs[job_id].pop("scraped", None)
            scraped = None

        focus = (getattr(start_req, "one_up_focus", None) or "").strip()
        if focus:
            # Treated as instruction only, not to be echoed.
            job_desc = (job_desc or "") + "\n\nAdditional instruction (focus): " + focus

        min_desc_len = getattr(config, "MIN_JOB_DESCRIPTION_LENGTH", 0) or 0
        if min_desc_len > 0 and len((job_desc or "").strip()) < min_desc_len:
            jobs[job_id]["status"] = "error"
            scrape_source = str(jobs[job_id].get("scrape_source") or "unknown")
            scrape_meta = jobs[job_id].get("scrape_meta") if isinstance(jobs[job_id].get("scrape_meta"), dict) else {}
            char_count = int(scrape_meta.get("char_count") or len((job_desc or "").strip()))
            method = str(scrape_meta.get("method") or scrape_source)
            attempted = scrape_meta.get("attempted_sources")
            attempted_txt = ""
            if isinstance(attempted, list) and attempted:
                attempted_txt = f"; attempted: {', '.join(str(x) for x in attempted)}"
            jobs[job_id]["message"] = (
                f"Job description is too short or missing (source: {method}, chars: {char_count}{attempted_txt})."
            )
            logger.warning(
                "job=%s short description: source=%s method=%s chars=%s attempted=%s",
                job_id, scrape_source, method, char_count, attempted,
            )
            await push_stats({"status": "error", "message": jobs[job_id]["message"]})
            return

        jobs[job_id]["job_title"] = title
        jobs[job_id]["job_key"] = job_key_for_url(url)

        jobs[job_id]["status"] = "generating"
        logger.info("job=%s generate begin", job_id)
        context_dir = Path(config.CONTEXT_DIR)
        if not context_dir.is_absolute():
            context_dir = Path(__file__).parent.parent / context_dir
        context_files = _get_context_files(context_dir)

        # Optional pipeline preset (server-side)
        pipeline = None
        if getattr(start_req, "pipeline_preset", None):
            try:
                p = _pipeline_path(str(start_req.pipeline_preset))
                pipeline = json.loads(p.read_text(encoding="utf-8"))
                pipeline = inject_snippets_into_pipeline(pipeline)
            except Exception:
                pipeline = None

        ollama_model = getattr(start_req, "ollama_model", None) or getattr(config, "DEFAULT_OLLAMA_MODEL", config.OLLAMA_MODEL)
        use_deepseek = getattr(start_req, "use_deepseek", None)
        if use_deepseek is None:
            use_deepseek = getattr(config, "USE_DEEPSEEK", True)
        model_sequence = getattr(start_req, "model_sequence", None)
        if model_sequence and not isinstance(model_sequence, list):
            model_sequence = None
        parallel_flags = getattr(start_req, "parallel_flags", None)
        if parallel_flags and not isinstance(parallel_flags, list):
            parallel_flags = None
        # Attach parallel_with_prev to each step in the pipeline steps lists when flags are provided.
        if parallel_flags and isinstance(model_sequence, list):
            # Flags are applied positionally; extra flags are ignored, missing flags default to False.
            model_sequence = list(model_sequence)  # ensure mutable copy
        resume_steps = (pipeline.get("resume_steps") if isinstance(pipeline, dict) else None)
        cover_steps = (pipeline.get("cover_steps") if isinstance(pipeline, dict) else None)

        # Determine step counts early so the shared parser can report accurate totals.
        _p_steps = resume_steps if isinstance(resume_steps, list) else None
        _c_steps = cover_steps if isinstance(cover_steps, list) else None
        humanize_extra = 1 if getattr(config, "HUMANIZE_STEP", True) else 0
        if _p_steps:
            resume_n = len(_p_steps)
        elif model_sequence:
            resume_n = len(model_sequence) + humanize_extra
        else:
            resume_n = 4 + humanize_extra
        if _c_steps:
            cover_n = len(_c_steps)
        elif model_sequence:
            cover_n = len(model_sequence) + humanize_extra
        else:
            cover_n = 4 + humanize_extra
        total_progress_steps = 1 + resume_n + cover_n

        # Shared Step 1: parse job description into structured requirements (used by both resume + cover prompts).
        parsed_job_json = None
        try:
            await push_stats({"stage": "starting", "log": "Step 1 — Parsing job requirements...", "pass": 1, "total_steps": total_progress_steps, "phase": "Parser"})
            parsed_job_json = await parse_job(job_desc, config)
            try:
                pp = (Path(job_folder) / "job_parsed.json").resolve()
                pp.write_text(parsed_job_json or "{}", encoding="utf-8")
            except Exception as e:
                logger.warning("job=%s failed to write job_parsed.json: %s", job_id, e)
            await push_stats({"stage": "complete", "log": "Step 1 done — Parsed job requirements", "pass": 1, "total_steps": total_progress_steps, "phase": "Parser"})
        except Exception as e:
            # Parsing is helpful but should not block generation.
            parsed_job_json = None
            await push_stats({"stage": "skipped", "log": f"Step 1 skipped — parser failed: {type(e).__name__}: {e}", "pass": 1, "total_steps": total_progress_steps, "phase": "Parser"})
            if getattr(config, "PIPELINE_FAIL_IF_PARSER_SKIPPED", False):
                raise

        # Load base resume: after parse so we can use role-based selection (leadership vs engineering vs combined).
        project_root = Path(__file__).parent.parent
        base_dir = project_root / "data" / "base_resume"
        info = resolve_base_resume(base_dir, project_root, config)
        base_resume: str
        try:
            if info.role_based and info.leadership_path and info.engineering_path:
                role = classify_role(title, job_desc, parsed_job_json)
                if role == "leadership":
                    base_resume = _get_resume_text(info.leadership_path)
                elif role == "engineering":
                    base_resume = _get_resume_text(info.engineering_path)
                else:
                    text_lead = _get_resume_text(info.leadership_path)
                    text_eng = _get_resume_text(info.engineering_path)
                    base_resume = "[LEADERSHIP EXPERIENCE]\n" + text_lead + "\n\n[ENGINEERING EXPERIENCE]\n" + text_eng
                    if len(base_resume) > 40000:
                        base_resume = base_resume[:40000]
            else:
                base_resume_path = jobs[job_id].get("base_resume_path")
                if not base_resume_path:
                    base_resume_path = Path(config.BASE_RESUME_PATH)
                    if not base_resume_path.is_absolute():
                        base_resume_path = project_root / base_resume_path
                base_resume = _get_resume_text(base_resume_path)
        except FileNotFoundError:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["message"] = (
                "Base resume file not found (leadership or engineering)."
                if (info.role_based and info.leadership_path and info.engineering_path)
                else "Base resume not found. Place resume.docx or resume.txt in data/base_resume/"
            )
            await push_stats({"status": "error", "message": jobs[job_id]["message"]})
            return

        # Prepare combined context ONCE and reuse for resume + cover (saves time).
        # Use the first resume step model to summarise/truncate context if needed.
        try:
            ctx_mid = None
            if isinstance(resume_steps, list) and resume_steps:
                ctx_mid = str((resume_steps[0] or {}).get("model_id") or "").strip() or None
            if not ctx_mid and isinstance(model_sequence, list) and model_sequence:
                ctx_mid = str(model_sequence[0] or "").strip() or None
            if not ctx_mid:
                # Back-compat: treat ollama_model as an Ollama name
                ctx_mid = str(ollama_model or "").strip() or None
            _ctx_model = ctx_mid or config.OLLAMA_FALLBACK_MODEL_ID
            ctx_client, _ = get_ai_client_for_model_id(_ctx_model, config)
            combined_context = await prepare_context(
                base_resume, context_files, job_desc, ctx_client,
                get_context_window(_ctx_model, config),
                model_id=_ctx_model,
            )
        except Exception as e:
            logger.warning("prepare_context failed, continuing without combined context: %s", e)
            combined_context = None

        async def _resume_progress(evt: Dict[str, Any]) -> None:
            evt = dict(evt)
            if "total_steps" in evt:
                evt["total_steps"] = total_progress_steps
            if "pass" in evt:
                evt["pass"] = 1 + int(evt["pass"])
            evt["phase"] = "Resume"
            await push_stats(evt)

        async def _cover_progress(evt: Dict[str, Any]) -> None:
            evt = dict(evt)
            if "total_steps" in evt:
                evt["total_steps"] = total_progress_steps
            if "pass" in evt:
                evt["pass"] = 1 + resume_n + int(evt["pass"])
            evt["phase"] = "Cover letter"
            await push_stats(evt)

        # Decide whether to run resume and cover letter concurrently.
        # Parallel is only safe for pure DeepSeek pipelines: DeepSeek calls are stateless
        # HTTP requests that can overlap freely.  Local Ollama pipelines use keep_alive=0
        # (evict_between_steps), which unloads the model from VRAM after every step.
        # Running two Ollama pipelines concurrently causes constant evict/reload cycles —
        # each step in one pipeline evicts the model just as the other needs it, turning a
        # 2-minute job into a 15+ minute hang.
        _use_parallel = (
            (_p_steps and _c_steps and _all_deepseek(_p_steps) and _all_deepseek(_c_steps))
            or (
                not (_p_steps or _c_steps)
                and (
                    use_deepseek is True
                    or (
                        model_sequence
                        and all((m or "").startswith("deepseek:") for m in model_sequence)
                    )
                )
            )
        )

        if _use_parallel:
            (final_resume, resume_suggestions), (final_cover, cover_suggestions) = await asyncio.gather(
                generate_resume(
                    job_desc, job_folder, config, base_resume, context_files, combined_context,
                    progress_callback=_resume_progress,
                    ollama_model_override=ollama_model,
                    use_deepseek=use_deepseek,
                    model_sequence=model_sequence,
                    parallel_flags=parallel_flags,
                    pipeline_steps=(_p_steps),
                    parsed_job_json=parsed_job_json,
                ),
                generate_cover_letter(
                    job_desc, title, job_folder, config, base_resume, context_files, combined_context,
                    progress_callback=_cover_progress,
                    ollama_model_override=ollama_model,
                    use_deepseek=use_deepseek,
                    model_sequence=model_sequence,
                    parallel_flags=parallel_flags,
                    pipeline_steps=(_c_steps),
                    parsed_job_json=parsed_job_json,
                ),
            )
        else:
            final_resume, resume_suggestions = await generate_resume(
                job_desc, job_folder, config, base_resume, context_files, combined_context,
                progress_callback=_resume_progress,
                ollama_model_override=ollama_model,
                use_deepseek=use_deepseek,
                model_sequence=model_sequence,
                parallel_flags=parallel_flags,
                pipeline_steps=(_p_steps),
                parsed_job_json=parsed_job_json,
            )
            final_cover, cover_suggestions = await generate_cover_letter(
                job_desc, title, job_folder, config, base_resume, context_files, combined_context,
                progress_callback=_cover_progress,
                ollama_model_override=ollama_model,
                use_deepseek=use_deepseek,
                model_sequence=model_sequence,
                parallel_flags=parallel_flags,
                pipeline_steps=(_c_steps),
                parsed_job_json=parsed_job_json,
            )

        jobs[job_id]["status"] = "complete"
        jobs[job_id]["progress"] = None
        jobs[job_id]["message"] = "Done"
        jobs[job_id]["resume"] = final_resume
        jobs[job_id]["cover_letter"] = final_cover
        # Artifact list: include user-facing outputs only (avoid exposing prompts)
        try:
            folder = Path(job_folder)
            jobs[job_id]["artifacts"] = sorted(
                    [
                        p.name
                        for p in folder.iterdir()
                        if p.is_file() and (p.suffix.lower() in {".md", ".png", ".pdf"})
                    ]
            )
        except Exception:
            jobs[job_id]["artifacts"] = []
        await push_stats({"status": "complete", "log": "Done"})
        # Persist history (best-effort).
        try:
            jk = str(jobs[job_id].get("job_key") or job_key_for_url(url))
            history.upsert_job(job_key=jk, url=url, job_title=title, job_description=(job_desc[:200000] if job_desc else None))
            history.add_generation(
                job_key=jk,
                run_id=job_id,
                focus=(focus or None),
                pipeline_preset=(getattr(start_req, "pipeline_preset", None) or None),
                model_sequence_json=(json.dumps(model_sequence) if model_sequence else None),
                resume_md=final_resume,
                cover_md=final_cover,
            )
            logger.info("job=%s saved to history key=%s", job_id, jk)
        except Exception as _he:
            logger.warning("job=%s history save failed: %s", job_id, _he)
        # Best-effort model eviction to keep long sessions stable.
        try:
            from app.llamacpp_client import evict_all_llama_cache

            evict_all_llama_cache()
        except Exception:
            pass
        try:
            from app.transformers_client import evict_all_transformers_cache

            evict_all_transformers_cache()
        except Exception:
            pass
        logger.info("job=%s complete", job_id)
    except Exception as e:
        logger.exception("job=%s failed: %s", job_id, e)
        jobs[job_id]["status"] = "error"
        jobs[job_id]["progress"] = None
        jobs[job_id]["message"] = str(e)
        await stats_queue.put({"status": "error", "message": str(e)})


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request})


@app.get("/manage-models", response_class=HTMLResponse)
async def manage_models_page(request: Request):
    return templates.TemplateResponse(request=request, name="models.html", context={"request": request})


@app.get("/jobs", response_class=HTMLResponse)
async def jobs_page(request: Request):
    return templates.TemplateResponse(request=request, name="jobs.html", context={"request": request})


@app.get("/sequencer", response_class=HTMLResponse)
async def sequencer_page(request: Request):
    return templates.TemplateResponse(request=request, name="sequencer.html", context={"request": request})


@app.get("/pipeline", response_class=HTMLResponse)
async def pipeline_page(request: Request):
    return templates.TemplateResponse(request=request, name="pipeline.html", context={"request": request})


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    return templates.TemplateResponse(request=request, name="history.html", context={"request": request})


@app.get("/job/{job_id}")
async def job_page(request: Request, job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={"request": request, "job_id": job_id, "job_title": jobs[job_id].get("job_title"), "job_url": jobs[job_id].get("url", "")},
    )


async def _check_gpu_status() -> dict:
    """
    Quick GPU readiness check.  Called at startup and via /api/gpu-check.
    Returns a dict with 'ok', 'warnings', and 'details' keys.
    """
    warnings: list[str] = []
    details: dict = {}

    # --- Check 1: Ollama GPU usage ---
    try:
        import subprocess
        result = subprocess.run(
            ["ollama", "ps"], capture_output=True, text=True, timeout=5
        )
        details["ollama_ps"] = result.stdout.strip()
        if result.stdout and "CPU" in result.stdout and "GPU" not in result.stdout:
            warnings.append(
                "Ollama appears to be running a model on CPU only. "
                "If a large model won't fit in VRAM, Ollama falls back to CPU. "
                "Check VRAM usage with: nvidia-smi"
            )
    except Exception as e:
        details["ollama_ps_error"] = str(e)

    # --- Check 2: llama-cpp-python CUDA support (GGUF readiness) ---
    # IMPORTANT: register torch CUDA DLL dirs BEFORE importing llama_cpp.
    # llamacpp_client.py does this at module level, but it is lazily imported.
    # We must call it here explicitly so the DLL search path is set before the
    # 'from llama_cpp import ...' below, otherwise Windows fails to resolve
    # llama.dll's dependency on cublas64_12.dll / ggml-cuda.dll.
    try:
        from app.llamacpp_client import _register_cuda_dll_dirs as _reg
        _reg()
    except Exception:
        pass

    llamacpp_ok = False
    try:
        from llama_cpp import llama_cpp as _lc
        fn = getattr(_lc, "llama_supports_gpu_offload", None)
        llamacpp_ok = bool(fn() if callable(fn) else False)
        details["llamacpp_gpu_offload"] = llamacpp_ok
        if not llamacpp_ok:
            warnings.append(
                "llama-cpp-python is installed but reports NO GPU offload support. "
                "This means GGUF models will run on CPU (extremely slow). "
                "Run python scripts/jat.py setup to reinstall dependencies."
            )
    except Exception as e:
        details["llamacpp_import_error"] = str(e)
        warnings.append(
            f"llama-cpp-python failed to load ({type(e).__name__}): {e}. "
            "GGUF models cannot run. Run python scripts/jat.py setup to reinstall dependencies."
        )

    # --- Check 3: NVIDIA GPU visible ---
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            details["nvidia_smi"] = r.stdout.strip()
        else:
            warnings.append("nvidia-smi not found or no NVIDIA GPU detected. Check GPU drivers.")
            details["nvidia_smi_error"] = r.stderr.strip()
    except FileNotFoundError:
        warnings.append("nvidia-smi not found — cannot verify GPU. Check that NVIDIA drivers are installed.")
    except Exception as e:
        details["nvidia_smi_error"] = str(e)

    ok = len(warnings) == 0
    # Expose key booleans at the top-level so the UI can make decisions.
    return {"ok": ok, "warnings": warnings, "details": details, "gguf_ready": bool(llamacpp_ok)}


@app.on_event("startup")
async def startup_event() -> None:
    asyncio.create_task(job_queue_worker())
    # Ensure jobs directory exists and log its path so users can verify where outputs are saved.
    jobs_base = Path(config.JOBS_DIR)
    if not jobs_base.is_absolute():
        jobs_base = Path(__file__).parent.parent / jobs_base
    jobs_base = jobs_base.resolve()
    ensure_dir(jobs_base)
    logger.info("Job outputs will be saved under: %s", jobs_base)
    if os.environ.get("JAT_SKIP_GPU_CHECK", "").strip().lower() in {"1", "true", "yes", "on"}:
        logger.info("Startup GPU check skipped (JAT_SKIP_GPU_CHECK=1).")
        return
    # Run GPU check at startup and log any issues so they appear in app.log.
    gpu_status = await _check_gpu_status()
    if gpu_status["ok"]:
        logger.info("GPU check passed: llama-cpp-python CUDA ready, Ollama on GPU.")
    else:
        for w in gpu_status["warnings"]:
            logger.warning("GPU WARNING: %s", w)
        logger.warning(
            "GPU issues detected at startup. Run python scripts/jat.py setup to repair runtime dependencies. "
            "Details: %s", gpu_status["details"]
        )


@app.get("/api/gpu-check")
async def gpu_check():
    """Return GPU readiness status. Use this to diagnose CPU fallback."""
    return await _check_gpu_status()


@app.get("/api/jobs-dir")
async def get_jobs_dir():
    """Return the absolute path where job outputs (resume, cover, artifacts) are saved."""
    jobs_base = Path(config.JOBS_DIR)
    if not jobs_base.is_absolute():
        jobs_base = Path(__file__).parent.parent / jobs_base
    return {"jobs_dir": str(jobs_base.resolve())}


@app.get("/api/context-config", response_model=dict)
async def get_context_config():
    """Return effective context window configuration for all LLMs (proves 32k setup)."""
    from app.config import Settings
    cfg = get_settings()
    windows = dict(Settings._MODEL_CONTEXT_WINDOWS)
    sample_models = [
        "ollama:glm-4.7-flash-neo-code",
        "ollama:qwen3:14b",
        cfg.OLLAMA_FALLBACK_MODEL_ID,
    ]
    effective = {mid: get_context_window(mid, cfg) for mid in sample_models}
    return {
        "OLLAMA_NUM_CTX": cfg.OLLAMA_NUM_CTX,
        "MAX_CONTEXT_TOKENS": cfg.MAX_CONTEXT_TOKENS,
        "per_model_windows": windows,
        "effective_for_sample_models": effective,
    }


@app.get("/base-resume", response_model=dict)
async def base_resume_status():
    project_root = Path(__file__).parent.parent
    base_dir = project_root / "data" / "base_resume"
    info = resolve_base_resume(base_dir, project_root, config)
    return {
        "status": info.status,
        "candidates": info.candidates,
        "selected": info.selected,
        "role_based": info.role_based,
        "leadership_path": str(info.leadership_path) if info.leadership_path else None,
        "engineering_path": str(info.engineering_path) if info.engineering_path else None,
    }


@app.get("/api/models/catalog", response_model=dict)
async def models_catalog():
    catalog_path = Path(config.MODELS_CATALOG_PATH)
    if not catalog_path.is_absolute():
        catalog_path = Path(__file__).parent.parent / catalog_path
    try:
        data = json.loads(catalog_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        data = []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load catalog: {e}")
    return {"models": data}


@app.get("/api/models/local", response_model=dict)
async def local_models():
    mgr = ModelManager(config)
    models = await mgr.list_local_models()
    return {"models": models}


@app.get("/api/models/installed", response_model=dict)
async def installed_models():
    """
    Provider-agnostic installed models list for UI + sequencer.
    Includes local runtime models and registry-installed GGUF models.
    """
    # Determine GGUF readiness once per request.
    # Register torch CUDA DLL dirs first so the llama_cpp import can find cublas64_12.dll.
    try:
        from app.llamacpp_client import _register_cuda_dll_dirs as _reg2
        _reg2()
    except Exception:
        pass
    gguf_ready = False
    gguf_reason = ""
    try:
        from llama_cpp import llama_cpp as _lc

        fn = getattr(_lc, "llama_supports_gpu_offload", None)
        gguf_ready = bool(fn() if callable(fn) else False)
        if not gguf_ready:
            gguf_reason = "llama.cpp GPU offload not available (CPU-only build)"
    except Exception as e:
        gguf_ready = False
        gguf_reason = f"llama-cpp-python unavailable: {type(e).__name__}: {e}"

    mgr = ModelManager(config)
    local = await mgr.list_local_models()
    registry = ModelRegistry(config)
    extra = [m.to_dict() for m in registry.list()]

    # Build a map by model id so we don't show duplicate "aliases" (e.g., registry display name vs ollama tag).
    by_id: dict[str, dict[str, Any]] = {}

    # Local Ollama models
    for m in local:
        name = str(m.get("name") or "").strip()
        if not name:
            continue
        mid = f"ollama:{name}"
        by_id[mid] = {
            "id": mid,
            "display_name": name,
            "runtime": "ollama",
            "source_icon": "●",
            # keep the real tag for delete/info actions in the UI
            "ollama_model": name,
            "size": m.get("size"),
            "modified_at": m.get("modified_at"),
            "digest": m.get("digest"),
        }

    # Registry models (GGUF/HF and any ollama-tuning entries). Merge into local when ids match.
    for it in extra:
        mid = str(it.get("id") or "").strip()
        if not mid:
            continue
        runtime = str(it.get("runtime") or "")
        it["available"] = True
        it["availability_reason"] = ""
        if runtime == "llamacpp" and not gguf_ready:
            it["available"] = False
            it["availability_reason"] = gguf_reason or "GGUF runtime not ready"

        if mid in by_id:
            # Overlay registry metadata (pretty name, params) but keep local fields like size + ollama_model.
            merged = dict(by_id[mid])
            merged.update({k: v for k, v in it.items() if v is not None})
            by_id[mid] = merged
        else:
            by_id[mid] = it

    installed = list(by_id.values())

    # DeepSeek "virtual" models (cloud) so they can be selected in sequencer/pipeline steps.
    # Re-read settings fresh so the key is detected even if .env was updated after startup.
    try:
        _fresh = get_settings()
        deepseek_key_ok = bool((_fresh.DEEPSEEK_API_KEY or "").strip())
    except Exception:
        deepseek_key_ok = bool((getattr(config, "DEEPSEEK_API_KEY", "") or "").strip())
    deepseek_reason = "" if deepseek_key_ok else "DEEPSEEK_API_KEY not set in .env"
    installed.extend(
        [
            {
                "id": "deepseek:deepseek-chat",
                "display_name": "DeepSeek Chat (cloud)",
                "runtime": "deepseek",
                "source_icon": "DS",
                "available": deepseek_key_ok,
                "availability_reason": deepseek_reason,
            },
            {
                "id": "deepseek:deepseek-reasoner",
                "display_name": "DeepSeek Reasoner (cloud)",
                "runtime": "deepseek",
                "source_icon": "DS",
                "available": deepseek_key_ok,
                "availability_reason": deepseek_reason,
            },
        ]
    )
    return {"models": installed}


@app.get("/api/pipelines", response_model=dict)
async def list_pipelines():
    d = _pipelines_dir()
    d.mkdir(parents=True, exist_ok=True)
    items = []
    for p in sorted(d.glob("*.json")):
        items.append(p.stem)
    return {"pipelines": items}


@app.get("/api/pipelines/{name}", response_model=dict)
async def get_pipeline(name: str):
    p = _pipeline_path(name)
    if not p.exists():
        raise HTTPException(status_code=404, detail="pipeline not found")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to read pipeline: {e}")


@app.post("/api/pipelines/{name}", response_model=dict)
async def save_pipeline(name: str, payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="payload must be an object")
    d = _pipelines_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = _pipeline_path(name)
    try:
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to save pipeline: {e}")
    return {"ok": True}


@app.get("/api/jobs", response_model=dict)
async def list_jobs():
    items: list[dict[str, Any]] = []
    for jid, j in jobs.items():
        if not isinstance(j, dict):
            continue
        if jid.startswith("__"):
            continue
        created_at = j.get("created_at")
        items.append(
            {
                "job_id": j.get("job_id") or jid,
                "status": j.get("status"),
                "job_title": j.get("job_title"),
                "message": j.get("message"),
                "url": j.get("url"),
                "queue_position": j.get("queue_position"),
                "created_at": created_at if isinstance(created_at, (int, float)) else 0,
            }
        )
    items.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return {"jobs": items}


@app.get("/api/history/jobs", response_model=dict)
async def history_jobs(limit: int = 20, offset: int = 0):
    return {"jobs": history.list_jobs(limit=limit, offset=offset), "total": history.count_jobs()}


@app.delete("/api/history/jobs", response_model=dict)
async def history_delete_jobs(payload: dict):
    keys = [str(k) for k in (payload.get("job_keys") or []) if k]
    if not keys:
        raise HTTPException(status_code=400, detail="job_keys required")
    deleted = history.delete_jobs(keys)
    return {"deleted": deleted}


@app.get("/api/history/{job_key}", response_model=dict)
async def history_job(job_key: str):
    j = history.get_job(job_key)
    if not j:
        raise HTTPException(status_code=404, detail="not found")
    return j


@app.get("/api/history/{job_key}/generations", response_model=dict)
async def history_generations(job_key: str):
    j = history.get_job(job_key)
    if not j:
        raise HTTPException(status_code=404, detail="not found")
    return {
        "job_key": job_key,
        "job_title": j.get("job_title"),
        "url": j.get("url"),
        "generations": history.list_generations(job_key),
    }


@app.get("/api/history/{job_key}/generation/{version}", response_model=dict)
async def history_generation_content(job_key: str, version: int):
    g = history.get_generation_by_version(job_key, version)
    if not g:
        raise HTTPException(status_code=404, detail="not found")
    return g


@app.post("/api/history/oneup", response_model=dict)
async def history_oneup(payload: dict):
    job_key = str((payload or {}).get("job_key") or "").strip()
    focus = str((payload or {}).get("focus") or "").strip()
    if not job_key or not focus:
        raise HTTPException(status_code=400, detail="job_key and focus are required")
    job = history.get_job(job_key)
    latest = history.get_latest_generation_content(job_key)
    if not job or not latest:
        raise HTTPException(status_code=404, detail="job_key not found")

    project_root = Path(__file__).parent.parent
    base_dir = project_root / "data" / "base_resume"
    info = resolve_base_resume(base_dir, project_root, config)
    if info.status != "ok":
        raise HTTPException(status_code=400, detail="base resume not configured")
    if not info.role_based and not info.selected_path:
        raise HTTPException(status_code=400, detail="base resume not configured")

    url = str(job.get("url") or "")
    job_desc = str(job.get("job_description") or "")
    job_title = str(job.get("job_title") or "")
    if not url or not job_desc:
        raise HTTPException(status_code=500, detail="history record missing url/job_description")

    req = StartRequest(
        url=url,
        pipeline_preset=(payload or {}).get("pipeline_preset") or latest.get("pipeline_preset"),
        model_sequence=(payload or {}).get("model_sequence"),
        use_deepseek=False,
        job_title_override=job_title,
        job_description_override=job_desc,
        one_up_focus=focus,
    )

    job_id = str(uuid.uuid4())[:8]
    stats_queue: asyncio.Queue = asyncio.Queue()
    queue_position = job_queue.qsize() + 1
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": None,
        "message": None,
        "question": None,
        "url": url,
        "created_at": time.time(),
        "resume": None,
        "cover_letter": None,
        "artifacts": [],
        "job_title": job_title or None,
        "stats_queue": stats_queue,
        "stats": {},
        "queue_position": queue_position,
        "base_resume_path": info.selected_path if not info.role_based else None,
    }
    await job_queue.put((job_id, url, req))
    await stats_queue.put({"status": "pending", "log": f"Queued (position {queue_position})", "queue_position": queue_position})
    return {"job_id": job_id}


# ---------------------------------------------------------------------------
# Interview prep routes
# ---------------------------------------------------------------------------

async def _run_interview_prep(job_id: str, job_key: str, job_desc: str, base_resume: str, context_text: str, model_sequence: list, stats_queue: asyncio.Queue, extra_context: str = "") -> None:
    """Background task: two-pass interview prep with real-time step progress via SSE."""
    import json as _json

    async def push(evt: dict) -> None:
        jobs[job_id]["stats"] = {**jobs[job_id].get("stats", {}), **evt}
        await stats_queue.put(dict(evt))

    reasoner_model = model_sequence[0] if model_sequence else config.OLLAMA_MODEL
    chat_model = model_sequence[1] if len(model_sequence) > 1 else reasoner_model

    jobs[job_id]["status"] = "generating"
    try:
        # ── Pass 1: Reasoner ──────────────────────────────────────────────
        await push({
            "status": "generating",
            "step": 1,
            "log": f"Pass 1 — Analyzing job requirements with {reasoner_model.split(':')[-1]}…",
        })

        async def on_chunk_pass1(text: str) -> None:
            await push({"chunk": text, "step": 1})

        analysis = await run_pass1(
            job_description=job_desc,
            base_resume=base_resume,
            context_text=context_text,
            model_id=reasoner_model,
            on_chunk=on_chunk_pass1,
            extra_context=extra_context,
        )

        # ── Pass 2: Chat / Polish ─────────────────────────────────────────
        # "clear" tells the frontend to wipe the viewer before the final pass streams in.
        await push({
            "status": "generating",
            "step": 2,
            "clear": True,
            "log": f"Pass 2 — Writing prep document with {chat_model.split(':')[-1]}…",
        })

        async def on_chunk_pass2(text: str) -> None:
            await push({"chunk": text, "step": 2})

        prep_md = await run_pass2(
            analysis=analysis,
            job_description=job_desc,
            context_text=context_text,
            model_id=chat_model,
            on_chunk=on_chunk_pass2,
        )

        # ── Store ─────────────────────────────────────────────────────────
        version = history.store_interview_prep(
            job_key=job_key,
            prep_md=prep_md,
            model_sequence_json=_json.dumps(model_sequence),
        )
        jobs[job_id]["status"] = "complete"
        jobs[job_id]["interview_prep_version"] = version
        await push({
            "status": "complete",
            "step": 2,
            "log": f"Interview prep ready (version {version})",
            "interview_prep_version": version,
        })
    except Exception as exc:
        logger.exception("interview_prep job=%s failed: %s", job_id, exc)
        jobs[job_id]["status"] = "error"
        jobs[job_id]["message"] = str(exc)
        await push({"status": "error", "message": str(exc)})


@app.post("/api/history/interview-prep", response_model=dict)
async def history_start_interview_prep(payload: dict):
    """Trigger async interview prep generation for a historical job."""
    job_key = str((payload or {}).get("job_key") or "").strip()
    model_sequence = (payload or {}).get("model_sequence") or []
    extra_context = str((payload or {}).get("extra_context") or "").strip()
    if not job_key:
        raise HTTPException(status_code=400, detail="job_key is required")

    job = history.get_job(job_key)
    if not job:
        raise HTTPException(status_code=404, detail="job_key not found")

    job_desc = str(job.get("job_description") or "").strip()
    if not job_desc:
        raise HTTPException(status_code=400, detail="No job description stored for this job; re-run the tailor first")

    # Load base resume text
    project_root = Path(__file__).parent.parent
    base_dir = project_root / "data" / "base_resume"
    info = resolve_base_resume(base_dir, project_root, config)
    base_resume_text = ""
    if info.selected_path and Path(info.selected_path).exists():
        try:
            base_resume_text = extract_text_from_docx(info.selected_path)
        except Exception:
            pass

    # Load context files
    context_dir = Path(config.CONTEXT_DIR)
    if not context_dir.is_absolute():
        context_dir = project_root / context_dir
    context_files = load_context_files(context_dir)
    context_text = "\n\n".join(f"### {k}\n{v}" for k, v in context_files.items() if v.strip())

    if not model_sequence:
        model_sequence = [config.OLLAMA_MODEL, config.OLLAMA_MODEL]

    job_id = str(uuid.uuid4())[:8]
    stats_queue: asyncio.Queue = asyncio.Queue()
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": None,
        "message": None,
        "url": str(job.get("url") or ""),
        "created_at": time.time(),
        "stats_queue": stats_queue,
        "stats": {},
    }
    await stats_queue.put({"status": "pending", "log": "Interview prep queued"})
    asyncio.create_task(
        _run_interview_prep(job_id, job_key, job_desc, base_resume_text, context_text, model_sequence, stats_queue, extra_context=extra_context)
    )
    return {"job_id": job_id}


@app.get("/api/history/{job_key}/interview-preps", response_model=dict)
async def history_list_interview_preps(job_key: str):
    """List all interview prep versions for a job."""
    j = history.get_job(job_key)
    if not j:
        raise HTTPException(status_code=404, detail="not found")
    return {
        "job_key": job_key,
        "job_title": j.get("job_title"),
        "preps": history.list_interview_preps(job_key),
    }


@app.get("/api/history/{job_key}/interview-prep/{version}", response_model=dict)
async def history_get_interview_prep(job_key: str, version: int):
    """Return the full content of a specific interview prep version."""
    p = history.get_interview_prep(job_key, version)
    if not p:
        raise HTTPException(status_code=404, detail="not found")
    return p


@app.post("/api/models/show", response_model=dict)
async def show_model(payload: dict):
    model = (payload or {}).get("model")
    verbose = bool((payload or {}).get("verbose", False))
    if not model:
        raise HTTPException(status_code=400, detail="model is required")
    mgr = ModelManager(config)
    return await mgr.show_model(model, verbose=verbose)


@app.post("/api/models/delete", response_model=dict)
async def delete_model(payload: dict):
    model = (payload or {}).get("model")
    if not model:
        raise HTTPException(status_code=400, detail="model is required")
    mgr = ModelManager(config)
    try:
        await mgr.delete_model(model)
        return {"ok": True}
    except Exception as e:
        # Surface a readable error to the UI
        raise HTTPException(status_code=500, detail=f"Failed to delete '{model}': {e}")


async def _pull_model_task(download_id: str, model: str) -> None:
    st = downloads.get(download_id)
    if not st:
        return
    mgr = ModelManager(config)
    try:
        st["status"] = "starting"
        async for evt in mgr.pull_events(model):
            if st.get("cancelled"):
                st["status"] = "cancelled"
                st["done"] = True
                return
            st["updated_at"] = time.time()
            st["status"] = str(evt.get("status") or "downloading")
            if evt.get("total") is not None:
                st["total"] = evt.get("total")
            if evt.get("completed") is not None:
                st["completed"] = evt.get("completed")
            if evt.get("error") is not None:
                st["error"] = str(evt.get("error"))
            if st["status"] in {"success", "error"}:
                st["done"] = True
                return
    except Exception as e:
        st["status"] = "error"
        st["error"] = str(e)
        st["updated_at"] = time.time()
        st["done"] = True
    finally:
        st["updated_at"] = time.time()
        st["done"] = bool(st.get("done"))


@app.post("/api/models/pull", response_model=dict)
async def pull_model(payload: dict):
    model = (payload or {}).get("model")
    if not model:
        raise HTTPException(status_code=400, detail="model is required")
    download_id = str(uuid.uuid4())[:8]
    downloads[download_id] = {
        "download_id": download_id,
        "kind": "ollama_pull",
        "label": str(model),
        "status": "queued",
        "error": None,
        "total": 0,
        "completed": 0,
        "started_at": time.time(),
        "updated_at": time.time(),
        "done": False,
        "cancelled": False,
        "path": None,
    }
    task = asyncio.create_task(_pull_model_task(download_id, model))
    downloads[download_id]["task"] = task
    return {"download_id": download_id}


async def _install_gguf_task(download_id: str, repo_id: str, filename: str, display_name: str | None) -> None:
    st = downloads.get(download_id)
    if not st:
        return
    hub = HubManager(config)
    reg = ModelRegistry(config)
    try:
        st["status"] = "starting"
        st["updated_at"] = time.time()
        async for evt in hub.download_gguf_stream(
            repo_id,
            filename,
            cancel_check=lambda: bool(st.get("cancelled")),
        ):
            st["updated_at"] = time.time()
            st["status"] = str(evt.get("status") or st.get("status") or "downloading")
            if evt.get("total") is not None:
                st["total"] = evt.get("total")
            if evt.get("completed") is not None:
                st["completed"] = evt.get("completed")
            if evt.get("error") is not None:
                st["error"] = str(evt.get("error"))

            if st["status"] == "cancelled":
                st["done"] = True
                return

            if st["status"] == "success":
                path = evt.get("path")
                if path:
                    st["path"] = str(path)
                    pending = jobs.get("__install_params__", {}).get(download_id) if isinstance(jobs.get("__install_params__"), dict) else None
                    reg.add_llamacpp(
                        display_name or f"{repo_id}:{filename}",
                        gguf_path=str(path),
                        source_icon="⬡",
                        llamacpp_params=(pending if isinstance(pending, dict) else None),
                    )
                st["done"] = True
                return
            if st["status"] == "error":
                st["done"] = True
                return
    except Exception as e:
        st["status"] = "error"
        st["error"] = str(e)
        st["done"] = True
    finally:
        # cleanup any pending param stash
        if isinstance(jobs.get("__install_params__"), dict):
            jobs["__install_params__"].pop(download_id, None)
        st["updated_at"] = time.time()
        st["done"] = bool(st.get("done"))


@app.post("/api/models/search", response_model=dict)
async def search_models(payload: dict):
    query = (payload or {}).get("query") or ""
    query = str(query).strip()
    if not query:
        return {"results": []}
    hub = HubManager(config)
    results = [r.to_dict() for r in hub.search(query, limit=12)]
    return {"results": results}


@app.post("/api/models/install", response_model=dict)
async def install_model(payload: dict):
    repo_id = (payload or {}).get("repo_id")
    filename = (payload or {}).get("filename")
    display_name = (payload or {}).get("display_name")
    llamacpp_params = (payload or {}).get("llamacpp_params")
    if not repo_id or not filename:
        raise HTTPException(status_code=400, detail="repo_id and filename are required")
    download_id = str(uuid.uuid4())[:8]
    downloads[download_id] = {
        "download_id": download_id,
        "kind": "gguf_install",
        "label": f"{repo_id} / {filename}",
        "status": "queued",
        "error": None,
        "total": 0,
        "completed": 0,
        "started_at": time.time(),
        "updated_at": time.time(),
        "done": False,
        "cancelled": False,
        "path": None,
    }
    # stash params until install registers in the end
    if llamacpp_params and isinstance(llamacpp_params, dict):
        if not isinstance(jobs.get("__install_params__"), dict):
            jobs["__install_params__"] = {}
        jobs["__install_params__"][download_id] = llamacpp_params
    task = asyncio.create_task(_install_gguf_task(download_id, str(repo_id), str(filename), str(display_name) if display_name else None))
    downloads[download_id]["task"] = task
    return {"download_id": download_id}


@app.post("/api/models/installed/update", response_model=dict)
async def update_installed_model(payload: dict):
    model_id = (payload or {}).get("model_id")
    params = (payload or {}).get("llamacpp_params")
    if not model_id or not isinstance(model_id, str):
        raise HTTPException(status_code=400, detail="model_id is required")
    if params is None or not isinstance(params, dict):
        raise HTTPException(status_code=400, detail="llamacpp_params is required")
    reg = ModelRegistry(config)
    ok = reg.update_llamacpp_params(model_id, params)
    if not ok:
        raise HTTPException(status_code=404, detail="installed model not found")
    return {"ok": True}


@app.post("/api/models/uninstall", response_model=dict)
async def uninstall_model(payload: dict):
    model_id = (payload or {}).get("model_id")
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")
    reg = ModelRegistry(config)
    # remove file if we can
    for m in reg.list():
        if m.id == model_id and m.gguf_path:
            try:
                Path(m.gguf_path).unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass
            break
    reg.remove(str(model_id))
    return {"ok": True}


@app.get("/api/downloads", response_model=dict)
async def list_downloads():
    # Return only active downloads (not done) + recently finished (last 10 minutes).
    now = time.time()
    items: list[dict[str, Any]] = []
    for st in downloads.values():
        if not isinstance(st, dict):
            continue
        done = bool(st.get("done"))
        updated = float(st.get("updated_at") or 0)
        if done and (now - updated) > 600:
            continue
        items.append(_public_download_state(st))
    items.sort(key=lambda x: float(x.get("updated_at") or 0), reverse=True)
    return {"downloads": items}


@app.get("/api/downloads/{download_id}", response_model=dict)
async def download_status(download_id: str):
    st = _get_download(download_id)
    return _public_download_state(st)


@app.post("/api/downloads/{download_id}/cancel", response_model=dict)
async def cancel_download(download_id: str):
    st = _get_download(download_id)
    st["cancelled"] = True
    st["status"] = "cancelled"
    st["updated_at"] = time.time()
    task = st.get("task")
    try:
        if task and hasattr(task, "cancel"):
            task.cancel()
    except Exception:
        pass
    st["done"] = True
    return {"ok": True}


@app.get("/api/downloads/stream/{download_id}")
async def download_stream(download_id: str, request: Request):
    st = _get_download(download_id)

    async def event_gen():
        last = None
        while True:
            if await request.is_disconnected():
                break
            pub = _public_download_state(st)
            data = json.dumps(pub)
            if data != last:
                last = data
                yield ServerSentEvent(data=data)
            if pub.get("done"):
                break
            await asyncio.sleep(0.5)

    return EventSourceResponse(event_gen())


@app.post("/base-resume/select", response_model=dict)
async def base_resume_select(payload: dict):
    filename = (payload or {}).get("filename")
    if not filename or not isinstance(filename, str):
        raise HTTPException(status_code=400, detail="filename is required")
    base_dir = Path(__file__).parent.parent / "data" / "base_resume"
    info = resolve_base_resume(base_dir)
    if filename not in info.candidates:
        raise HTTPException(status_code=400, detail="filename is not a valid candidate")
    write_selected(base_dir, filename)
    return {"ok": True}


@app.post("/start", response_model=dict)
async def start(req: StartRequest):
    project_root = Path(__file__).parent.parent
    base_dir = project_root / "data" / "base_resume"
    info = resolve_base_resume(base_dir, project_root, config)
    if info.status != "ok":
        raise HTTPException(status_code=400, detail={"base_resume": {"status": info.status, "candidates": info.candidates}})
    # When role_based, worker will choose resume by role; otherwise we need a selected path.
    if not info.role_based and not info.selected_path:
        raise HTTPException(status_code=400, detail={"base_resume": {"status": info.status, "candidates": info.candidates}})

    job_id = str(uuid.uuid4())[:8]
    stats_queue: asyncio.Queue = asyncio.Queue()
    queue_position = job_queue.qsize() + 1
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": None,
        "message": None,
        "question": None,
        "url": req.url,
        "created_at": time.time(),
        "resume": None,
        "cover_letter": None,
        "artifacts": [],
        "job_title": None,
        "stats_queue": stats_queue,
        "stats": {},
        "queue_position": queue_position,
        "base_resume_path": info.selected_path if not info.role_based else None,
    }
    await job_queue.put((job_id, req.url, req))
    await stats_queue.put({"status": "pending", "log": f"Queued (position {queue_position})", "queue_position": queue_position})
    return {"job_id": job_id}


@app.post("/start-batch", response_model=dict)
async def start_batch(req: BatchStartRequest):
    """Queue multiple job URLs at once. Returns list of job_ids in the same order."""
    project_root = Path(__file__).parent.parent
    base_dir = project_root / "data" / "base_resume"
    info = resolve_base_resume(base_dir, project_root, config)
    if info.status != "ok":
        raise HTTPException(status_code=400, detail={"base_resume": {"status": info.status, "candidates": info.candidates}})
    if not info.role_based and not info.selected_path:
        raise HTTPException(status_code=400, detail={"base_resume": {"status": info.status, "candidates": info.candidates}})

    urls = [u.strip() for u in (req.urls or []) if u.strip()]
    if not urls:
        raise HTTPException(status_code=400, detail="No URLs provided")

    job_ids = []
    for url in urls:
        job_id = str(uuid.uuid4())[:8]
        stats_queue: asyncio.Queue = asyncio.Queue()
        queue_position = job_queue.qsize() + 1
        single_req = StartRequest(
            url=url,
            ollama_model=req.ollama_model,
            model_sequence=req.model_sequence,
            parallel_flags=req.parallel_flags,
        )
        jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "progress": None,
            "message": None,
            "question": None,
            "url": url,
            "created_at": time.time(),
            "resume": None,
            "cover_letter": None,
            "artifacts": [],
            "job_title": None,
            "stats_queue": stats_queue,
            "stats": {},
            "queue_position": queue_position,
            "base_resume_path": info.selected_path if not info.role_based else None,
        }
        await job_queue.put((job_id, url, single_req))
        await stats_queue.put({"status": "pending", "log": f"Queued (position {queue_position})", "queue_position": queue_position})
        job_ids.append(job_id)

    return {"job_ids": job_ids}


@app.get("/models", response_model=dict)
async def list_models():
    return {
        "ollama": getattr(config, "AVAILABLE_OLLAMA_MODELS", [config.OLLAMA_MODEL]),
        "deepseek": [getattr(config, "DEFAULT_DEEPSEEK_MODEL", "deepseek-chat")],
    }


@app.get("/stream/{job_id}")
async def stream_events(request: Request, job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    queue = jobs[job_id].get("stats_queue")
    if not queue:
        raise HTTPException(status_code=404, detail="Job has no stream")

    async def event_generator():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=2.0)
                except asyncio.TimeoutError:
                    hw = await get_hardware_stats()
                    yield ServerSentEvent(data=json.dumps({"hardware": hw}))
                    continue
                data = json.dumps(event) if isinstance(event, dict) else json.dumps({"data": str(event)})
                yield ServerSentEvent(data=data)
                if event.get("status") in ("complete", "error", "cancelled"):
                    break
        except asyncio.CancelledError:
            pass

    return EventSourceResponse(event_generator())


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a running or queued job."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    status_val = job.get("status", "")
    if status_val in ("complete", "cancelled", "error"):
        return {"ok": True, "already": status_val}
    # Cancel the underlying asyncio task if it exists
    task = job.get("_task")
    if task and not task.done():
        task.cancel()
    # Immediately mark so the SSE stream terminates
    job["status"] = "cancelled"
    job["progress"] = None
    job["message"] = "Stopped by user"
    sq = job.get("stats_queue")
    if sq:
        try:
            await sq.put({"status": "cancelled", "message": "Stopped by user"})
        except Exception:
            pass
    logger.info("job=%s cancel requested", job_id)
    return {"ok": True}


@app.get("/status/{job_id}", response_model=JobStatus)
async def status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    j = jobs[job_id]
    return JobStatus(
        job_id=j["job_id"],
        status=j["status"],
        progress=j.get("progress"),
        message=j.get("message"),
        question=j.get("question"),
    )


@app.post("/answer")
async def answer(req: AnswerRequest):
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    # Optional: resume processing with answer; for now just store and could extend
    jobs[req.job_id]["answer"] = req.answer
    jobs[req.job_id]["status"] = "generating"
    jobs[req.job_id]["question"] = None
    return {"ok": True}


@app.get("/result/{job_id}", response_model=JobResult)
async def result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    j = jobs[job_id]
    folder = _job_folder(job_id)
    artifacts = [a for a in j.get("artifacts", []) if (folder / a).exists()]
    return JobResult(
        job_id=j["job_id"],
        resume=j.get("resume"),
        cover_letter=j.get("cover_letter"),
        artifacts=artifacts,
        job_title=j.get("job_title"),
    )


@app.get("/download/{job_id}/{filename}")
async def download(job_id: str, filename: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    folder = _job_folder(job_id)
    safe_name = sanitise_filename(filename)
    path = (folder / safe_name).resolve()
    folder_resolved = folder.resolve()
    try:
        path.relative_to(folder_resolved)
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid path")
    if not path.exists() or path.is_dir():
        raise HTTPException(status_code=404, detail="File not found")
    # Use job-specific download filename for resume and cover so browser gets a friendly name.
    response_filename = safe_name
    if safe_name == "resume_final.md":
        response_filename = sanitise_filename(make_download_filename(jobs[job_id].get("job_title"), "resume"))
    elif safe_name == "cover_final.md":
        response_filename = sanitise_filename(make_download_filename(jobs[job_id].get("job_title"), "cover_letter"))
    return FileResponse(path, filename=response_filename)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
