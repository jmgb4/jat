"""Dynamic multi-step pipeline execution."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from app.ai_client import get_ai_client_for_model_id
from app.config import Settings
from app.model_profiles import get_params_for_model, load_profiles
from app.vram_advisor import advise_ollama_step

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]


def _step_is_review(step_name: str, system_prompt: str) -> bool:
    """Heuristic: a step is a review/suggestion step if its name contains 'review'
    or its system prompt asks for a bullet list of suggestions."""
    if "review" in (step_name or "").lower():
        return True
    if "output only a bullet list" in (system_prompt or "").lower():
        return True
    return False


@dataclass(frozen=True)
class PipelineStep:
    name: str
    model_id: str
    system_prompt: str
    prompt_template: str
    max_tokens: Optional[int] = None
    params: dict[str, Any] = field(default_factory=dict)
    is_review: bool = False
    # When True this step runs concurrently with the previous sequential step's output as input.
    # Only honored for deepseek: models; local models silently run sequentially.
    parallel_with_prev: bool = False


def steps_from_pipeline_config(
    pipeline_steps: list,
    default_system: str,
    default_max_tokens: int,
) -> list[PipelineStep]:
    """Build a list of PipelineStep from a list of step dicts (e.g. from pipeline JSON).
    Uses default_system when a step has no system_prompt; uses default_max_tokens for
    non-review steps when a step has no max_tokens. Review steps default to 2000.
    """
    steps: list[PipelineStep] = []
    for it in pipeline_steps:
        if not isinstance(it, dict):
            continue
        mid = str(it.get("model_id") or "").strip()
        if not mid:
            continue
        step_name = str(it.get("name") or "Step")
        step_sys = str(it.get("system_prompt") or default_system)
        is_review = bool(it.get("is_review")) or _step_is_review(step_name, step_sys)
        raw_max = it.get("max_tokens")
        step_max = int(raw_max) if raw_max is not None else (2000 if is_review else default_max_tokens)
        steps.append(
            PipelineStep(
                name=step_name,
                model_id=mid,
                system_prompt=step_sys,
                prompt_template=str(it.get("prompt_template") or "{{input}}"),
                max_tokens=step_max,
                params=(it.get("params") if isinstance(it.get("params"), dict) else {}),
                is_review=is_review,
            )
        )
    return steps


def _render_template(template: str, vars: dict[str, str]) -> str:
    """
    Very small templating helper.

    Replaces {{key}} with vars[key]. Unknown keys are left unchanged.
    """
    out = template or ""
    for k, v in (vars or {}).items():
        out = out.replace("{{" + k + "}}", v or "")
    return out


def _make_streaming_callbacks(
    pass_num: int,
    total_steps: int,
    model_name: str,
    progress_callback: Optional[ProgressCallback],
    step_name: str = "",
):
    if not progress_callback:
        return None, None

    _last_log_at = [0]  # mutable cell so the closure can mutate it

    def on_token(count: int):
        evt: dict = {
            "pass": pass_num,
            "total_steps": total_steps,
            "stage": "generating",
            "model": model_name,
            "tokens_generated": count,
            "step_name": step_name,
        }
        if count - _last_log_at[0] >= 250:
            _last_log_at[0] = count
            evt["log"] = f"Step {pass_num}/{total_steps} [{model_name}] — {count} tokens..."
        asyncio.create_task(progress_callback(evt))

    def on_complete(total_tokens: int, elapsed_sec: float):
        speed = (total_tokens / elapsed_sec) if elapsed_sec > 0 else 0.0
        asyncio.create_task(
            progress_callback(
                {
                    "pass": pass_num,
                    "total_steps": total_steps,
                    "stage": "complete",
                    "model": model_name,
                    "tokens": total_tokens,
                    "elapsed_sec": elapsed_sec,
                    "speed": speed,
                    "step_name": step_name,
                    "log": f"Step {pass_num} done — {total_tokens} tokens in {elapsed_sec:.1f}s ({speed:.1f} tok/s)",
                }
            )
        )

    return on_token, on_complete


async def _run_single_step(
    *,
    config: Settings,
    step: PipelineStep,
    pass_num: int,
    total_steps: int,
    state: dict[str, str],
    profiles: dict[str, Any],
    progress_callback: Optional[ProgressCallback],
    evict_between_steps: bool,
) -> tuple[str, str, bool]:
    """
    Execute one pipeline step.

    Returns (output_text, display_name, is_review).
    """
    start_time = time.monotonic()
    step_params = dict(step.params or {})

    # Apply model profile defaults UNDER step params (step always wins).
    profile_params = get_params_for_model(step.model_id, profiles)
    step_params = {**profile_params, **step_params}

    run_model_id = step.model_id

    # VRAM guardrails for Ollama steps.
    if run_model_id.startswith("ollama:"):
        advised_mid, advised_params, advisory_logs = advise_ollama_step(
            model_id=run_model_id, step_params=step_params, config=config
        )
        run_model_id = advised_mid
        step_params = advised_params
        if progress_callback and advisory_logs:
            for msg in advisory_logs:
                await progress_callback(
                    {
                        "pass": pass_num,
                        "total_steps": total_steps,
                        "stage": "guardrail",
                        "step_name": step.name,
                        "log": msg,
                    }
                )

    # Unload Ollama model after use when the next step uses a different model.
    if (
        evict_between_steps
        and run_model_id.startswith("ollama:")
        and "keep_alive" not in step_params
    ):
        step_params["keep_alive"] = 0

    try:
        client, display_name = get_ai_client_for_model_id(run_model_id, config)
    except Exception as e:
        if run_model_id.startswith("gguf:"):
            fallback_mid = getattr(config, "OLLAMA_FALLBACK_MODEL_ID", "ollama:qwen3:14b")
            if progress_callback:
                await progress_callback(
                    {
                        "pass": pass_num,
                        "total_steps": total_steps,
                        "stage": "guardrail",
                        "step_name": step.name,
                        "log": (
                            f"GGUF unavailable for step '{step.name}' "
                            f"({type(e).__name__}: {e}) — falling back to {fallback_mid}. "
                            f"Run python scripts/jat.py setup to restore runtime support."
                        ),
                    }
                )
            step_params.setdefault("temperature", 0.2)
            run_model_id = fallback_mid
            client, display_name = get_ai_client_for_model_id(run_model_id, config)
        else:
            raise

    if progress_callback:
        await progress_callback({
            "pass": pass_num,
            "total_steps": total_steps,
            "stage": "starting",
            "model": display_name,
            "step_name": step.name,
            "log": f"Step {pass_num}/{total_steps} — {step.name} [{display_name}]",
        })

    prompt = _render_template(step.prompt_template, state).strip()
    on_token, on_complete = _make_streaming_callbacks(pass_num, total_steps, display_name, progress_callback, step.name)

    max_attempts = 1 + getattr(config, "PIPELINE_RETRY_ATTEMPTS", 2)
    backoff_sec = getattr(config, "PIPELINE_RETRY_BACKOFF_SECONDS", 2.0)
    last_error: Optional[Exception] = None
    out = ""

    for attempt in range(1, max_attempts + 1):
        try:
            out = await client.generate(
                prompt,
                system_prompt=step.system_prompt,
                max_tokens=step.max_tokens,
                on_token=on_token,
                on_complete=on_complete,
                params=step_params,
            )
            break
        except Exception as e:
            last_error = e
            if attempt < max_attempts:
                if progress_callback:
                    await progress_callback(
                        {
                            "pass": pass_num,
                            "total_steps": total_steps,
                            "stage": "generating",
                            "step_name": step.name,
                            "log": (
                                f"Step {pass_num} failed ({type(e).__name__}: {e}) — "
                                f"retrying in {backoff_sec}s (attempt {attempt}/{max_attempts})"
                            ),
                        }
                    )
                await asyncio.sleep(backoff_sec)
            else:
                if run_model_id.startswith("deepseek:"):
                    is_review = step.is_review or _step_is_review(step.name, step.system_prompt)
                    fail_on_content_skip = getattr(config, "PIPELINE_FAIL_ON_CONTENT_SKIP", True)
                    fail_on_any_skip = getattr(config, "PIPELINE_FAIL_ON_ANY_SKIP", False)
                    if (not is_review and fail_on_content_skip) or fail_on_any_skip:
                        logger.exception(
                            "pipeline content step failed after retries: step=%s, model=%s",
                            step.name,
                            step.model_id,
                        )
                        raise RuntimeError(
                            f"Step '{step.name}' failed after {max_attempts} attempts "
                            f"(DeepSeek: {type(last_error).__name__}: {last_error}). "
                            "Resume/cover incomplete. Set PIPELINE_FAIL_ON_CONTENT_SKIP=false to allow pass-through."
                        ) from last_error
                    logger.warning(
                        "skipping step (review=%s, fail_on_content_skip=%s): step=%s, model=%s, error=%s",
                        is_review,
                        fail_on_content_skip,
                        step.name,
                        step.model_id,
                        last_error,
                    )
                    if progress_callback:
                        await progress_callback(
                            {
                                "pass": pass_num,
                                "total_steps": total_steps,
                                "stage": "skipped",
                                "model": display_name,
                                "step_name": step.name,
                                "log": (
                                    f"Step {pass_num} skipped after {max_attempts} attempts: "
                                    f"DeepSeek failed ({type(last_error).__name__}: {last_error})"
                                ),
                            }
                        )
                    out = state.get("input", "")
                else:
                    logger.exception(
                        "pipeline step failed: step=%s, model=%s, attempts=%s",
                        step.name,
                        step.model_id,
                        max_attempts,
                    )
                    raise

    out = (out or "").strip()

    # VRAM cleanup for local model types.
    if evict_between_steps and step.model_id.startswith("gguf:"):
        try:
            from app.llamacpp_client import evict_all_llama_cache
            evict_all_llama_cache()
        except Exception as ev_err:
            logger.warning(
                "evict cache failed: step=%s, backend=gguf, error=%s",
                step.name,
                ev_err,
            )
    if evict_between_steps and step.model_id.startswith("hf:"):
        try:
            from app.transformers_client import evict_all_transformers_cache
            evict_all_transformers_cache()
        except Exception as ev_err:
            logger.warning(
                "evict cache failed: step=%s, backend=hf, error=%s",
                step.name,
                ev_err,
            )

    is_review = step.is_review or _step_is_review(step.name, step.system_prompt)
    elapsed_sec = time.monotonic() - start_time
    logger.info("pipeline step complete: step=%s, duration_sec=%.1f", step.name, elapsed_sec)
    return out, display_name, is_review


async def run_pipeline(
    *,
    config: Settings,
    steps: list[PipelineStep],
    vars: dict[str, str],
    progress_callback: Optional[ProgressCallback] = None,
    evict_between_steps: bool = True,
) -> dict[str, Any]:
    """
    Execute steps sequentially, with optional parallel groups for cloud steps.

    A step with parallel_with_prev=True runs concurrently with the step before it
    (sharing the same input). Parallel execution is only allowed for deepseek: steps;
    local models silently fall back to sequential even if the flag is set.

    Parallel step outputs are added to suggestions but do not advance the main chain.

    vars must include any {{placeholders}} referenced by prompt_template.
    During execution, vars["input"] is set to the previous sequential step's output,
    and vars["stepN"] is set for each completed step (1-indexed).

    Returns:
      { "final": str, "outputs": [str...], "suggestions": [str...], "prompts": [str...], "models": [str...] }
    """
    outputs: list[str] = []
    prompts: list[str] = []
    models: list[str] = []
    suggestions: list[str] = []

    state: dict[str, str] = dict(vars or {})
    state.setdefault("input", "")
    total_steps = len(steps or [])

    # Load model profiles once for the whole pipeline run.
    profiles = load_profiles(getattr(config, "MODEL_PROFILES_PATH", "data/model_profiles.json"))

    # Build execution groups. A "parallel" step must be deepseek:* to actually run in parallel;
    # otherwise it is demoted to sequential.
    # Each group is either [single_sequential_step] or [anchor_step, parallel_step, ...].
    groups: list[list[tuple[int, PipelineStep]]] = []
    for idx, step in enumerate(steps):
        is_parallel = (
            step.parallel_with_prev
            and step.model_id.startswith("deepseek:")
            and groups  # can't be parallel if there's no previous group
        )
        if is_parallel:
            groups[-1].append((idx, step))
        else:
            groups.append([(idx, step)])

    sequential_step_num = 0  # tracks position in the main chain for display

    for group in groups:
        anchor_idx, anchor_step = group[0]
        sequential_step_num += 1
        pass_num = sequential_step_num

        # Always run the anchor step first (it defines the input for parallel steps).
        out, display_name, is_review = await _run_single_step(
            config=config,
            step=anchor_step,
            pass_num=pass_num,
            total_steps=total_steps,
            state=state,
            profiles=profiles,
            progress_callback=progress_callback,
            evict_between_steps=evict_between_steps,
        )

        models.append(display_name)
        prompts.append(_render_template(anchor_step.prompt_template, state))
        outputs.append(out)
        if is_review:
            suggestions.append(out)
        state["input"] = out
        state[f"step{anchor_idx + 1}"] = out

        # Run any parallel steps concurrently using the anchor output as shared input.
        if len(group) > 1:
            parallel_entries = group[1:]

            async def _run_parallel(entry: tuple[int, PipelineStep]) -> tuple[int, str, str, bool]:
                pidx, pstep = entry
                pout, pname, p_is_review = await _run_single_step(
                    config=config,
                    step=pstep,
                    pass_num=pidx + 1,
                    total_steps=total_steps,
                    state=dict(state),  # snapshot so parallel steps don't race on state
                    profiles=profiles,
                    progress_callback=progress_callback,
                    evict_between_steps=evict_between_steps,
                )
                return pidx, pout, pname, p_is_review

            par_results = await asyncio.gather(
                *[_run_parallel(e) for e in parallel_entries],
                return_exceptions=True,
            )

            for i, result in enumerate(par_results):
                if isinstance(result, Exception):
                    # Retry this parallel step once before marking skipped
                    pidx, pstep = parallel_entries[i]
                    logger.warning(
                        "parallel step failed (first attempt), retrying: step=%s, error=%s",
                        pstep.name,
                        result,
                    )
                    try:
                        pout, pname, p_is_review = await _run_single_step(
                            config=config,
                            step=pstep,
                            pass_num=pidx + 1,
                            total_steps=total_steps,
                            state=dict(state),
                            profiles=profiles,
                            progress_callback=progress_callback,
                            evict_between_steps=evict_between_steps,
                        )
                        models.append(pname)
                        prompts.append("")
                        outputs.append(pout)
                        suggestions.append(pout)
                        state[f"step{pidx + 1}"] = pout
                    except Exception as retry_err:
                        logger.exception(
                            "parallel pipeline step failed: step=%s, error=%s",
                            pstep.name,
                            retry_err,
                        )
                        # Apply same fail-on-content logic as sequential path
                        p_is_review = pstep.is_review or _step_is_review(pstep.name, pstep.system_prompt)
                        fail_on_content_skip = getattr(config, "PIPELINE_FAIL_ON_CONTENT_SKIP", True)
                        fail_on_any_skip = getattr(config, "PIPELINE_FAIL_ON_ANY_SKIP", False)
                        if pstep.model_id.startswith("deepseek:") and (
                            (not p_is_review and fail_on_content_skip) or fail_on_any_skip
                        ):
                            raise RuntimeError(
                                f"Parallel step '{pstep.name}' failed after retry "
                                f"({type(retry_err).__name__}: {retry_err}). "
                                "Resume/cover incomplete."
                            ) from retry_err
                        if progress_callback:
                            await progress_callback({
                                "pass": sequential_step_num,
                                "total_steps": total_steps,
                                "stage": "skipped",
                                "step_name": pstep.name,
                                "log": f"Parallel step failed after retry: {retry_err}",
                            })
                    continue
                pidx, pout, pname, p_is_review = result
                models.append(pname)
                prompts.append("")
                outputs.append(pout)
                suggestions.append(pout)  # parallel outputs always go to suggestions
                state[f"step{pidx + 1}"] = pout

    return {
        "final": (outputs[-1] if outputs else ""),
        "outputs": outputs,
        "suggestions": suggestions,
        "prompts": prompts,
        "models": models,
    }
