"""Configuration using pydantic-settings with .env support."""

from typing import ClassVar, Dict, List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    BASE_RESUME_PATH: str = "data/base_resume/resume.docx"
    # Role-based resumes: if both are set (or discovered by convention), the app picks leadership vs engineering by job role.
    BASE_RESUME_LEADERSHIP_PATH: str = ""
    BASE_RESUME_ENGINEERING_PATH: str = ""
    CONTEXT_DIR: str = "data/context/"
    JOBS_DIR: str = "data/jobs/"

    DEEPSEEK_API_KEY: str = ""
    # Privacy: redact likely PII before sending to DeepSeek
    REDACT_PII_FOR_DEEPSEEK: bool = True
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen3:14b"
    OLLAMA_API_BASE: str = "http://localhost:11434/api"

    # Model management
    MODELS_CATALOG_PATH: str = "data/models_catalog.json"
    MODEL_REGISTRY_PATH: str = "data/models_registry.json"
    HUB_CACHE_DIR: str = "data/hub_cache/"
    # Optional Hugging Face token for higher rate limits.
    HF_TOKEN: str = ""
    # Default 4-pass local sequence using Ollama on RTX 4090.
    # qwen2.5:32b fits comfortably in 24 GB VRAM (Q4_K_M ~19 GB).
    # Pull with: .\fix_ollama.ps1
    # Override via the Sequencer tab or pipeline presets.
    DEFAULT_MODEL_SEQUENCE: List[str] = [
        "ollama:qwen3:14b",
        "ollama:qwen3:14b",
        "ollama:qwen3:14b",
        "ollama:qwen3:14b",
    ]

    # Per-step-role temperature defaults (override in .env)
    DRAFT_TEMPERATURE: float = 0.7
    REVIEW_TEMPERATURE: float = 0.2
    APPLY_TEMPERATURE: float = 0.4
    REFINE_TEMPERATURE: float = 0.5
    POLISH_TEMPERATURE: float = 0.7
    # Humanize step: rewrite final output for human-like rhythm and variation (reduces AI-detection flags).
    HUMANIZE_STEP: bool = True
    HUMANIZE_TEMPERATURE: float = 0.6

    # Personal info fill: substitute {{PLACEHOLDER}} from data/personal/ (local-only by default; PII never sent to API).
    PERSONAL_FILL_ENABLED: bool = True
    PERSONAL_FILL_ONLY_LOCAL: bool = True
    PERSONAL_DIR: str = "data/personal/"

    # Per-step context window overrides (0 = use global OLLAMA_NUM_CTX)
    DRAFT_NUM_CTX: int = 0
    REVIEW_NUM_CTX: int = 0

    # Path to model tuning profiles JSON (user-editable, pattern-matched by model name)
    MODEL_PROFILES_PATH: str = "data/model_profiles.json"

    # llama.cpp runtime defaults (for installed GGUF models)
    LLAMACPP_N_CTX: int = 8192
    LLAMACPP_N_THREADS: int = 0  # 0 lets llama.cpp pick
    # GPU-only: require a CUDA-enabled llama-cpp-python build.
    # Use -1 to offload as many layers as possible (full GPU).
    LLAMACPP_N_GPU_LAYERS: int = -1
    LLAMACPP_TEMPERATURE: float = 0.3
    # If true, GGUF execution will fail fast unless GPU offload is available.
    LLAMACPP_REQUIRE_GPU: bool = True

    # Ollama GPU / context enforcement
    # num_gpu: number of model layers to offload to GPU. 999 = all (force full GPU).
    OLLAMA_NUM_GPU: int = 999
    # num_ctx: context window size sent to Ollama (env: OLLAMA_NUM_CTX). Use 32768 for 32k.
    # Override in .env: OLLAMA_NUM_CTX=32768
    OLLAMA_NUM_CTX: int = 32768
    # VRAM guardrails: attempt to avoid GPU->RAM spill before running a step.
    OLLAMA_VRAM_HEADROOM_MB: int = 2500
    OLLAMA_MIN_NUM_CTX: int = 4096
    # If a requested Ollama model is too large, substitute this model to stay GPU-first.
    OLLAMA_FALLBACK_MODEL_ID: str = "ollama:qwen3:14b"

    # Model selection
    # RTX 4090 recommended models (pull any of these via: ollama pull <name>)
    # qwen2.5:32b  ~19 GB - best quality for resume writing
    # gemma3:27b   ~16 GB - excellent narrative
    # qwen3:30b    ~18 GB - fast MoE reasoning
    # mistral-small3.1:22b ~13 GB - fastest option
    AVAILABLE_OLLAMA_MODELS: List[str] = [
        "qwen3:14b",               # installed
        "glm-4.7-flash-neo-code",  # installed
        "qwen2.5:32b",             # pull for higher quality (~19 GB)
        "gemma3:27b",
        "qwen3:30b",
        "mistral-small3.1:22b",
        "llama3.1:8b",
        # glm-5:cloud removed — broken manifest, not a real model
    ]
    DEFAULT_OLLAMA_MODEL: str = "qwen3:14b"
    DEFAULT_DEEPSEEK_MODEL: str = "deepseek-chat"
    # DeepSeek is used as a last-resort fallback when Ollama is unavailable.
    # Requires DEEPSEEK_API_KEY to be set (in .env or environment).
    USE_DEEPSEEK: bool = True
    # Test helper: when enabled, all AI calls are replaced with fast deterministic output.
    # This keeps Playwright smoke tests quick and avoids large model loads.
    SMOKE_TEST_MODE: bool = False

    MAX_CONTEXT_TOKENS: int = 32000

    # Known context window sizes per model (substring-matched, same convention as model_profiles.py).
    # Used by get_context_window() to pick the right budget when calling prepare_context().
    # Declared as ClassVar so Pydantic treats it as a class-level constant, not a private attr.
    _MODEL_CONTEXT_WINDOWS: ClassVar[Dict[str, int]] = {
        "glm-4.7":           32768,
        "glm":               32768,
        "qwen2.5":           16384,
        "qwen3":             32768,
        "mistral":           32768,
        "magistral":         32768,
        "deepseek-chat":     65536,
        "deepseek-reasoner": 65536,
    }
    QUALITY_OVER_SPEED: bool = True

    AI_PROVIDERS: List[str] = ["ollama", "deepseek"]

    # Stats / SSE
    STATS_UPDATE_INTERVAL: int = 1
    TOKEN_ESTIMATION_FACTOR: int = 4

    # Minimum job description length (chars). 0 = disabled. Shorter text fails the job with a clear message.
    MIN_JOB_DESCRIPTION_LENGTH: int = 100

    # Timeouts (seconds); longer when quality over speed
    SCRAPE_TIMEOUT: int = 30
    OLLAMA_TIMEOUT: int = 120
    DEEPSEEK_TIMEOUT: int = 90

    # Optional: warn in UI when DeepSeek usage approaches this (free tier limit)
    DEEPSEEK_WARN_TOKENS: int = 0

    # Pipeline: retries before marking a step skipped or failing
    PIPELINE_RETRY_ATTEMPTS: int = 2
    PIPELINE_RETRY_BACKOFF_SECONDS: float = 2.0
    # When True (default), a content step (non-review) that fails after retries raises and fails the job.
    # When False, pass-through previous output (may produce incomplete resume/cover).
    PIPELINE_FAIL_ON_CONTENT_SKIP: bool = True
    # When True, any step (including review) that would be skipped due to DeepSeek failure raises and fails the job.
    PIPELINE_FAIL_ON_ANY_SKIP: bool = False
    # When True, job fails if Step 1 (Parser) is skipped due to parse_job failure.
    PIPELINE_FAIL_IF_PARSER_SKIPPED: bool = False


def get_settings() -> Settings:
    return Settings()


def get_context_window(model_id: str, config: "Settings") -> int:
    """
    Return the known context window size (in tokens) for a given model ID.

    Matches by case-insensitive substring against _MODEL_CONTEXT_WINDOWS keys,
    using the longest matching key (most specific wins), identical to the
    model_profiles.get_params_for_model() convention.

    Falls back to config.OLLAMA_NUM_CTX when no key matches, and is capped
    at config.MAX_CONTEXT_TOKENS so we never exceed the global safety limit.
    """
    # Strip provider prefix (ollama:, gguf:, hf:, deepseek:)
    name = model_id or ""
    for prefix in ("ollama:", "gguf:", "hf:", "deepseek:"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    name = name.lower()

    windows = Settings._MODEL_CONTEXT_WINDOWS
    matches = sorted(
        [(k, v) for k, v in windows.items() if k.lower() in name],
        key=lambda kv: len(kv[0]),
    )
    if matches:
        window = matches[-1][1]  # longest (most specific) match
    else:
        window = config.OLLAMA_NUM_CTX  # fallback to configured global

    return min(window, config.MAX_CONTEXT_TOKENS)
