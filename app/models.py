"""Pydantic request/response models for the API."""

from typing import List, Optional

from pydantic import BaseModel, Field


class StartRequest(BaseModel):
    url: str = Field(..., description="Job posting URL")
    ollama_model: Optional[str] = None
    use_deepseek: Optional[bool] = None
    pipeline_preset: Optional[str] = Field(
        default=None,
        description="Optional pipeline preset name (stored server-side in data/pipelines/).",
    )
    job_title_override: Optional[str] = Field(
        default=None,
        description="Optional job title override (skips title extraction).",
    )
    job_description_override: Optional[str] = Field(
        default=None,
        description="Optional job description override (skips scraping).",
    )
    one_up_focus: Optional[str] = Field(
        default=None,
        description="Optional one-up focus to apply as extra instruction.",
    )
    model_sequence: Optional[List[str]] = Field(
        default=None,
        description="Optional model sequence for pipeline passes. Each entry must be a prefixed model ID: ollama:<name>, deepseek:<name>, gguf:<name>, or hf:<name>. Saved by the Sequencer UI.",
    )
    parallel_flags: Optional[List[bool]] = Field(
        default=None,
        description="Per-step parallel flag (same length as model_sequence). Only honored for deepseek: steps; local models always run sequentially regardless of this flag.",
    )


class BatchStartRequest(BaseModel):
    urls: List[str] = Field(..., description="List of job posting URLs to queue")
    ollama_model: Optional[str] = None
    model_sequence: Optional[List[str]] = None
    parallel_flags: Optional[List[bool]] = None


class JobStatus(BaseModel):
    job_id: str
    status: str = Field(..., description="pending, scraping, generating, needs_info, complete, error")
    progress: Optional[str] = None
    message: Optional[str] = None
    question: Optional[str] = None


class JobResult(BaseModel):
    job_id: str
    resume: Optional[str] = None
    cover_letter: Optional[str] = None
    artifacts: List[str] = Field(default_factory=list)
    job_title: Optional[str] = None


class AnswerRequest(BaseModel):
    job_id: str
    answer: str
