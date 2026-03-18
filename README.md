# Job Application Tailor (JAT)

A local, privacy-first web application that scrapes job postings, reads your base resume, and runs a multi-pass AI pipeline to produce tailored resumes and cover letters — all on your own machine.

Runs fully offline with [Ollama](https://ollama.com/). DeepSeek API is optional.

> **Privacy notice:** Your resume, context files, and personal info live in gitignored directories and never touch this repository. See [Data Files Setup](#data-files-setup) below.

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally
- A pulled Ollama model (e.g. `ollama pull llama3:8b`)
- DeepSeek API key — **optional**, falls back to Ollama if not set

---

## Quick Start

```powershell
# 1. Clone and set up
git clone https://github.com/YOUR_USERNAME/job_app_tailor.git
cd job_app_tailor
.\setup.ps1           # creates venv, installs deps, installs Playwright Chromium

# 2. Configure
copy .env.example .env
# Edit .env: set OLLAMA_MODEL to a model you have pulled

# 3. Add your data (see Data Files Setup below)

# 4. Run
.\run.ps1
# Open http://127.0.0.1:8000
```

---

## Data Files Setup

All three data directories are **gitignored** — your personal files never leave your machine.

### `data/base_resume/`

Drop your resume here as a single `.docx` or `.txt` file. If you have multiple files (e.g. an engineering version and a leadership version), the app will prompt you to choose one on the home page and remember your selection in `.selected`.

```
data/base_resume/
├── my_resume.docx          ← your file (gitignored)
└── .gitkeep
```

### `data/context/` — AI context files

These files give the AI deep knowledge of your background so it can write accurate, grounded output instead of generic filler. Six template files are included — copy each one, remove the `example_` prefix, and fill it in with your real content.

```powershell
cd data/context

# Copy all templates at once (PowerShell)
Get-ChildItem example_*.txt | ForEach-Object {
    Copy-Item $_ ($_.Name -replace '^example_', '')
}
```

| File | What to put in it |
|------|-------------------|
| `story.txt` | First-person career narrative covering your 2–5 flagship projects. Include specific tools, teams, and outcomes. This is the most important file. |
| `achievements.txt` | Your accomplishments as bullet points in multiple framings (technical, cross-functional, leadership). The AI picks the framing that fits the role. |
| `keywords.txt` | Your real tools, platforms, frameworks, methodologies, and target role titles. The AI uses this to match job description language. |
| `role_adaptation.txt` | Per-role instructions — what to emphasize, de-emphasize, and how to frame your work for each major role type you apply to. |
| `dos_and_donts.txt` | Hard writing rules: no buzzwords, bullet structure, quantification policy, tone. These are enforced on every pass. |
| `style_guide.txt` | One-liner style rules: section order, sentence length, heading format, tense. Applied on the final polish pass. |

Each `example_*.txt` file has detailed instructions at the top. The renamed files (without `example_`) are gitignored automatically — the examples stay in the repo for reference.

### `data/personal/` — Contact info substitution

This directory (fully gitignored) lets the app inject your real contact info into the final output using `{{KEY}}` placeholders.

```powershell
copy data\personal\template.txt data\personal\personal_info.txt
# Edit personal_info.txt with your name, email, phone, LinkedIn, etc.
```

The app replaces `{{EMAIL}}`, `{{PHONE}}`, `{{LINKEDIN}}` and other keys in the final resume and cover letter. This directory is gitignored and never committed.

---

## Configuration (`.env`)

Copy `.env.example` to `.env` and edit:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `llama3:8b` | Model to use for all passes (must be pulled) |
| `USE_DEEPSEEK` | `false` | Set `true` to use DeepSeek for review and polish passes |
| `DEEPSEEK_API_KEY` | _(empty)_ | Your DeepSeek key from [platform.deepseek.com](https://platform.deepseek.com/api_keys) |
| `REDACT_PII_FOR_DEEPSEEK` | `true` | Redacts emails, phones, LinkedIn URLs before sending to DeepSeek |
| `HF_TOKEN` | _(empty)_ | HuggingFace token — reduces rate limits for large GGUF downloads |
| `MAX_CONTEXT_TOKENS` | `32000` | Context window budget per pass |

---

## How It Works

1. **Scrape** — Playwright (Chromium) visits the job URL and extracts the title and full description.
2. **Four-pass generation:**
   - **Pass 1:** Initial tailored draft against your base resume + context
   - **Pass 2:** Review against the base resume and your rules
   - **Pass 3:** Apply review suggestions
   - **Pass 4:** Final polish — tone, bullets, style
3. **Output** — Resume and cover letter saved under `data/jobs/<job_id>/`. All intermediate passes are saved so you can inspect every step.

By default all passes run with Ollama (local). If `USE_DEEPSEEK=true`, passes 2 and 4 use DeepSeek with automatic fallback to Ollama on failure.

### Role-based resume selection

If you have multiple resume files in `data/base_resume/`, the app detects the job role from the description and selects the best-matching resume automatically. It falls back to a combined version when the role is ambiguous.

---

## Model Management

- **Ollama models:** Pull any model with `ollama pull <model>`. Set `OLLAMA_MODEL` in `.env`.
- **GGUF models:** Open `/manage-models` in the app to browse, download, and manage local GGUF models. Cached under `data/hub_cache/` (gitignored).
- **Sequencer:** Open `/sequencer` to assign a different model to each of the four passes.

---

## Scripts

| Script | Purpose | When to run |
|--------|---------|-------------|
| `setup.ps1` | Create venv, install dependencies, install Playwright Chromium | First-time setup |
| `run.ps1` | Start the web app (`uvicorn`) | Every session |
| `commit-build.ps1` | Commit all changes and push to GitHub | After code changes |
| `fix_ollama.ps1` | Diagnose Ollama, pull primary model from `.env` | When Ollama or GPU models need checking |
| `download_pass_models.ps1` | Download GGUF models used by the pipeline | When you want local GGUF models |
| `build.ps1` | Build a standalone executable with PyInstaller | When you want a distributable `.exe` |

---

## Privacy and Security

- **All personal data is gitignored.** `data/base_resume/`, `data/context/`, `data/personal/`, `data/jobs/`, and `data/job_history.db` are never committed.
- **PII redaction for DeepSeek:** When `REDACT_PII_FOR_DEEPSEEK=true`, the app strips emails, phone numbers, LinkedIn/GitHub URLs, and address-like patterns before sending any prompt to the DeepSeek API. Ollama (local) always receives the full text.
- **API keys:** Store your keys only in `.env` (gitignored). Never hardcode them in source files.

---

## Project Structure

```
job_app_tailor/
├── app/
│   ├── main.py              # FastAPI routes and background job runner
│   ├── spider.py            # Playwright scraping
│   ├── ai_client.py         # Provider router (Ollama / DeepSeek / GGUF / HF)
│   ├── resume_builder.py    # Resume generation passes
│   ├── cover_builder.py     # Cover letter generation
│   ├── context_manager.py   # Context file loading and token budgeting
│   ├── resume_locator.py    # Base resume detection and role-based selection
│   ├── role_classifier.py   # Classifies job role from description
│   ├── config.py            # Settings loaded from .env
│   ├── privacy.py           # PII redaction for external APIs
│   ├── templates/           # Jinja2 HTML templates
│   └── static/              # CSS
├── data/
│   ├── base_resume/         # Your resume files (gitignored)
│   ├── context/             # AI context files (gitignored; example_*.txt committed)
│   ├── personal/            # Contact info substitution (gitignored)
│   ├── jobs/                # Per-job output folders (gitignored)
│   └── pipelines/           # Saved pipeline configurations
├── .env.example             # Configuration template — copy to .env
├── requirements.txt
├── setup.ps1
├── run.ps1
└── README.md
```

---

## Running Tests

```powershell
# Unit tests
.\venv\Scripts\python.exe -m pytest tests/ -q

# E2E smoke test against a real job URL
.\scripts\e2e_one_job.ps1
```
