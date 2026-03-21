# Cross-Platform Inventory

This file tracks platform assumptions that impact setup, run, and test workflows.

## Core runtime status

- `app/` runtime code is mostly cross-platform (`pathlib`, relative paths, `sys.platform` guards).
- Windows-specific runtime logic that should stay:
  - `asyncio.WindowsProactorEventLoopPolicy` setup for Playwright subprocess compatibility.
  - `os.add_dll_directory` handling for CUDA DLL lookup on Windows.

## Setup/run/test workflow status

- Unified launcher exists in `scripts/jat.py` and is the primary workflow:
  - `python scripts/jat.py setup`
  - `python scripts/jat.py run`
  - `python scripts/jat.py test`
- PowerShell wrappers retained for Windows convenience:
  - `setup.ps1`
  - `run.ps1`

## Remaining Windows-first helper scripts (non-blocking)

- `build.ps1` (Windows executable packaging helper)
- `fix_ollama.ps1` (Windows diagnostics helper)
- `download_pass_models.ps1` (legacy helper; should delegate to Python launcher)
- `scripts/e2e_one_job.ps1` (PowerShell E2E helper)
- `commit-build.ps1` (repo automation helper)

## Message/docs assumptions to normalize

- Any user-facing remediation text should prefer:
  - `python scripts/jat.py setup`
  - `python scripts/jat.py run`
  - `python scripts/jat.py test`
- Avoid Windows-only paths like `venv\\Scripts\\...` unless shown inside explicit Windows-only sections.
