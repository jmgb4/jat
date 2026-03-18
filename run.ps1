# Run Job Application Tailor (activate venv and start server)
# Usage: .\run.ps1

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$venvPython = Join-Path $scriptDir "venv\Scripts\python.exe"
$venvUvicorn = Join-Path $scriptDir "venv\Scripts\uvicorn.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Virtual environment not found. Create it first:"
    Write-Host "  python -m venv venv"
    Write-Host "  .\venv\Scripts\pip.exe install -r requirements.txt"
    exit 1
}

# Redirect HF cache to project folder (avoids permission issues with default ~/.cache/huggingface)
$env:HF_HOME = Join-Path $scriptDir "data\hf_cache"

# Use python -m uvicorn so it always uses venv Python.
# IMPORTANT: when using --reload, restrict reload watching to the app/ folder.
# Otherwise large downloads written under data/ can trigger a reload which kills background tasks.
$reloadArgs = @()
if ($env:JAT_NO_RELOAD -ne "1") {
    $reloadArgs = @("--reload", "--reload-dir", "app")
}
& $venvPython -m uvicorn app.main:app @reloadArgs --host 127.0.0.1 --port 8000
