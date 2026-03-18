# fix_ollama.ps1 - Diagnose Ollama, pull the best model for RTX 4090
# Usage: .\fix_ollama.ps1
# Run from: C:\Users\adsfo\Documents\scripts\job_app_tailor

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-Ollama {
    $cmd = Get-Command ollama -ErrorAction SilentlyContinue
    if (-not $cmd) {
        Write-Host ""
        Write-Host "ERROR: ollama not found on PATH." -ForegroundColor Red
        Write-Host "Install from: https://ollama.com/download" -ForegroundColor Yellow
        Write-Host "Then restart PowerShell and re-run this script." -ForegroundColor Yellow
        exit 1
    }
}

Assert-Ollama

Write-Host ""
Write-Host "=== Ollama Diagnostics ===" -ForegroundColor Cyan

Write-Host ""
Write-Host "1. Checking Ollama service (ollama list)..."
& ollama list | Out-Host

Write-Host ""
Write-Host "2. Currently installed models (ollama ps - running)..."
& ollama ps | Out-Host

# Read the primary model from .env (OLLAMA_MODEL key).
# Falls back to qwen2.5:32b if the key is missing or the file doesn't exist.
# Good choices for RTX 4090 (24 GB VRAM):
#   qwen2.5:32b  - ~19 GB VRAM (Q4_K_M), best all-round writing + reasoning
#   gemma3:27b   - ~16 GB VRAM, excellent narrative quality
#   qwen3:30b    - ~18 GB VRAM (MoE), very fast inference
$primaryModel = "qwen2.5:32b"
$envFile = Join-Path $PSScriptRoot ".env"
if (Test-Path $envFile) {
    $envLines = Get-Content $envFile | Where-Object { $_ -match '^\s*OLLAMA_MODEL\s*=' }
    if ($envLines) {
        $val = ($envLines[-1] -split '=', 2)[1].Trim().Trim('"').Trim("'")
        # Strip "ollama:" prefix if present (config may store it either way)
        $val = $val -replace '^ollama:', ''
        if ($val) { $primaryModel = $val }
    }
}
Write-Host "Primary model (from .env): $primaryModel" -ForegroundColor Cyan

Write-Host ""
Write-Host "3. Removing broken qwen3-coder:30b (if present)..." -ForegroundColor Yellow
try {
    & ollama rm "qwen3-coder:30b" 2>&1 | Out-Host
} catch {
    Write-Host "    (not present or already removed)" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "4. Pulling primary model: $primaryModel (~19 GB, fits RTX 4090 comfortably)" -ForegroundColor Green
Write-Host "   This may take a while on first download..."
& ollama pull $primaryModel | Out-Host

Write-Host ""
Write-Host "5. Verifying model is runnable..."
$testOut = & ollama run $primaryModel "Reply with exactly: READY" 2>&1
Write-Host $testOut

Write-Host ""
Write-Host "6. Final model list:"
& ollama list | Out-Host

Write-Host ""
Write-Host "=== Done ===" -ForegroundColor Green
Write-Host "Model '$primaryModel' is ready. Start the app with: .\run.ps1" -ForegroundColor Green
Write-Host ""
Write-Host "To change the default model, set OLLAMA_MODEL in .env and re-run this script." -ForegroundColor DarkGray
Write-Host "Optional smaller fallback (13 GB):  ollama pull mistral-small3.1:22b" -ForegroundColor DarkGray
