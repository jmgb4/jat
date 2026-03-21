# Cross-platform setup wrapper.
# Usage: .\setup.ps1
# This delegates to scripts/jat.py so setup logic stays in one place.

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "Python is required but not found on PATH." -ForegroundColor Red
    exit 1
}

& python "scripts/jat.py" setup
exit $LASTEXITCODE
