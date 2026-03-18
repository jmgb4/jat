# Optional: build a standalone executable with PyInstaller.
# Run from project root: .\build.ps1
# Requires: pip install pyinstaller
# Playwright: after build, run "playwright install chromium" from the dist folder
# or set PLAYWRIGHT_BROWSERS_PATH so the built app can find Chromium.

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
Set-Location $projectRoot

if (-not (Get-Command pyinstaller -ErrorAction SilentlyContinue)) {
    Write-Host "Install PyInstaller first: pip install pyinstaller"
    exit 1
}

# Single-file or single-dir; single-dir is more reliable with Playwright
pyinstaller --noconfirm `
    --name job_app_tailor `
    --add-data "app;app" `
    --add-data "data;data" `
    --hidden-import uvicorn.logging `
    --hidden-import uvicorn.loops `
    --hidden-import uvicorn.loops.auto `
    --hidden-import uvicorn.protocols `
    --hidden-import uvicorn.protocols.http `
    --hidden-import uvicorn.protocols.http.auto `
    --hidden-import uvicorn.protocols.websockets `
    --hidden-import uvicorn.protocols.websockets.auto `
    --collect-all playwright `
    app/main.py

Write-Host "Build complete. Run: .\dist\job_app_tailor\job_app_tailor.exe"
Write-Host "Ensure Chromium is installed for Playwright: playwright install chromium"
Write-Host "Remember to run .\commit-build.ps1 to push this build to GitHub."
