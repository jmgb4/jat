# Commit all changes and push to GitHub. Run after every build or significant change session.
# Usage: .\commit-build.ps1
#        .\commit-build.ps1 "resume pipeline fixes"
$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
Set-Location $projectRoot

if (-not (Test-Path .git)) {
    Write-Host "Not a git repository. Run from project root."
    exit 1
}
$remote = git remote get-url origin 2>$null
if (-not $remote) {
    Write-Host "No remote 'origin' configured. Add a remote and try again."
    exit 1
}

$status = git status --porcelain
if (-not $status) {
    Write-Host "No changes to commit. Working tree clean."
    exit 0
}

git add -A
$date = Get-Date -Format "yyyy-MM-dd"
$suffix = $args[0]
$msg = if ($suffix) { "Build: $date - $suffix" } else { "Build: $date - resume/cover pipeline and style fixes" }
git commit -m $msg
if ($LASTEXITCODE -ne 0) {
    Write-Host "Commit failed."
    exit 1
}
git push
if ($LASTEXITCODE -ne 0) {
    Write-Host "Push failed (network or auth). Fix and run 'git push' manually."
    exit 1
}
Write-Host "Committed and pushed: $msg"
