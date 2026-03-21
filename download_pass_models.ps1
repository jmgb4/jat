# Download pass models wrapper.
# Usage: .\download_pass_models.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = $PSScriptRoot
Set-Location $root

& python "scripts/jat.py" download-pass-models
exit $LASTEXITCODE

