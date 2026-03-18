# One-time setup: create venv, install deps, GPU wheels, install Chromium for Playwright
# Usage: .\setup.ps1

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
}

Write-Host "Installing Python packages..."
.\venv\Scripts\pip.exe install -r requirements.txt

# -----------------------------------------------------------------------
# GPU SETUP: Install CUDA-enabled torch + llama-cpp-python
# These are NOT in requirements.txt because they need special index URLs.
# Safe to re-run: pip will skip if already at correct version.
# -----------------------------------------------------------------------

Write-Host ""
Write-Host "Installing CUDA-enabled torch (cu124)..." -ForegroundColor Cyan
Write-Host "  (Bundles cublas64_12.dll needed by llama-cpp-python on CUDA 13.x systems)"
.\venv\Scripts\pip.exe install torch torchvision `
    --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) {
    Write-Host "  WARNING: torch CUDA install failed (exit $LASTEXITCODE). GGUF GPU offload may not work." -ForegroundColor Yellow
}

Write-Host "Verifying torch CUDA..."
$torchCuda = & .\venv\Scripts\python.exe -c "import torch; print('1' if torch.cuda.is_available() else '0')" 2>$null
if ($torchCuda -eq "1") {
    $gpuName = & .\venv\Scripts\python.exe -c "import torch; print(torch.cuda.get_device_name(0))" 2>$null
    Write-Host "  torch CUDA OK: $gpuName" -ForegroundColor Green
} else {
    Write-Host "  WARNING: torch CUDA not available. GGUF and transformers models will run on CPU (very slow)." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Installing llama-cpp-python cu124 (prebuilt wheel, Python 3.12)..." -ForegroundColor Cyan
$llamaWheelUrl = "https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.4-cu124/llama_cpp_python-0.3.4-cp312-cp312-win_amd64.whl"
$currentLlama = & .\venv\Scripts\pip.exe show llama-cpp-python 2>$null | Select-String "Version:" | ForEach-Object { $_ -replace "Version: ", "" }
# Check if we already have the GPU build.
# New wheel layout (v0.3.4-cu124+): CUDA backend lives in ggml-cuda.dll (hundreds of MB).
# Old layout: single large llama.dll.
# CPU builds have no ggml-cuda.dll and a tiny llama.dll (~1.4 MB).
$llamaLibDir = Join-Path $scriptDir "venv\Lib\site-packages\llama_cpp\lib"
$ggmlCudaDll = Join-Path $llamaLibDir "ggml-cuda.dll"
$llamaDll    = Join-Path $llamaLibDir "llama.dll"
$isGpuBuild = $false
if (Test-Path $ggmlCudaDll) {
    $sizeMB = [math]::Round((Get-Item $ggmlCudaDll).Length / 1MB, 1)
    if ($sizeMB -gt 50) { $isGpuBuild = $true }
} elseif (Test-Path $llamaDll) {
    $sizeMB = [math]::Round((Get-Item $llamaDll).Length / 1MB, 1)
    if ($sizeMB -gt 50) { $isGpuBuild = $true }
}
if (-not $isGpuBuild) {
    Write-Host "  GPU build not detected. Installing cu124 GPU build (443 MB, one-time download)..."
    .\venv\Scripts\pip.exe install $llamaWheelUrl --force-reinstall --no-cache-dir
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "  ERROR: llama-cpp-python GPU wheel install FAILED (exit $LASTEXITCODE)." -ForegroundColor Red
        Write-Host "  GGUF models will NOT use the GPU. Try again or check your Python/CUDA version." -ForegroundColor Red
        Write-Host "  Expected: Python 3.12 + CUDA 12.4 compatible drivers." -ForegroundColor Yellow
    } else {
        Write-Host "  llama-cpp-python GPU wheel installed successfully." -ForegroundColor Green
    }
} else {
    Write-Host "  GPU build already installed (ggml-cuda.dll present), skipping." -ForegroundColor Green
}

Write-Host ""
Write-Host "Verifying llama-cpp-python GPU support..."
# Write result to a temp file to avoid PowerShell's NativeCommandError on llama_cpp stderr output
$gpuCheckScript = @"
import os, sys, pathlib, tempfile, traceback
result = '0'
try:
    import torch
    p = pathlib.Path(torch.__file__).parent / 'lib'
    if hasattr(os, 'add_dll_directory') and p.is_dir():
        os.add_dll_directory(str(p))
except Exception:
    pass
try:
    from llama_cpp import llama_cpp as lc
    fn = getattr(lc, 'llama_supports_gpu_offload', None)
    result = '1' if (callable(fn) and fn()) else '0'
except Exception:
    result = 'err'
outfile = sys.argv[1]
with open(outfile, 'w') as f:
    f.write(result)
"@
$tmpScript = [System.IO.Path]::GetTempFileName() + ".py"
$tmpResult = [System.IO.Path]::GetTempFileName()
$gpuCheckScript | Set-Content -Path $tmpScript -Encoding UTF8
$_saved = $ErrorActionPreference
$ErrorActionPreference = "SilentlyContinue"
& .\venv\Scripts\python.exe $tmpScript $tmpResult 2>$null
$ErrorActionPreference = $_saved
$gpuOk = (Get-Content $tmpResult -ErrorAction SilentlyContinue).Trim()
Remove-Item $tmpScript, $tmpResult -ErrorAction SilentlyContinue
if ($gpuOk -eq "1") {
    Write-Host "  llama-cpp-python GPU offload: OK" -ForegroundColor Green
} elseif ($gpuOk -eq "0") {
    Write-Host "  WARNING: llama-cpp-python GPU offload not available (CPU build)." -ForegroundColor Yellow
} else {
    Write-Host "  WARNING: Could not verify GPU support. Check that llama-cpp-python installed correctly." -ForegroundColor Yellow
}

# -----------------------------------------------------------------------
# Update transformers to version that supports Qwen3
# -----------------------------------------------------------------------
Write-Host ""
Write-Host "Upgrading transformers (Qwen3 support requires >=4.51)..." -ForegroundColor Cyan
.\venv\Scripts\pip.exe install "transformers>=4.51.0" "accelerate>=0.30.0" --upgrade

Write-Host ""
Write-Host "Installing Playwright Chromium (required for scraping)..."
.\venv\Scripts\python.exe -m playwright install chromium

Write-Host ""
Write-Host "=== Setup complete ===" -ForegroundColor Green
Write-Host "Run the app:        .\run.ps1"
Write-Host "Pull Ollama model:  .\fix_ollama.ps1"
Write-Host "Run tests:          .\venv\Scripts\python.exe -m pytest tests\ -q"
