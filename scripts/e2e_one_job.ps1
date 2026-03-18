# E2E: POST one job with job_description_override, poll until complete, assert outputs.
# Prereq: App must be running (e.g. .\run.ps1). Two base resumes (leadership + engineering) in data/base_resume/ for role-based flow.
# Usage: .\scripts\e2e_one_job.ps1 [-BaseUrl "http://127.0.0.1:8000"] [-PipelinePreset "smoke_e2e"] [-PollTimeoutSeconds 300]

param(
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [string]$PipelinePreset = "smoke_e2e",
    [int]$PollTimeoutSeconds = 300,
    [int]$PollIntervalSeconds = 3
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir

$body = @{
    url = "https://example.com/job/e2e-test"
    job_title_override = "Security Specialist I"
    job_description_override = "Security Specialist I – Remote. Requirements: 3+ years security operations; SIEM, vulnerability management; compliance (HIPAA). Nice to have: CISSP, CEH."
} | ConvertTo-Json

if ($PipelinePreset) {
    $bodyObj = $body | ConvertFrom-Json
    $bodyObj | Add-Member -NotePropertyName "pipeline_preset" -NotePropertyValue $PipelinePreset -Force
    $body = $bodyObj | ConvertTo-Json
}

Write-Host "POST $BaseUrl/start ..."
try {
    $startResp = Invoke-RestMethod -Uri "$BaseUrl/start" -Method Post -Body $body -ContentType "application/json"
} catch {
    Write-Error "POST /start failed. Is the app running at $BaseUrl ? $_"
    exit 1
}

$jobId = $startResp.job_id
if (-not $jobId) {
    Write-Error "Response missing job_id: $startResp"
    exit 1
}
Write-Host "job_id: $jobId (polling up to $PollTimeoutSeconds s)..."

$deadline = [DateTimeOffset]::UtcNow.AddSeconds($PollTimeoutSeconds)
$status = $null
$lastStatus = $null

while ([DateTimeOffset]::UtcNow -lt $deadline) {
    try {
        $job = Invoke-RestMethod -Uri "$BaseUrl/status/$jobId" -Method Get
    } catch {
        Write-Error "GET /status/$jobId failed: $_"
        exit 1
    }
    $status = $job.status
    if ($status -ne $lastStatus) {
        Write-Host "  status: $status"
        $lastStatus = $status
    }
    if ($status -eq "complete") {
        break
    }
    if ($status -eq "error") {
        Write-Error "Job failed: $($job.message)"
        exit 1
    }
    Start-Sleep -Seconds $PollIntervalSeconds
}

if ($status -ne "complete") {
    Write-Error "Job did not complete within $PollTimeoutSeconds s (status: $status)"
    exit 1
}

# Fetch result (resume and cover_letter)
try {
    $result = Invoke-RestMethod -Uri "$BaseUrl/result/$jobId" -Method Get
} catch {
    Write-Error "GET /result/$jobId failed: $_"
    exit 1
}

# Assert non-empty resume and cover_letter
$resume = $result.resume
$cover = $result.cover_letter
if (-not $resume -or ($resume -match "^\s*$")) {
    Write-Error "Job complete but resume is empty"
    exit 1
}
if (-not $cover -or ($cover -match "^\s*$")) {
    Write-Error "Job complete but cover_letter is empty"
    exit 1
}

Write-Host "OK: resume and cover_letter are non-empty (resume length: $($resume.Length), cover length: $($cover.Length))"

# Optional: check resume_final.md exists
$resumeFinalPath = Join-Path $projectRoot "data\jobs\$jobId\resume_final.md"
if (Test-Path $resumeFinalPath) {
    Write-Host "OK: $resumeFinalPath exists"
} else {
    Write-Host "Note: $resumeFinalPath not found (artifact path may differ)"
}

Write-Host "E2E one-job test passed."
