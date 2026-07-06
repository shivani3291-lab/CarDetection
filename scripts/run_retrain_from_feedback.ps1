# Wrapper for Task Scheduler: fine-tune on accumulated feedback, log the run.
# Registered to run every 8 days - see CLAUDE.md for the schtasks command.
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$python = "$env:USERPROFILE\anaconda3\envs\car-detect\python.exe"
$logDir = Join-Path $root "data\feedback"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir "retrain_runs.log"

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"[$timestamp] starting feedback retrain" | Out-File -FilePath $logFile -Append -Encoding utf8

Push-Location $root
try {
    & $python "scripts\retrain_from_feedback.py" 2>&1 | Out-File -FilePath $logFile -Append -Encoding utf8
} finally {
    Pop-Location
}

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"[$timestamp] finished feedback retrain (exit code $LASTEXITCODE)" | Out-File -FilePath $logFile -Append -Encoding utf8
