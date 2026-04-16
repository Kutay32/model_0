# Start the FastAPI server and open the browser UI.
# Usage (from repo root): .\scripts\run_demo.ps1
# API: http://127.0.0.1:8080  ·  Health: http://127.0.0.1:8080/api/health

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Uvicorn = Join-Path $Root ".venv\Scripts\uvicorn.exe"
if (-not (Test-Path $Uvicorn)) {
  Write-Error "Install deps: .\.venv\Scripts\pip install -r requirements-app.txt"
}

$env:MODEL_ROOT = Join-Path $Root "mammography\checkpoints"
Start-Process "http://127.0.0.1:8080/"
& $Uvicorn "api.main:app" --host "127.0.0.1" --port "8080"
