# Harmonia Hub — local training dashboard (http://127.0.0.1:8765)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = $root
& "$root\.venv\Scripts\python.exe" -m uvicorn harmonia_vision.hub_app:app --host 127.0.0.1 --port 8765
