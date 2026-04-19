# Full CBIS-DDSM mass pipeline -> harmonia_processed_cbis_full/ (IID, no --max-groups)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = $root
$out = Join-Path $root "harmonia_processed_cbis_full"
$log = Join-Path $root "harmonia_processed_cbis_full_preprocess.log"
$data = Join-Path $root "data"
New-Item -ItemType Directory -Force -Path $out | Out-Null
Write-Host "Logging to $log"
& "$root\.venv\Scripts\python.exe" -m harmonia_vision.data_pipeline `
  --dataset-root $data `
  --out-root $out `
  --split-mode iid 2>&1 | Tee-Object -FilePath $log
