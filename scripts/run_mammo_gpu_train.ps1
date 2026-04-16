# Full mammography training on GPU: UNet++ then CLIP (from repo root).
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$env:PYTHONUNBUFFERED = "1"

$Py = Join-Path $Repo ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }

$seg = Join-Path $Repo "mammography\cache\manifest_segmentation.csv"
$cls = Join-Path $Repo "mammography\cache\manifest_classification.csv"
$ck = Join-Path $Repo "mammography\checkpoints"

if (-not (Test-Path $seg)) {
    Write-Host "Missing $seg — run mammography\build_manifest.py first." -ForegroundColor Red
    exit 1
}

Write-Host "=== UNet++ (GPU) ===" -ForegroundColor Cyan
& $Py -u (Join-Path $Repo "mammography\train_unetplusplus.py") --manifest $seg --checkpoint-dir $ck --epochs 20 --batch-size 8 --lr 3e-4
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== CLIP (GPU) ===" -ForegroundColor Cyan
& $Py -u (Join-Path $Repo "mammography\train_clip_mammogram.py") --manifest $cls --checkpoint-dir $ck --epochs 12 --batch-size 16 --lr 1e-5
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Done. Checkpoints: $ck" -ForegroundColor Green
