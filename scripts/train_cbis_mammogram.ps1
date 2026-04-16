# Train CLIP + UNet++ on CBIS-DDSM (requires downloaded dataset).
# Set CBIS_ROOT to the folder that contains `csv/` and `jpeg/`.
# Example after kagglehub: $env:CBIS_ROOT = (python -c "import kagglehub; print(kagglehub.dataset_download('awsaf49/cbis-ddsm-breast-cancer-image-dataset'))")

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

if (-not $env:CBIS_ROOT) {
    Write-Host "Set CBIS_ROOT to your CBIS-DDSM root (must contain csv/ and jpeg/)." -ForegroundColor Yellow
    exit 1
}

$Py = Join-Path $Repo ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }

& $Py (Join-Path $Repo "mammography\build_manifest.py") --data-root $env:CBIS_ROOT --out-dir (Join-Path $Repo "mammography\cache")

$segManifest = Join-Path $Repo "mammography\cache\manifest_segmentation.csv"
$clsManifest = Join-Path $Repo "mammography\cache\manifest_classification.csv"

& $Py (Join-Path $Repo "mammography\train_unetplusplus.py") --manifest $segManifest --checkpoint-dir (Join-Path $Repo "mammography\checkpoints") --epochs 20 --batch-size 8
& $Py (Join-Path $Repo "mammography\train_clip_mammogram.py") --manifest $clsManifest --checkpoint-dir (Join-Path $Repo "mammography\checkpoints") --epochs 12 --batch-size 16

Write-Host "Checkpoints:" (Join-Path $Repo "mammography\checkpoints")
Write-Host "Demo: `$env:MODEL_ROOT = '$(Join-Path $Repo 'mammography\checkpoints')'"
