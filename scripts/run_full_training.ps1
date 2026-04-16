# Full CBIS-DDSM mammogram training: manifests + UNet++ + CLIP (see train_cbis_mammogram.ps1).
# Requires CBIS_ROOT pointing at the dataset root (must contain csv/ and jpeg/).
# Run from repo root:
#   $env:CBIS_ROOT = "C:\path\to\cbis-ddsm"
#   .\scripts\run_full_training.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not $env:CBIS_ROOT) {
  Write-Error "Set CBIS_ROOT to your CBIS-DDSM root (folder containing csv/ and jpeg/). Example: `$env:CBIS_ROOT = (python -c `"import kagglehub; print(kagglehub.dataset_download('awsaf49/cbis-ddsm-breast-cancer-image-dataset'))`")"
}

$TrainScript = Join-Path $Root "scripts\train_cbis_mammogram.ps1"
& $TrainScript
