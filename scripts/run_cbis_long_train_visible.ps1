# Harmonia: uzun merkezi eğitimi GÖRÜNÜR terminalde çalıştırır (stdout canlı, ayrıca log dosyasına yazar).
# Kullanım (repo kökünden):
#   .\scripts\run_cbis_long_train_visible.ps1
# Yeni pencerede açmak için:
#   Start-Process powershell -ArgumentList '-NoExit','-NoProfile','-ExecutionPolicy','Bypass','-File','scripts\run_cbis_long_train_visible.ps1' -WorkingDirectory (Get-Location)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
$env:PYTHONPATH = $Root
$env:PYTHONUNBUFFERED = "1"

Write-Warning "Aynı anda iki kez uzun eğitim başlatmayın (GPU/CPU çakışır). Zaten arka planda python eğitimi varsa Görev Yöneticisi'nden kapatın."
Write-Host "Devam için Enter — iptal için bu pencerede Ctrl+C" -ForegroundColor Yellow
$null = Read-Host

$Log = Join-Path $Root "harmonia_long_train.log"

Write-Host ""
Write-Host "========== Harmonia — hangi Python? ==========" -ForegroundColor Cyan
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if ($py) { Write-Host "  where python -> $py" }
& python -c "import sys; print('  sys.executable =', sys.executable); print('  version       =', sys.version.split()[0])"
Write-Host "  CUDA (PyTorch):" -ForegroundColor Cyan
& python -c "import torch; print('  torch.cuda.is_available() =', torch.cuda.is_available()); print('  torch.version.cuda      =', getattr(torch.version, 'cuda', None))"
Write-Host "  repo          = $Root"
Write-Host "  log (tee)     = $Log"
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""

$args = @(
    "-u", "-m", "harmonia_vision.benchmark",
    "--mode", "centralized",
    "--data-root", (Join-Path $Root "harmonia_processed_cbis_full"),
    "--preset", "long",
    "--checkpoint", (Join-Path $Root "harmonia_checkpoints\cbis_full_centralized_long.pth"),
    "--device", "auto"
)

Write-Host "Komut: python $($args -join ' ')" -ForegroundColor Green
Write-Host ""

# Ekranda göster + dosyaya ekle (önceki arka plan sürecinden bağımsız yeni koşu)
& python @args 2>&1 | Tee-Object -FilePath $Log -Append

Write-Host ""
Write-Host "Çıkış kodu: $LASTEXITCODE" -ForegroundColor $(if ($LASTEXITCODE -eq 0) { "Green" } else { "Red" })
