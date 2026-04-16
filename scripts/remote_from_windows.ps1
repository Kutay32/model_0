# Windows PowerShell: repoyu uzak Linux sunucuya kopyala (scp) ve kurulum/eğitim için ssh kullan.
# Gereksinim: OpenSSH istemcisi (ssh, scp).
#
# Örnekler (repo kökünden):
#   .\scripts\remote_from_windows.ps1 -Deploy
#   .\scripts\remote_from_windows.ps1 -Deploy -RunSetup
#   .\scripts\remote_from_windows.ps1 -RemoteHost "138.197.134.68" -RemoteUser "root" -Deploy -RunSetup
#
# GPU işi her zaman sunucuda çalışır; buradan "kullanmak" = SSH / VS Code Remote / Jupyter tüneli.
# Jupyter tüneli: sunucuda `jupyter lab --no-browser --port=8888`
#   yerelde: ssh -L 8888:127.0.0.1:8888 root@HOST

param(
    [string] $RemoteHost = "138.197.134.68",
    [string] $RemoteUser = "root",
    [string] $RemotePath = "/root/model_0",
    [switch] $Deploy,
    [switch] $RunSetup
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Split-Path -Parent $PSScriptRoot)).Path
$Remote = "${RemoteUser}@${RemoteHost}"

Write-Host "Local repo: $Repo" -ForegroundColor Cyan
Write-Host "Remote: ${Remote}:${RemotePath}" -ForegroundColor Cyan

if ($Deploy) {
    Write-Host "`n=== Uzak klasor olusturuluyor ===" -ForegroundColor Green
    & ssh $Remote "mkdir -p '$RemotePath'"
    Write-Host "`n=== Dosyalar kopyalaniyor (scp -r) — buyuk veri setlerini ayri tasiyin ===" -ForegroundColor Yellow
    & scp -r "$Repo\*" "${Remote}:${RemotePath}/"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "scp basarisiz. WSL ile rsync veya git clone alternatiflerini kullanin."
    }
    Write-Host "Kopya tamam." -ForegroundColor Green
}

if ($RunSetup) {
    Write-Host "`n=== Uzakta venv + pip + GPU testi ===" -ForegroundColor Green
    # Uzak yol bosluk icermemeli veya ssh tarafinda ayrica kaçilmali.
    & ssh -t $Remote "cd '$RemotePath' && chmod +x scripts/remote_linux_setup_and_train.sh && ./scripts/remote_linux_setup_and_train.sh '$RemotePath'"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Uzak kurulum hatasi (exit $LASTEXITCODE)."
    }
}

if (-not $Deploy -and -not $RunSetup) {
    Write-Host "`nKullanim: -Deploy ile kopyala, -RunSetup ile sunucuda scripts/remote_linux_setup_and_train.sh calistir." -ForegroundColor Yellow
    Write-Host "Ornek uzun egitim (sunucuda): ssh $Remote -> tmux new -s train -> cd $RemotePath && source .venv/bin/activate -> python mammography/train_multitask.py ..."
}
