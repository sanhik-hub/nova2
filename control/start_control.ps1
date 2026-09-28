$ErrorActionPreference = "Stop"

$NovaRoot = "C:\Users\sanhi\Documents\nova2"
$Python = "C:\Users\sanhi\Documents\code\python\.venv\Scripts\python.exe"
$ControlServer = Join-Path $NovaRoot "control\server.py"

if (-not (Test-Path $Python)) {
    throw "Python executable not found: $Python"
}

if (-not (Test-Path $ControlServer)) {
    throw "Control server not found: $ControlServer"
}

Write-Host ""
Write-Host "============================================"
Write-Host " PrismOffline Control Center"
Write-Host "============================================"
Write-Host "Dashboard: http://127.0.0.1:8100"
Write-Host "============================================"
Write-Host ""

Start-Process `
    -FilePath $Python `
    -ArgumentList @($ControlServer) `
    -WorkingDirectory $NovaRoot `
    -WindowStyle Hidden

Start-Sleep -Seconds 1

Start-Process `
    "http://127.0.0.1:8100"

Write-Host "Control Center started."
