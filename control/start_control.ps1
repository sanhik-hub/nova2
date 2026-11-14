$ErrorActionPreference = "Stop"

$NovaRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = if ($env:NOVA_PYTHON) { $env:NOVA_PYTHON } else { "C:\Users\sanhi\Documents\code\python\.venv\Scripts\python.exe" }
$ControlServer = Join-Path $NovaRoot "control\server.py"

$env:NOVA_ROOT = $NovaRoot
$env:NOVA_PYTHON = $Python
$ControlPort = 8100
$BackendPort = 8000
$FrontendPort = 5173
$ControlUrl = "http://127.0.0.1:$ControlPort"
if (-not (Test-Path $Python)) { throw "Python executable not found: $Python" }
if (-not (Test-Path $ControlServer)) { throw "Control server not found: $ControlServer" }

Write-Host ""
Write-Host "============================================"
Write-Host " PrismOffline Control Center"
Write-Host "============================================"
Write-Host "Root      : $NovaRoot"
Write-Host "Dashboard : $ControlUrl"
Write-Host "Backend   : $BackendPort"
Write-Host "Frontend  : $FrontendPort"
Write-Host "============================================"
Write-Host ""

$AlreadyRunning = $false
try {
    $response = Invoke-WebRequest -Uri $ControlUrl -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
    if ($response.StatusCode -eq 200) { $AlreadyRunning = $true }
} catch {}

if (-not $AlreadyRunning) {
    Start-Process -FilePath $Python -ArgumentList @($ControlServer) -WorkingDirectory $NovaRoot -WindowStyle Hidden
    $Ready = $false
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Milliseconds 250
        try {
            $response = Invoke-WebRequest -Uri $ControlUrl -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
            if ($response.StatusCode -eq 200) { $Ready = $true; break }
        } catch {}
    }
    if (-not $Ready) { throw "Control Center server did not become ready at $ControlUrl." }
}
Start-Process $ControlUrl
Write-Host "Control Center started."
