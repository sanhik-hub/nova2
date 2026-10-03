$ErrorActionPreference = "Stop"

$NovaRoot = "C:\Users\sanhi\Documents\nova2"
$Python = "C:\Users\sanhi\Documents\code\python\.venv\Scripts\python.exe"
$ControlServer = Join-Path $NovaRoot "control\server.py"
$ControlUrl = "http://127.0.0.1:8100"

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
Write-Host "Dashboard: $ControlUrl"
Write-Host "============================================"
Write-Host ""

# Reuse an already-running Control Center instead of starting a
# second server on the same port.
$AlreadyRunning = $false

try {
    $response = Invoke-WebRequest -Uri $ControlUrl -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
    if ($response.StatusCode -eq 200) {
        $AlreadyRunning = $true
    }
}
catch {
}

if (-not $AlreadyRunning) {
    Start-Process -FilePath $Python -ArgumentList @($ControlServer) -WorkingDirectory $NovaRoot -WindowStyle Hidden

    $Ready = $false
    for ($i = 0; $i -lt 20; $i++) {
        Start-Sleep -Milliseconds 250
        try {
            $response = Invoke-WebRequest -Uri $ControlUrl -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                $Ready = $true
                break
            }
        }
        catch {
        }
    }

    if (-not $Ready) {
        throw "Control Center server did not become ready on 127.0.0.1:8100."
    }
}

Start-Process $ControlUrl

Write-Host "Control Center started."