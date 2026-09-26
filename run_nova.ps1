$ErrorActionPreference = "Stop"

$NovaRoot = "C:\Users\sanhi\Documents\nova2"
$Python = "C:\Users\sanhi\Documents\code\python\.venv\Scripts\python.exe"
$LlamaServer = Join-Path $NovaRoot "engines\llama.cpp\llama-server.exe"
$QwenModel = Join-Path $NovaRoot "models\local\Qwen3-4B-Q4_K_M.gguf"

$BackendHost = "127.0.0.1"
$BackendPort = 8000

$QwenHost = "127.0.0.1"
$QwenPort = 8080

Write-Host ""
Write-Host "============================================"
Write-Host " Nova"
Write-Host "============================================"
Write-Host "Root:    $NovaRoot"
Write-Host "Backend: http://${BackendHost}:${BackendPort}"
Write-Host "Qwen:    http://${QwenHost}:${QwenPort}"
Write-Host "Wolfram: persistent inside backend"
Write-Host "============================================"
Write-Host ""

if (-not (Test-Path $Python)) {
    throw "Python executable not found: $Python"
}

if (-not (Test-Path $LlamaServer)) {
    throw "llama-server.exe not found: $LlamaServer"
}

if (-not (Test-Path $QwenModel)) {
    throw "Qwen model not found: $QwenModel"
}

$BackendListener = Get-NetTCPConnection `
    -LocalAddress $BackendHost `
    -LocalPort $BackendPort `
    -State Listen `
    -ErrorAction SilentlyContinue

$QwenListener = Get-NetTCPConnection `
    -LocalAddress $QwenHost `
    -LocalPort $QwenPort `
    -State Listen `
    -ErrorAction SilentlyContinue

if ($BackendListener) {
    Write-Host "Backend is already running on port $BackendPort."
}
else {
    Write-Host "Starting FastAPI backend on port $BackendPort..."

    Start-Process `
        -FilePath "powershell.exe" `
        -ArgumentList @(
            "-NoExit",
            "-Command",
            "Set-Location '$NovaRoot'; & '$Python' -m uvicorn backend.main:app --host $BackendHost --port $BackendPort"
        ) `
        -WorkingDirectory $NovaRoot
}

if ($QwenListener) {
    Write-Host "Qwen is already running on port $QwenPort."
}
else {
    Write-Host "Starting Qwen on port $QwenPort..."

    Start-Process `
        -FilePath "powershell.exe" `
        -ArgumentList @(
            "-NoExit",
            "-Command",
            "Set-Location '$NovaRoot'; & '$LlamaServer' -m '$QwenModel' --host $QwenHost --port $QwenPort"
        ) `
        -WorkingDirectory $NovaRoot
}

Write-Host ""
Write-Host "Waiting for Qwen..."

$QwenReady = $false

for ($i = 0; $i -lt 60; $i++) {
    try {
        $response = Invoke-WebRequest `
            -Uri "http://${QwenHost}:${QwenPort}/health" `
            -UseBasicParsing `
            -TimeoutSec 2 `
            -ErrorAction Stop

        if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
            $QwenReady = $true
            break
        }
    }
    catch {
        Start-Sleep -Seconds 1
    }
}

if ($QwenReady) {
    Write-Host "Qwen is ready."
}
else {
    Write-Warning "Qwen did not report ready within 60 seconds."
}

Write-Host ""
Write-Host "Waiting for backend..."

$BackendReady = $false

for ($i = 0; $i -lt 30; $i++) {
    try {
        $response = Invoke-WebRequest `
            -Uri "http://${BackendHost}:${BackendPort}/docs" `
            -UseBasicParsing `
            -TimeoutSec 2 `
            -ErrorAction Stop

        if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
            $BackendReady = $true
            break
        }
    }
    catch {
        Start-Sleep -Seconds 1
    }
}

Write-Host ""

if ($BackendReady) {
    Write-Host "============================================"
    Write-Host " Nova is running"
    Write-Host "============================================"
    Write-Host "Backend : http://${BackendHost}:${BackendPort}"
    Write-Host "Qwen    : http://${QwenHost}:${QwenPort}"
    Write-Host "Wolfram : persistent session in backend"
    Write-Host "============================================"
}
else {
    Write-Warning "Backend did not become ready within 30 seconds."
}

Write-Host ""