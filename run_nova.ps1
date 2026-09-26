$ErrorActionPreference = "Stop"

$NovaRoot = "C:\Users\sanhi\Documents\nova2"
$Python = "C:\Users\sanhi\Documents\code\python\.venv\Scripts\python.exe"
$LlamaServer = Join-Path $NovaRoot "engines\llama.cpp\llama-server.exe"
$LocalLLMConfigScript = Join-Path $NovaRoot "backend\providers\local_llm\config.py"

if (-not (Test-Path $Python)) {
    throw "Python executable not found: $Python"
}

if (-not (Test-Path $LlamaServer)) {
    throw "llama-server.exe not found: $LlamaServer"
}

if (-not (Test-Path $LocalLLMConfigScript)) {
    throw "Local model configuration not found: $LocalLLMConfigScript"
}

$LocalLLMConfigJson = & $Python $LocalLLMConfigScript --json
if ($LASTEXITCODE -ne 0) {
    throw "Unable to read local model configuration."
}

try {
    $LocalLLMConfig = $LocalLLMConfigJson | ConvertFrom-Json -ErrorAction Stop
}
catch {
    throw "Local model configuration did not return valid JSON."
}

$ConfiguredModelPath = [string]$LocalLLMConfig.model_path
if ([string]::IsNullOrWhiteSpace($ConfiguredModelPath)) {
    throw "Local model path is missing from the configuration."
}

if ([System.IO.Path]::IsPathRooted($ConfiguredModelPath)) {
    $LocalModel = $ConfiguredModelPath
}
else {
    $LocalModel = Join-Path $NovaRoot $ConfiguredModelPath
}

try {
    $LocalLLMServerUri = [System.Uri]$LocalLLMConfig.server_url
}
catch {
    throw "Local model server URL is invalid: $($LocalLLMConfig.server_url)"
}

if (-not $LocalLLMServerUri.IsAbsoluteUri -or $LocalLLMServerUri.Scheme -ne "http" -or $LocalLLMServerUri.AbsolutePath -ne "/") {
    throw "Local model server URL must be an HTTP origin, such as http://127.0.0.1:8080."
}

$LocalLLMHost = $LocalLLMServerUri.Host
$LocalLLMPort = $LocalLLMServerUri.Port
$LocalLLMHealthUrl = "$($LocalLLMServerUri.GetLeftPart([System.UriPartial]::Authority))/health"
$LocalModelName = [System.IO.Path]::GetFileNameWithoutExtension($LocalModel)

$BackendHost = "127.0.0.1"
$BackendPort = 8000

Write-Host ""
Write-Host "============================================"
Write-Host " PrismOffline"
Write-Host "============================================"
Write-Host "Root:    $NovaRoot"
Write-Host "Backend: http://${BackendHost}:${BackendPort}"
Write-Host "Model:   $LocalModelName"
Write-Host "Server:  $($LocalLLMServerUri.GetLeftPart([System.UriPartial]::Authority))"
Write-Host "Wolfram: persistent inside backend"
Write-Host "============================================"
Write-Host ""

if (-not (Test-Path $LocalModel)) {
    throw "Configured local model not found: $LocalModel"
}

$BackendListener = Get-NetTCPConnection `
    -LocalAddress $BackendHost `
    -LocalPort $BackendPort `
    -State Listen `
    -ErrorAction SilentlyContinue

$LocalLLMListener = Get-NetTCPConnection `
    -LocalAddress $LocalLLMHost `
    -LocalPort $LocalLLMPort `
    -State Listen `
    -ErrorAction SilentlyContinue

if ($BackendListener) {
    Write-Host "Backend is already running on port $BackendPort."
}
else {
    Write-Host "Starting FastAPI backend on port $BackendPort..."

    Start-Process `
        -FilePath $Python `
        -ArgumentList @(
            "-m",
            "uvicorn",
            "backend.main:app",
            "--host",
            $BackendHost,
            "--port",
            [string]$BackendPort
        ) `
        -WorkingDirectory $NovaRoot `
        -WindowStyle Hidden
}

if ($LocalLLMListener) {
    Write-Host "A local model server is already listening on port $LocalLLMPort."
}
else {
    Write-Host "Starting $LocalModelName on $LocalLLMHost`:$LocalLLMPort..."

    Start-Process `
        -FilePath $LlamaServer `
        -ArgumentList @(
            "-m",
            "`"$LocalModel`"",
            "--host",
            $LocalLLMHost,
            "--port",
            [string]$LocalLLMPort
        ) `
        -WorkingDirectory $NovaRoot `
        -WindowStyle Hidden
}

Write-Host ""
Write-Host "Waiting for the local model server..."

$LocalLLMReady = $false

for ($i = 0; $i -lt 60; $i++) {
    try {
        $response = Invoke-WebRequest `
            -Uri $LocalLLMHealthUrl `
            -UseBasicParsing `
            -TimeoutSec 2 `
            -ErrorAction Stop

        if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
            $LocalLLMReady = $true
            break
        }
    }
    catch {
        Start-Sleep -Seconds 1
    }
}

if ($LocalLLMReady) {
    Write-Host "$LocalModelName is ready."
}
else {
    Write-Warning "The local model server did not report ready within 60 seconds."
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
    Write-Host " PrismOffline is running"
    Write-Host "============================================"
    Write-Host "Backend : http://${BackendHost}:${BackendPort}"
    Write-Host "Model   : $LocalModelName"
    Write-Host "Server  : $($LocalLLMServerUri.GetLeftPart([System.UriPartial]::Authority))"
    Write-Host "Wolfram : persistent session in backend"
    Write-Host "============================================"
}
else {
    Write-Warning "Backend did not become ready within 30 seconds."
}

Write-Host ""
