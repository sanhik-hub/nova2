$ErrorActionPreference = "Stop"

$NovaRoot = "C:\Users\sanhi\Documents\nova2"
$Python = "C:\Users\sanhi\Documents\code\python\.venv\Scripts\python.exe"

$LlamaServer = Join-Path $NovaRoot "engines\llama.cpp\llama-server.exe"
$LocalLLMConfigScript = Join-Path $NovaRoot "backend\providers\local_llm\config.py"
$FrontendRoot = Join-Path $NovaRoot "frontend"

$BackendHost = "127.0.0.1"
$BackendPort = 8000

$FrontendHost = "127.0.0.1"
$FrontendPort = 5173

if (-not (Test-Path $Python)) {
    throw "Python executable not found: $Python"
}

if (-not (Test-Path $LlamaServer)) {
    throw "llama-server.exe not found: $LlamaServer"
}

if (-not (Test-Path $LocalLLMConfigScript)) {
    throw "Local model configuration not found: $LocalLLMConfigScript"
}

if (-not (Test-Path $FrontendRoot)) {
    throw "Frontend directory not found: $FrontendRoot"
}

if (-not (Test-Path (Join-Path $FrontendRoot "package.json"))) {
    throw "Frontend package.json not found: $FrontendRoot"
}

# ------------------------------------------------------------
# Resolve Node / npm
# ------------------------------------------------------------

$NodeCommand = Get-Command node.exe -ErrorAction SilentlyContinue

if (-not $NodeCommand) {
    throw "node.exe was not found in PATH."
}

$NodeExe = $NodeCommand.Source
$NodeDir = Split-Path $NodeExe
$NpmExe = Join-Path $NodeDir "npm.exe"

if (-not (Test-Path $NpmExe)) {
    throw "npm.exe was not found beside Node.js: $NpmExe"
}

# ------------------------------------------------------------
# Load local model configuration
# ------------------------------------------------------------

$LocalLLMConfigJson = & $Python $LocalLLMConfigScript --json 2>&1

if ($LASTEXITCODE -ne 0) {
    throw "Unable to read local model configuration.`n$LocalLLMConfigJson"
}

try {
    $LocalLLMConfig = $LocalLLMConfigJson | ConvertFrom-Json -ErrorAction Stop
}
catch {
    throw "Local model configuration did not return valid JSON.`n$LocalLLMConfigJson"
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

if (
    -not $LocalLLMServerUri.IsAbsoluteUri -or
    $LocalLLMServerUri.Scheme -ne "http" -or
    $LocalLLMServerUri.AbsolutePath -ne "/"
) {
    throw "Local model server URL must be an HTTP origin."
}

$LocalLLMHost = $LocalLLMServerUri.Host
$LocalLLMPort = $LocalLLMServerUri.Port

$LocalLLMHealthUrl =
    "$($LocalLLMServerUri.GetLeftPart([System.UriPartial]::Authority))/health"

$LocalModelName =
    [System.IO.Path]::GetFileNameWithoutExtension($LocalModel)

if (-not (Test-Path $LocalModel)) {
    throw "Configured local model not found: $LocalModel"
}

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

function Get-Listener {
    param(
        [string]$HostAddress,
        [int]$Port
    )

    return Get-NetTCPConnection `
        -LocalAddress $HostAddress `
        -LocalPort $Port `
        -State Listen `
        -ErrorAction SilentlyContinue
}

function Wait-ForHttp {
    param(
        [string]$Url,
        [int]$TimeoutSeconds,
        [string]$Name
    )

    Write-Host "Waiting for $Name..."

    for ($i = 0; $i -lt ($TimeoutSeconds * 2); $i++) {
        try {
            $response = Invoke-WebRequest `
                -Uri $Url `
                -UseBasicParsing `
                -TimeoutSec 2 `
                -ErrorAction Stop

            if (
                $response.StatusCode -ge 200 -and
                $response.StatusCode -lt 500
            ) {
                Write-Host "$Name is ready."
                return $true
            }
        }
        catch {
        }

        Start-Sleep -Milliseconds 500
    }

    Write-Warning "$Name did not become ready within $TimeoutSeconds seconds."
    return $false
}

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

Write-Host ""
Write-Host "============================================"
Write-Host " PrismOffline"
Write-Host "============================================"
Write-Host "Root     : $NovaRoot"
Write-Host "Frontend : http://${FrontendHost}:${FrontendPort}"
Write-Host "Backend  : http://${BackendHost}:${BackendPort}"
Write-Host "Model    : $LocalModelName"
Write-Host "LLM      : http://${LocalLLMHost}:${LocalLLMPort}"
Write-Host "Node     : $NodeExe"
Write-Host "npm      : $NpmExe"
Write-Host "Wolfram  : persistent inside backend"
Write-Host "============================================"
Write-Host ""

# ------------------------------------------------------------
# Local LLM
# ------------------------------------------------------------

$LocalLLMListener = Get-Listener $LocalLLMHost $LocalLLMPort

if ($LocalLLMListener) {
    Write-Host "Local model server is already running on port $LocalLLMPort."
}
else {
    Write-Host "Starting local model server on port $LocalLLMPort..."

    Start-Process `
        -FilePath $LlamaServer `
        -ArgumentList @(
            "-m",
            $LocalModel,
            "--host",
            $LocalLLMHost,
            "--port",
            [string]$LocalLLMPort
        ) `
        -WorkingDirectory $NovaRoot `
        -WindowStyle Hidden
}

$LocalLLMReady = Wait-ForHttp `
    $LocalLLMHealthUrl `
    60 `
    "Local model server"

if (-not $LocalLLMReady) {
    throw "The local model server did not become ready."
}

# ------------------------------------------------------------
# Backend
# ------------------------------------------------------------

$BackendListener = Get-Listener $BackendHost $BackendPort

if ($BackendListener) {
    Write-Host "Backend is already running on port $BackendPort."
    Write-Host "Existing backend will be reused."
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

$BackendReady = Wait-ForHttp `
    "http://${BackendHost}:${BackendPort}/docs" `
    30 `
    "Backend"

if (-not $BackendReady) {
    throw "Backend did not become ready."
}

# ------------------------------------------------------------
# Frontend
# ------------------------------------------------------------

$FrontendListener = Get-Listener $FrontendHost $FrontendPort

if ($FrontendListener) {
    Write-Host "Frontend is already running on port $FrontendPort."
    Write-Host "Existing frontend will be reused."
}
else {
    Write-Host "Starting Vite frontend on port $FrontendPort..."

    Start-Process `
        -FilePath $NpmExe `
        -ArgumentList @(
            "run",
            "dev",
            "--",
            "--host",
            $FrontendHost,
            "--port",
            [string]$FrontendPort
        ) `
        -WorkingDirectory $FrontendRoot `
        -WindowStyle Hidden
}

$FrontendReady = Wait-ForHttp `
    "http://${FrontendHost}:${FrontendPort}/" `
    30 `
    "Frontend"

if (-not $FrontendReady) {
    throw "Frontend did not become ready."
}

# ------------------------------------------------------------
# Final verification
# ------------------------------------------------------------

$FinalBackend = Get-Listener $BackendHost $BackendPort
$FinalLLM = Get-Listener $LocalLLMHost $LocalLLMPort
$FinalFrontend = Get-Listener $FrontendHost $FrontendPort

Write-Host ""
Write-Host "============================================"

if (
    $FinalFrontend -and
    $FinalBackend -and
    $FinalLLM
) {
    Write-Host " PrismOffline is running"
}
else {
    Write-Host " PrismOffline startup incomplete"
}

Write-Host "============================================"
Write-Host "Frontend : http://${FrontendHost}:${FrontendPort}"
Write-Host "Backend  : http://${BackendHost}:${BackendPort}"
Write-Host "Model    : $LocalModelName"
Write-Host "LLM      : http://${LocalLLMHost}:${LocalLLMPort}"
Write-Host "Wolfram  : persistent session in backend"
Write-Host "============================================"
Write-Host ""