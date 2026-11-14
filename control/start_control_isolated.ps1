$ErrorActionPreference = "Stop"

# Launch the test checkout on Nova's normal fixed ports. Only one stack may run at a time.
$NovaRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$SharedRoot = Join-Path (Split-Path -Parent $NovaRoot) "nova2"
$Python = "C:\Users\sanhi\Documents\code\python\.venv\Scripts\python.exe"

if (-not (Test-Path $SharedRoot)) { throw "Shared main repository not found: $SharedRoot" }
if (-not (Test-Path $Python)) { throw "Python executable not found: $Python" }

$LocalConfigScript = Join-Path $SharedRoot "backend\providers\local_llm\config.py"
$PlannerConfigScript = Join-Path $SharedRoot "backend\providers\planner_llm\config.py"
$LlamaServer = Join-Path $SharedRoot "engines\llama.cpp\llama-server.exe"
foreach ($required in @($LocalConfigScript, $PlannerConfigScript, $LlamaServer)) {
    if (-not (Test-Path $required)) { throw "Shared runtime asset not found: $required" }
}

$LocalConfig = (& $Python $LocalConfigScript --json | ConvertFrom-Json)
if ($LASTEXITCODE -ne 0) { throw "Could not read main repository local-model configuration." }
$PlannerConfig = (& $Python $PlannerConfigScript --json | ConvertFrom-Json)
if ($LASTEXITCODE -ne 0) { throw "Could not read main repository planner configuration." }

function Resolve-SharedModelPath {
    param([string]$ConfiguredPath)
    if ([System.IO.Path]::IsPathRooted($ConfiguredPath)) { return $ConfiguredPath }
    return Join-Path $SharedRoot $ConfiguredPath
}
$LocalModel = Resolve-SharedModelPath ([string]$LocalConfig.model_path)
$PlannerModel = Resolve-SharedModelPath ([string]$PlannerConfig.model_path)
if (-not (Test-Path $LocalModel)) { throw "Main repository local model not found: $LocalModel" }
if (-not (Test-Path $PlannerModel)) { throw "Main repository planner model not found: $PlannerModel" }

try {
    Invoke-WebRequest -Uri "http://127.0.0.1:8100/" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop | Out-Null
    throw "A Nova Control Centre is already running on port 8100. Close it before launching this test checkout."
} catch {
    if ($_.Exception.Message -like "A Nova Control Centre is already running*") { throw }
}

$EnvironmentNames = @(
    "NOVA_ROOT", "NOVA_PYTHON", "NOVA_LLAMA_SERVER",
    "LOCAL_LLM_MODEL", "LOCAL_LLM_SERVER", "PLANNER_LLM_MODEL",
    "PLANNER_LLM_SERVER", "NOVA_BACKEND_URL"
)
$PreviousEnvironment = @{}
foreach ($name in $EnvironmentNames) {
    $PreviousEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}

try {
    $env:NOVA_ROOT = $NovaRoot
    $env:NOVA_PYTHON = $Python
    $env:NOVA_LLAMA_SERVER = $LlamaServer
    $env:LOCAL_LLM_MODEL = $LocalModel
    $env:LOCAL_LLM_SERVER = "http://127.0.0.1:8080"
    $env:PLANNER_LLM_MODEL = $PlannerModel
    $env:PLANNER_LLM_SERVER = "http://127.0.0.1:8081"
    $env:NOVA_BACKEND_URL = "http://127.0.0.1:8000"

    Write-Host "Nova test checkout (single-instance fixed ports):"
    Write-Host "  Source   : $NovaRoot"
    Write-Host "  Control  : http://127.0.0.1:8100"
    Write-Host "  Backend  : http://127.0.0.1:8000"
    Write-Host "  Frontend : http://127.0.0.1:5173"
    Write-Host "  Local LLM: http://127.0.0.1:8080"
    Write-Host "  Planner  : http://127.0.0.1:8081"
    Write-Host "  Model    : $LocalModel"
    Write-Host "Stop any currently running Nova Control Center before launching this checkout."

    & (Join-Path $PSScriptRoot "start_control.ps1")
}
finally {
    foreach ($name in $EnvironmentNames) {
        $value = $PreviousEnvironment[$name]
        if ($null -eq $value) {
            Remove-Item "Env:$name" -ErrorAction SilentlyContinue
        } else {
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}
