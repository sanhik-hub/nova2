param(
    [int]$Port = 8080,
    [int]$ContextSize = 32768,
    [int]$Threads = 6,
    [int]$BatchSize = 128,
    [int]$GpuLayers = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$LlamaServer = Join-Path $RepoRoot "engines\llama.cpp\llama-server.exe"
$Model = Join-Path $RepoRoot "models\chemistry_test\base\Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf"

if (-not (Test-Path $LlamaServer)) {
    throw "llama-server.exe was not found at $LlamaServer"
}

if (-not (Test-Path $Model)) {
    throw "Qwen3-4B-Instruct-2507 GGUF was not found at $Model"
}

$existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    throw "Port $Port is already in use. Stop the existing llama-server before starting this instance."
}

Write-Host "Starting Qwen3-4B-Instruct-2507 for Nova..."
Write-Host "Native tool calling: enabled (--jinja)"
Write-Host "Reasoning/thinking: disabled (--reasoning off)"
Write-Host "Endpoint: http://127.0.0.1:$Port"
Write-Host "Model: $Model"

& $LlamaServer `
    -m $Model `
    --host 127.0.0.1 `
    --port $Port `
    --ctx-size $ContextSize `
    --threads $Threads `
    --batch-size $BatchSize `
    --n-gpu-layers $GpuLayers `
    --jinja `
    --reasoning off
