$ErrorActionPreference = 'Stop'

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Verbose 'Ollama not available; skipping model pulls.'
    return
}

$chatModel = if ($env:ATLAS_CHAT_MODEL) { $env:ATLAS_CHAT_MODEL } else { 'qwen2.5:latest' }
$embedModel = if ($env:ATLAS_EMBED_MODEL) { $env:ATLAS_EMBED_MODEL } else { 'mxbai-embed-large' }
$models = @($chatModel, $embedModel)

foreach ($model in $models) {
    if (-not $model) { continue }
    & ollama show $model > $null 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Model '$model' already present."
        continue
    }
    Write-Host "Pulling Ollama model '$model'..."
    & ollama pull $model
}
