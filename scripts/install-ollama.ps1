$ErrorActionPreference = 'Stop'

if (Get-Command ollama -ErrorAction SilentlyContinue) {
    return
}

$answer = Read-Host 'Ollama is not installed. Install automatically now? (y/N)'
if ($answer -notin @('y','Y','yes','YES')) {
    Write-Host 'Please install Ollama from https://ollama.com/download and rerun the script.' -ForegroundColor Yellow
    exit 1
}

Write-Host 'Attempting to download Ollama installer...'
$installerPath = Join-Path $env:TEMP 'OllamaSetup.exe'
Invoke-WebRequest -Uri 'https://ollama.com/download/OllamaSetup.exe' -OutFile $installerPath
Write-Host "Installer downloaded to $installerPath"
Write-Host 'Launching installer (follow on-screen instructions)...'
Start-Process -FilePath $installerPath -Wait
Write-Host 'Ollama installation process invoked. Make sure the Ollama app is running before launching Atlas.'
