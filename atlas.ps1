$ErrorActionPreference = 'Stop'

param(
    [switch]$InstallCommand
)

$repoDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoDir

if ($InstallCommand) {
    $windowsApps = Join-Path $env:LOCALAPPDATA 'Microsoft\WindowsApps'
    if (-not (Test-Path $windowsApps)) {
        $windowsApps = Join-Path $env:USERPROFILE 'bin'
        if (-not (Test-Path $windowsApps)) {
            New-Item -ItemType Directory -Path $windowsApps | Out-Null
        }
    }

    $cmdPath = Join-Path $windowsApps 'atlas.cmd'
    $escaped = $repoDir.Replace('"', '\"')
    $content = "@echo off`npowershell -ExecutionPolicy Bypass -File \"$escaped\atlas.ps1\" %*"
    Set-Content -Path $cmdPath -Value $content -Encoding ASCII
    Write-Host "Atlas command installed at $cmdPath" -ForegroundColor Green
    Write-Host "Make sure $windowsApps is on your PATH, then run 'atlas' from a new terminal." -ForegroundColor Green
    exit 0
}

function Ensure-Ollama {
    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
        if (Test-Path 'scripts\install-ollama.ps1') {
            & scripts\install-ollama.ps1
        } else {
            Write-Host "WARNING: Ollama is not installed. Download it from https://ollama.com/download" -ForegroundColor Yellow
            Write-Host "         and make sure the daemon is running on http://localhost:11434." -ForegroundColor Yellow
        }
    }
}

function Ensure-Venv {
    if (-not (Test-Path '.venv')) {
        Write-Host 'Setting up the Atlas environment (first run)...'
        & scripts\setup.ps1
    }
}

Ensure-Ollama
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    if (Test-Path 'scripts\pull-models.ps1') {
        & scripts\pull-models.ps1
    }
}
Ensure-Venv
& scripts\run.ps1 @Args
