$ErrorActionPreference = 'Stop'

function Get-PythonCommand {
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return 'python'
    }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return 'py'
    }
    Write-Error 'Python 3 is required but was not found. Install it from https://www.python.org/downloads/.'
}

$pythonCmd = Get-PythonCommand

if (-not (Test-Path '.venv')) {
    Write-Host 'Creating local virtual environment (.venv)...'
    if ($pythonCmd -eq 'py') {
        & py -3 -m venv .venv
    } else {
        & python -m venv .venv
    }
}

$activate = Join-Path '.venv' 'Scripts' 'Activate.ps1'
. $activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

if (Get-Command ollama -ErrorAction SilentlyContinue) {
    if (Test-Path 'scripts\pull-models.ps1') {
        & scripts\pull-models.ps1
    }
}

Write-Host "`nSetup complete!"
Write-Host 'Run the assistant with: scripts\run.ps1'
