$ErrorActionPreference = 'Stop'

if (-not (Test-Path '.venv')) {
    Write-Error 'Virtual environment not found. Run scripts\setup.ps1 first.'
}

$activate = Join-Path '.venv' 'Scripts' 'Activate.ps1'
. $activate

python -m atlas_main.cli @Args
