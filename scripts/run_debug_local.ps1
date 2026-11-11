$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
if (-not (Test-Path ".venv")) {
    python -m venv .venv
}
& ".venv\Scripts\python.exe" -m pip install --upgrade pip
& ".venv\Scripts\python.exe" -m pip install -e ".[dev]"
& ".venv\Scripts\python.exe" -m pytest
& ".venv\Scripts\python.exe" -m fconcord run --config configs/debug.yaml
