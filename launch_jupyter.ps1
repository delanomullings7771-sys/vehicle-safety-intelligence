$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project environment not found: $Python"
}

Set-Location -LiteralPath $ProjectRoot
$env:IPYTHONDIR = Join-Path $ProjectRoot ".ipython"
$env:JUPYTER_CONFIG_DIR = Join-Path $ProjectRoot ".jupyter"
New-Item -ItemType Directory -Force -Path $env:IPYTHONDIR, $env:JUPYTER_CONFIG_DIR | Out-Null
& $Python -m jupyter lab
