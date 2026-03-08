# PowerShell helper to activate the recommended venv and launch AUTO-DEEP
param(
    [string]$env = "venv"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
# Project root is one level above scripts folder
$root = Split-Path -Parent $scriptDir
$venvPath = Join-Path $root $env

if (-Not (Test-Path $venvPath)) {
    Write-Host "No se encontró el venv en: $venvPath" -ForegroundColor Yellow
    exit 1
}

$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
if (-Not (Test-Path $activateScript)) {
    Write-Host "No se encontró Activate.ps1 en: $activateScript" -ForegroundColor Yellow
    exit 1
}

Write-Host "Activando entorno virtual: $venvPath" -ForegroundColor Cyan
& $activateScript

Write-Host "Instalando dependencias si faltan (opcional)" -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "Ejecutando AUTO-DEEP..." -ForegroundColor Green
python run.py

# Recordar: para salir del venv usar 'deactivate' en cmd o PowerShell
