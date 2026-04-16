# Revert changes in ComfyUI comfy_env: uninstall opencv-contrib
param(
    [string]$envPath = "ui\tob\ComfyUI\comfy_env"
)
$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
$venv = Join-Path $root $envPath
if (-Not (Test-Path $venv)) {
    Write-Host "Error: venv no encontrado: $venv" -ForegroundColor Red
    exit 1
}
$activate = Join-Path $venv "Scripts\Activate.ps1"
& $activate

Write-Host "Desinstalando opencv-contrib-python desde: $venv" -ForegroundColor Yellow
pip uninstall -y opencv-contrib-python

# Asegurarse de que cv2 vuelva a importarse
try {
    python -c "import cv2; print('cv2 version:', cv2.__version__, 'ximgproc:', hasattr(cv2,'ximgproc'))"
} catch {
    Write-Host "cv2 no está instalado - reinstalando headless" -ForegroundColor Yellow
    pip install opencv-python-headless==4.11.0.86
}

Write-Host "Reversión completa. Verifica con: & \"$venv\Scripts\Activate.ps1\"; python -c \"import cv2; print(cv2.__version__, hasattr(cv2,'ximgproc'))\"" -ForegroundColor Green
