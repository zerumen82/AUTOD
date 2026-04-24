@echo off
chcp 65001 >nul
echo ============================================
echo Descarga de LTX Video 0.9.5 (RECOMENDADO 8GB)
echo ============================================
echo.
echo Esta version funciona directamente con ComfyUI.
echo Tamaño: ~6 GB
echo.

set "BASE=%~dp0..\ui\tob\ComfyUI\models"
mkdir "%BASE%\checkpoints\ltx-video-0.9.5" 2>nul

echo Descargando LTX Video 0.9.5...
echo URL: https://huggingface.co/Comfy-UI/LTX-Video-0.9.5
echo.

powershell -Command "cd '%BASE%\checkpoints\ltx-video-0.9.5'; (New-Object System.Net.WebClient).DownloadFile('https://huggingface.co/Comfy-UI/LTX-Video-0.9.5/resolve/main/ltxvideo.safetensors', 'ltxvideo.safetensors')"

echo.
echo Verificando...
dir "%BASE%\checkpoints\ltx-video-0.9.5\*.safetensors"

echo.
echo ============================================
echo COMPLETADO
echo ============================================
echo.
echo 1. Reinicia la aplicación: python run.py
echo 2. Ve a "Imagine" -> "Detectar Modelos"
echo.
pause
