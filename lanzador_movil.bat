@echo off
chcp 65001 >nul
echo ====================================================
echo   AUTO-AUTO — Modo Servidor Móvil (Android/Remoto)
echo ====================================================
echo.

cd /d "%~dp0"

if not exist "%~dp0venv\Scripts\python.exe" (
    echo [ERROR] No se encuentra el entorno virtual venv.
    pause
    exit /b
)

set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
set PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%

echo [INFO] Iniciando servidor Gradio para acceso local (WiFi) y remoto...
echo.
echo 📱 ACCESO DESDE TU MÓVIL ANDROID:
echo ----------------------------------------------------
echo  1. Red Local (WiFi de casa): Abre en Chrome del móvil la IP de este PC:
echo     Ejemplo: http://192.168.1.XX:7861
echo.
echo  2. Remoto / Fuera de casa: Si tienes server_share=true en config.yaml,
echo     copia la URL publica 'https://xxxx.gradio.live' que aparecerá abajo.
echo ----------------------------------------------------
echo.

"%~dp0venv\Scripts\python.exe" ui/main.py

pause
