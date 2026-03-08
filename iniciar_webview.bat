@echo off
chcp 65001 >nul
echo ====================
echo AUTO-DEEP v2.2.2 - PyWebView Mode
echo ====================
echo.

cd /d "%~dp0"

REM 1. Verificar Carpeta VENV
if not exist "%~dp0venv\Scripts\python.exe" (
    echo [ERROR] No se encuentra el entorno virtual en: %~dp0venv
    echo Por favor, verifica que la carpeta 'venv' existe.
    pause
    exit /b
)

REM 2. Limpiar archivos .pyc y __pycache__
echo [INFO] Limpiando archivos temporales (.pyc)...
if exist "%~dp0roop\__pycache__" rmdir /s /q "%~dp0roop\__pycache__" 2>nul
if exist "%~dp0ui\__pycache__" rmdir /s /q "%~dp0ui\__pycache__" 2>nul
if exist "%~dp0roop\*.pyc" del /q "%~dp0roop\*.pyc" 2>nul
if exist "%~dp0ui\*.pyc" del /q "%~dp0ui\*.pyc" 2>nul
echo [OK] Archivos temporales limpiados.

REM 3. Configurar CUDA
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
set PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%

echo [INFO] Entorno virtual detectado.
echo [INFO] Lanzando aplicación con pywebview...
echo.

echo [NOTA] Si pywebview no abre, la UI estará disponible en el navegador:
echo   - Busca la URL en la consola (ej: http://127.0.0.1:9000)
echo   - Copia y pega la URL en tu navegador
echo.

REM 4. Lanzar UI con pywebview
echo [OK] Iniciando aplicación...
echo.

REM Usar run.py que ahora incluye soporte para pywebview
"%~dp0venv\Scripts\python.exe" run.py

pause
