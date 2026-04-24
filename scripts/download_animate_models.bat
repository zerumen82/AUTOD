@echo off
chcp 65001 >nul
echo ============================================
echo Descarga de Modelos para Animate Image
echo ============================================
echo.
echo Selecciona los modelos a descargar:
echo 1) Wan 2.2 (14B GGUF) - Máxima calidad (~8GB) + VAE
echo 2) LTX Video 0.9.5 - Velocidad (~6GB)
echo 3) Ambos (Wan + LTX)
echo.
set /p choice="Opción (1-3): "

cd /d "%~dp0.."
set "BASE=%~dp0..\ui\tob\ComfyUI\models"

if "%choice%"=="1" goto wan
if "%choice%"=="2" goto ltx
if "%choice%"=="3" (
    goto wan
    call :ltx
    goto end
)

echo Opción no válida.
pause
exit /b 1

:wan
echo.
echo [1/2] Creando carpetas para Wan 2.2...
mkdir "%BASE%\diffusion_models\Wan2.2-I2V-14B-480P-Q4_K_M" 2>nul
mkdir "%BASE%\vae" 2>nul

echo [2/2] Descargando Wan 2.2 (8GB)...
echo    Esto puede tardar ~20 minutos. No canceles.
powershell -Command "cd '%BASE%\diffusion_models\Wan2.2-I2V-14B-480P-Q4_K_M'; (New-Object System.Net.WebClient).DownloadFile('https://huggingface.co/Comfy-UI/Wan2.2-I2V-14B-480P-Q4_K_M/resolve/main/Wan2.2-I2V-14B-480P-Q4_K_M.safetensors', 'Wan2.2-I2V-14B-480P-Q4_K_M.safetensors')"

echo.
echo [3/3] Descargando VAE de Wan 2.2...
powershell -Command "cd '%BASE%\vae'; (New-Object System.Net.WebClient).DownloadFile('https://huggingface.co/Comfy-UI/Wan2.2-VAE/resolve/main/Wan2.2_VAE.safetensors', 'Wan2.2_VAE.safetensors')"

echo.
echo ✅ Wan 2.2 instalado!
goto end

:ltx
echo.
echo [1/2] Creando carpeta para LTX Video...
mkdir "%BASE%\checkpoints\ltx-video-0.9.5" 2>nul

echo [2/2] Descargando LTX Video 0.9.5 (6GB)...
echo    Esto puede tardar ~15 minutos.
powershell -Command "cd '%BASE%\checkpoints\ltx-video-0.9.5'; (New-Object System.Net.WebClient).DownloadFile('https://huggingface.co/Comfy-UI/LTX-Video-0.9.5/resolve/main/ltxvideo.safetensors', 'ltxvideo.safetensors')"

echo.
echo ✅ LTX Video instalado!
goto end

:end
echo.
echo ============================================
echo PASO SIGUIENTE
echo ============================================
echo.
echo 1. Reinicia la aplicación: python run.py
echo 2. Ve a la pestaña "Imagine"
echo 3. Pulsa "🔍 Detectar Modelos"
echo.
echo Si la descarga falla por timeout, ejecuta de nuevo el script.
echo.
pause
