@echo off
chcp 65001 >nul
echo ============================================
echo Descarga de Modelos - Animate Image
echo ============================================
echo.
echo Se descargaran: Wan 2.2 + VAE + LTX Video
echo Tamanio total: ~14 GB
echo Tiempo estimado: 30-60 minutos
echo.
pause

set "BASE=%~dp0..\ui\tob\ComfyUI\models"

echo.
echo [1/5] Creando carpetas...
mkdir "%BASE%\diffusion_models\Wan2.2-I2V-14B-480P-Q4_K_M" 2>nul
mkdir "%BASE%\vae" 2>nul
mkdir "%BASE%\checkpoints\ltx-video-0.9.5" 2>nul

echo.
echo [2/5] Descargando Wan 2.2 (8GB)...
echo    URL: https://huggingface.co/Comfy-UI/Wan2.2-I2V-14B-480P-Q4_K_M
powershell -Command "cd '%BASE%\diffusion_models\Wan2.2-I2V-14B-480P-Q4K_M'; (New-Object System.Net.WebClient).DownloadFile('https://huggingface.co/Comfy-UI/Wan2.2-I2V-14B-480P-Q4_K_M/resolve/main/Wan2.2-I2V-14B-480P-Q4_K_M.safetensors', 'Wan2.2-I2V-14B-480P-Q4_K_M.safetensors')"

echo.
echo [3/5] Descargando VAE de Wan (50MB)...
powershell -Command "cd '%BASE%\vae'; (New-Object System.Net.WebClient).DownloadFile('https://huggingface.co/Comfy-UI/Wan2.2-VAE/resolve/main/Wan2.2_VAE.safetensors', 'Wan2.2_VAE.safetensors')"

echo.
echo [4/5] Descargando LTX Video 0.9.5 (6GB)...
echo    URL: https://huggingface.co/Comfy-UI/LTX-Video-0.9.5
powershell -Command "cd '%BASE%\checkpoints\ltx-video-0.9.5'; (New-Object System.Net.WebClient).DownloadFile('https://huggingface.co/Comfy-UI/LTX-Video-0.9.5/resolve/main/ltxvideo.safetensors', 'ltxvideo.safetensors')"

echo.
echo [5/5] Verificando archivos...
dir "%BASE%\diffusion_models\Wan2.2-I2V-14B-480P-Q4_K_M\*.safetensors" 2>nul
dir "%BASE%\vae\*.safetensors" 2>nul
dir "%BASE%\checkpoints\ltx-video-0.9.5\*.safetensors" 2>nul

echo.
echo ============================================
echo INSTALACION COMPLETADA
echo ============================================
echo.
echo Pasos siguientes:
echo 1. Reinicia la aplicacion: python run.py
echo 2. Ve a la pestana "Imagine"
echo 3. Pulsa "Detectar Modelos"
echo.
echo Si alguna descarga fallo, ejecuta de nuevo este script.
echo.
pause
