@echo off
echo ========================================
echo Descargando Z-Image Turbo GGUF (Q4_K_M)
echo ========================================
set "HF=https://huggingface.co/jayn7/Z-Image-Turbo-GGUF/resolve/main"

echo Descargando modelo GGUF...
curl -L -C - -o "D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models\unet\z_image_turbo-Q4_K_M.gguf" "%HF%/z_image_turbo-Q4_K_M.gguf"

echo.
echo Descargando completada!
pause