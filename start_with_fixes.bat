@echo off
echo ========================================
echo INICIANDO AUTO-AUTO CON FIXES APLICADOS
echo ========================================
echo.

echo [PRE] Verificando LoRAs incompatibles...
if exist "D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models\loras\Flux%%20Klein%%20-%%20NSFW%%20v2.safetensors" (
    ren "D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models\loras\Flux%%20Klein%%20-%%20NSFW%%20v2.safetensors" "Flux%%20Klein%%20-%%20NSFW%%20v2.safetensors.incompatible"
    echo [OK] LoRA NSFW v2 renombrado a .incompatible
)
if exist "D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models\loras\flux2klein_nsfw.safetensors" (
    ren "D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models\loras\flux2klein_nsfw.safetensors" "flux2klein_nsfw.safetensors.incompatible"
    echo [OK] LoRA flux2klein_nsfw renombrado a .incompatible
)
echo.

echo [START] Iniciando ComfyUI + Gradio...
cd /d "D:\PROJECTS\AUTOAUTO"
call venv\Scripts\activate.bat
python run.py

pause
