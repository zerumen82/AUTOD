@echo off
echo Instalando dependencias EasyUse en venv de ComfyUI...
cd /d "%~dp0..\ui\tob\ComfyUI"
venv\Scripts\python.exe -m pip install diffusers accelerate "clip_interrogator>=0.6.0" lark onnxruntime opencv-python-headless sentencepiece spandrel matplotlib peft
echo.
echoListo!
pause
