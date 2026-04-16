@echo off
echo Actualizando pyav en venv de ComfyUI...
cd /d "%~dp0..\ui\tob\ComfyUI"
venv\Scripts\python.exe -m pip install --upgrade pyav
pause
