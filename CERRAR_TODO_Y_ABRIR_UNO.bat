@echo off
cd /d "%~dp0"
echo Cerrando ventanas AutoAuto anteriores...
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":7861" ^| findstr "LISTENING"') do (
  echo Matando PID %%P en 7861...
  taskkill /F /PID %%P 2>nul
)
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":8188" ^| findstr "LISTENING"') do (
  echo Matando PID %%P en 8188 (ComfyUI)...
  taskkill /F /PID %%P 2>nul
)
timeout /t 3 /nobreak >nul
del /q logs\active_server.json 2>nul
echo.
echo Abriendo UNA sola ventana con codigo nuevo...
rem v5.1285 - con el Python del sistema (3.11 + gradio 3.41) el arranque
rem moria en demo.launch(css=...): TypeError: got an unexpected keyword
rem argument 'css' (log 2026-11-03 14:19:24). Solo el venv del proyecto
rem tiene gradio 6.13, que si acepta css.
if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" run.py
) else (
    python run.py
)
