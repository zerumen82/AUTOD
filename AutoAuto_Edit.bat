@echo off
cd /d "%~dp0"
rem v5.1285 - con el Python del sistema (3.11 + gradio 3.41) el arranque
rem moria en demo.launch(css=...): TypeError: got an unexpected keyword
rem argument 'css' (log 2026-11-03 14:19:24). Se usa el venv del proyecto,
rem que es donde estan torch 2.8 y gradio 6.13.
if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" run.py %*
) else (
    python run.py %*
)
if errorlevel 1 pause