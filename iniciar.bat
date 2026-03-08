@echo off
chcp 65001 >nul

cd /d "%~dp0"

if not exist "%~dp0venv\Scripts\python.exe" (
    echo Error: No se encuentra venv
    pause
    exit /b
)

"%~dp0venv\Scripts\python.exe" run.py
pause
