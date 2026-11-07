@echo off
cd /d "D:\PROJECTS\AUTOAUTO"
if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" run.py > log_autoauto.txt 2>&1
) else (
    python run.py > log_autoauto.txt 2>&1
)
timeout 30