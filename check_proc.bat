@echo off
echo ==========================================
echo   VERIFICACIÓN DE PROCESOS CARGados
echo ==========================================
echo.

echo --- Procesos AutoAuto ---
tasklist | findstr AutoAuto

echo.
echo --- Procesos Python ---
tasklist | findstr python

echo.
echo --- Procesos Gradio ---
tasklist | findstr "gradio" | findstr /i "server"

echo.
echo --- Puertos activos ---
netstat -an | findstr :7862
netstat -an | findstr :7861
netstat -an | findstr :8188

echo.
echo ==========================================
echo   FIN DE VERIFICACIÓN
echo ==========================================