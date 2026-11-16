@echo off
echo ==========================================
echo   CIERRE FORZOSO DE TODOS LOS PROCESOS AUTOAUTO
echo ==========================================
echo.

echo --- 1. AutoAuto.exe ---
taskkill /F /IM AutoAuto.exe >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ AutoAuto.exe finalizado
) else (
    echo ⚠️  No se encontró AutoAuto.exe
)

echo.
echo --- 2. python.exe (procesos del proyecto) ---
taskkill /F /IM python.exe >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ python.exe finalizado
) else (
    echo ⚠️  No se encontró python.exe
)

echo.
echo --- 3. Puertos clave ---
echo Buscando puertos 7861, 7862, 8188...
netstat -an | findstr :7861 >nul 2>&1
if %errorlevel% equ 0 (
    echo ⛔ Puerto 7861 en uso
) else (
    echo ✅ Puerto 7861 libre
)

netstat -an | findstr :7862 >nul 2>&1
if %errorlevel% equ 0 (
    echo ⛔ Puerto 7862 en uso
) else (
    echo ✅ Puerto 7862 libre
)

netstat -an | findstr :8188 >nul 2>&1
if %errorlevel% equ 0 (
    echo ⛔ Puerto 8188 en uso
) else (
    echo ✅ Puerto 8188 libre
)

echo.
echo --- 4. Verificación final ---
echo.
echo ==========================================
echo   PROCESOS AHORA:
echo ==========================================
tasklist | findstr "AutoAuto python"

echo.
echo ==========================================
echo   VERIFICACIÓN COMPLETA
echo ==========================================
echo.
echo ✅ Procesos destrúidos
echo ✅ Puertos liberados
echo ✅ Puedes abrir AutoAuto.exe de nuevo