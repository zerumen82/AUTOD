@echo off
chcp 65001 >nul
echo ==========================================================
echo   Generador de Enlace APK Directo - AutoAuto Image Editor
echo ==========================================================
echo.

cd /d "%~dp0"

echo [INFO] Detectando tu dirección IP local y preparando servidor...
"%~dp0venv\Scripts\python.exe" android_app\manifest_generator.py

echo.
echo ==========================================================
echo  📱 CÓMO OBTENER TU APK EN 1 CLIC (SIN ALGO EXTRA EN TU PC):
echo ==========================================================
echo  1. Inicia AutoAuto.exe en tu PC normalmente.
echo  2. En la pantalla del PC o consola verás tu URL local (ej: http://192.168.1.XX:7861)
echo     o tu enlace público Gradio (ej: https://xxxx.gradio.live).
echo  3. Abre en tu navegador (PC o Móvil) esta web gratuita:
echo     👉 https://www.webintoapp.com/
echo  4. Pega tu URL de AutoAuto y pulsa "Create APK".
echo  5. ¡Descarga e instala la APK en tu Android!
echo ==========================================================
echo.
pause
