@echo off
echo ============================================================
echo    DESCARGA DE MODELOS FACE ENHANCEMENT (2025)
echo ============================================================
echo.
echo Modelos disponibles:
echo   1. CodeFormer (RECOMENDADO - Mejor calidad 2025)
echo   2. RestoreFormer++ (Alternativa)
echo   3. huggingface-cli (Mejor metodo - requiere pip install)
echo   4. Salir
echo.
echo ============================================================

set /p choice="Elige una opcion (1-4): "

if "%choice%"=="1" goto download_codeformer
if "%choice%"=="2" goto download_restoreformer
if "%choice%"=="3" goto use_huggingface_cli
if "%choice%"=="4" goto end
goto invalid

:download_codeformer
echo.
echo ============================================================
echo Descargando CodeFormer...
echo ============================================================
echo.
python install_enhancer_models.py
if errorlevel 1 (
    echo.
    echo La descarga automatica fallo. Intenta con huggingface-cli:
    echo   pip install huggingface_hub
    echo   huggingface-cli download sczhou/CodeFormer CodeFormerv0.1.onnx --local-dir ./roop/models/CodeFormer
)
goto end

:download_restoreformer
echo.
echo ============================================================
echo Para descargar RestoreFormer++, usa huggingface-cli:
echo ============================================================
echo.
echo   pip install huggingface_hub
echo   huggingface-cli download sczhou/CodeFormer restoreformer_plus_plus.onnx --local-dir ./roop/models
echo.
goto end

:use_huggingface_cli
echo.
echo ============================================================
echo Usando huggingface-cli (RECOMENDADO)
echo ============================================================
echo.
echo 1. Instala huggingface_hub si no lo tienes:
echo    pip install huggingface_hub
echo.
echo 2. Descarga CodeFormer:
echo    huggingface-cli download sczhou/CodeFormer CodeFormerv0.1.onnx --local-dir ./roop/models/CodeFormer
echo.
echo 3. Descarga RestoreFormer++:
echo    huggingface-cli download sczhou/CodeFormer restoreformer_plus_plus.onnx --local-dir ./roop/models
echo.
pause
goto end

:invalid
echo.
echo Opcion invalida. Intenta nuevamente.
goto end

:end
echo.
echo ============================================================
echo.
pause
