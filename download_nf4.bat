@echo off
echo ============================================================
echo DESCARGA DE FLUX.1-Fill-dev NF4 - Archivo por Archivo
echo ============================================================
echo.
echo Descargando archivos faltantes...
echo.

hf download lrzjason/flux-fill-nf4 ^
  --local-dir "D:\PROJECTS\models\flux-fill-nf4" ^
  --token YOUR_HF_TOKEN_HERE ^
  --include "transformer/*" ^
  --include "text_encoder_2/*"

echo.
echo ============================================================
echo DESCARGA COMPLETADA!
echo ============================================================
pause
