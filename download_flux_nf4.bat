@echo off
echo ============================================================
echo DESCARGA DE FLUX.1-Fill-dev NF4
echo ============================================================
echo.
echo Descargando modelo completo desde HuggingFace...
echo Esto puede tardar 30-60 minutos dependiendo de tu conexion.
echo.
echo Token: YOUR_HF_TOKEN_HERE
echo.
pause

python -c "
from huggingface_hub import snapshot_download
import os

FLUX_NF4_DIR = r'D:\PROJECTS\models\FLUX.1-fill-dev-NF4'
HF_TOKEN = 'YOUR_HF_TOKEN_HERE'

print('Iniciando descarga...')
print('Destino: ' + FLUX_NF4_DIR)
print()

try:
    downloaded_path = snapshot_download(
        repo_id='black-forest-labs/FLUX.1-Fill-dev',
        local_dir=FLUX_NF4_DIR,
        local_dir_use_symlinks=False,
        token=HF_TOKEN,
        resume_download=True,
        ignore_patterns=['*.pt', '*.pth'],
    )
    print()
    print('DESCARGA COMPLETADA!')
    print('Modelo en: ' + downloaded_path)
except Exception as e:
    print('ERROR: ' + str(e))
"

echo.
pause
