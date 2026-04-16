Instrucciones rápidas para preparar `dll/cudnn8/` (cuDNN 8)

1) Descarga la versión de cuDNN 8 para Windows desde el sitio de NVIDIA (requiere cuenta): https://developer.nvidia.com/cudnn
   - Recomiendo cuDNN 8.9.x compatible con CUDA 11.8 si prefieres mantener compatibilidad histórica, o la versión que coincida con tu objetivo de compatibilidad.

2) Guarda el ZIP en tu equipo (por ejemplo: C:\Users\Usuario\Downloads\cudnn-windows-x86_64-8.9.1.zip)

3) Ejecuta desde la raíz del proyecto (con el entorno virtual activo):
   python scripts/install_local_cudnn8.py "C:\ruta\a\cudnn-windows-x86_64-8.x.x.zip"

4) Verifica:
   python verificar_cudnn_sistema.py

Notas:
- Si prefieres, puedo intentar descargar automáticamente el ZIP pero normalmente NVIDIA requiere autenticación; por eso el script espera un ZIP local.
- Una vez los DLLs de cuDNN 8 estén en `dll/cudnn8/`, `run.py` los cargará primero para forzar compatibilidad con ONNX Runtime 1.23.
