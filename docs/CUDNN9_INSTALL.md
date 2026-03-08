CUDNN9 local install (para build de ONNX Runtime)

Flujo recomendado:

1. Descarga cuDNN 9 desde la web de NVIDIA (requiere cuenta): https://developer.nvidia.com/cudnn
2. Guarda el ZIP en el repo, por ejemplo: `third_party/cudnn9.zip`
3. Ejecuta:
   ```
   python scripts/install_local_cudnn9.py --zip third_party/cudnn9.zip
   ```
4. El script extraerá y verificará `cudnnGetVersion`, y copiará el contenido a `dll/cudnn9/`.
5. Para compilar ONNX Runtime con CUDA/cuDNN, en la configuración CMake añade:
   -DCMAKE_TOOLCHAIN_FILE=<ruta_vcpkg>/scripts/buildsystems/vcpkg.cmake \
   -Donnxruntime_USE_CUDA=ON \
   -Donnxruntime_USE_CUDNN=ON \
   -DCUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2" \
   -Dcudnn_root="D:/PROJECTS/AUTOAUTO/dll/cudnn9"

Notas:
- Asegúrate de usar la versión de cuDNN que quieres compilar contra (evita mezclar cuDNN8 y cuDNN9 en PATH en tiempo de ejecución).
- Si necesitas que el script haga más (p. ej. copiar DLLs a onnxruntime capi directo o actualizar run.py), dímelo y lo añado.
