from distutils.core import setup
import py2exe
import os

# Obtener todos los archivos del directorio raíz del proyecto
root_dir = "D:/PROJECTS/AUTOAUTO"
include_files = [(os.path.join(root_dir, f), f) for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

# Agregar manualmente las rutas de las DLLs y otras dependencias necesarias
include_files += [
    "D:/PROJECTS/AUTOAUTO/.venv",
    "D:/PROJECTS/AUTOAUTO/checkpoints",
    "D:/PROJECTS/AUTOAUTO/clip",
    "D:/PROJECTS/AUTOAUTO/DEP",
    "D:/PROJECTS/AUTOAUTO/extensionsbuiltin",
    "D:/PROJECTS/AUTOAUTO/flagged",
    "D:/PROJECTS/AUTOAUTO/models",
    "D:/PROJECTS/AUTOAUTO/out-voice",
    "D:/PROJECTS/AUTOAUTO/output",
    "D:/PROJECTS/AUTOAUTO/roop",
    "D:/PROJECTS/AUTOAUTO/src",
    "D:/PROJECTS/AUTOAUTO/ui",
    "D:/PROJECTS/AUTOAUTO/dll/libpq.dll",
    "D:/PROJECTS/AUTOAUTO/dll/Qt53DAnimation.dll",
    "D:/PROJECTS/AUTOAUTO/dll/Qt53DCore.dll",
    "D:/PROJECTS/AUTOAUTO/dll/Qt53DExtras.dll",
    "D:/PROJECTS/AUTOAUTO/dll/Qt53DInput.dll",
    "D:/PROJECTS/AUTOAUTO/dll/Qt53DLogic.dll",
    "D:/PROJECTS/AUTOAUTO/dll/Qt53DQuickScene2D.dll",
    "D:/PROJECTS/AUTOAUTO/dll/Qt53DRender.dll",
    "D:/PROJECTS/AUTOAUTO/dll/Qt5MultimediaQuick.dll",
    "D:/PROJECTS/AUTOAUTO/dll/Qt5WebEngine.dll",
    "D:/PROJECTS/AUTOAUTO/dll/nvinfer.dll",
    "D:/PROJECTS/AUTOAUTO/dll/nvinfer_plugin.dll",
    "D:/PROJECTS/AUTOAUTO/dll/nvonnxparser.dll",
    "D:/PROJECTS/AUTOAUTO/dll/tbb12.dll",
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/bin/cublas64_11.dll",
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/bin/cublasLt64_11.dll",
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1",
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDNN",
    "D:/PROJECTS/AUTOAUTO/.venv/Lib/site-packages/tensorflow",
    "D:/PROJECTS/AUTOAUTO/dll/avcodec-58.dll",
    "D:/PROJECTS/AUTOAUTO/dll/avcodec-59.dll",
    "D:/PROJECTS/AUTOAUTO/dll/avcodec-60.dll",
    "D:/PROJECTS/AUTOAUTO/dll/avdevice-58.dll",
    "D:/PROJECTS/AUTOAUTO/dll/avdevice-59.dll",
    "D:/PROJECTS/AUTOAUTO/dll/avdevice-60.dll",
    "D:/PROJECTS/AUTOAUTO/dll/avfilter-7.dll",
    "D:/PROJECTS/AUTOAUTO/dll/avfilter-8.dll",
    "D:/PROJECTS/AUTOAUTO/dll/avfilter-9.dll",
    "D:/PROJECTS/AUTOAUTO/dll/avformat-58.dll",
    "D:/PROJECTS/AUTOAUTO/dll/avformat-59.dll",
    "D:/PROJECTS/AUTOAUTO/dll/avformat-60.dll",
    "D:/PROJECTS/AUTOAUTO/dll/avutil-56.dll",
    "D:/PROJECTS/AUTOAUTO/dll/avutil-57.dll",
    "D:/PROJECTS/AUTOAUTO/dll/avutil-58.dll"
]

setup(
    name="AUTODEEP",
    version="1.0",
    description="DEEPLEARN APP",
    options={
        "py2exe": {
            "includes": ["os", "sys", "ctypes", "threadpoolctl", "warnings"],
            "bundle_files": 1,
            "compressed": True,
            "optimize": 2,
            "dist_dir": "dist",
            "dll_excludes": ["w9xpopen.exe"]
        }
    },
    data_files=include_files,
    console=[{"script": "run.py"}],
    zipfile=None,
)