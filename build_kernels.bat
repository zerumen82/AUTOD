@echo off
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
cd /d D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\custom_nodes\hart\hart\kernels
python setup.py build_ext --inplace
