import subprocess
import shutil
import os
import sys

venv = r"D:\PROJECTS\AUTOAUTO\venv\Scripts"

# 1. Uninstall everything
pkgs = ["torch", "torchvision", "torchaudio", "xformers", "numpy"]
for pkg in pkgs:
    subprocess.run([f"{venv}\\pip.exe", "uninstall", pkg, "-y"], capture_output=True)

# 2. Remove directories
dirs = [
    r"D:\PROJECTS\AUTOAUTO\venv\lib\site-packages\torch",
    r"D:\PROJECTS\AUTOAUTO\venv\lib\site-packages\torchvision",
    r"D:\PROJECTS\AUTOAUTO\venv\lib\site-packages\torchaudio",
    r"D:\PROJECTS\AUTOAUTO\venv\lib\site-packages\xformers",
]
for d in dirs:
    if os.path.exists(d):
        shutil.rmtree(d)
        print(f"Removed: {d}")

# 3. Install in correct order: numpy first, then torch, then xformers
pip = f"{venv}\\pip.exe"
python = f"{venv}\\python.exe"

# Install numpy < 2 first
subprocess.run([pip, "install", "numpy<2"], capture_output=True)

# Install torch with CUDA
subprocess.run([pip, "install", "torch==2.4.1+cu121", "--index-url", "https://download.pytorch.org/whl/cu121"], capture_output=True)

# Install torchvision
subprocess.run([pip, "install", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu121"], capture_output=True)

# Install xformers for torch 2.4
subprocess.run([pip, "install", "xformers==0.0.27"], capture_output=True)

# Check final state
result = subprocess.run([python, "-c", "import torch; import xformers; print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda); print('xFormers:', xformers.__version__)"], 
                       capture_output=True, text=True)
print(result.stdout)
print(result.stderr)
