import subprocess
import sys
import shutil
import os

venv_python = r"D:\PROJECTS\AUTOAUTO\venv\Scripts\python.exe"

# Uninstall everything
subprocess.run([venv_python, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "xformers", "-y"], 
               capture_output=True)

# Remove torch directories
torch_dirs = [
    r"D:\PROJECTS\AUTOAUTO\venv\lib\site-packages\torch",
    r"D:\PROJECTS\AUTOAUTO\venv\lib\site-packages\torchvision", 
    r"D:\PROJECTS\AUTOAUTO\venv\lib\site-packages\torchaudio"
]

for d in torch_dirs:
    if os.path.exists(d):
        shutil.rmtree(d)
        print(f"Removed: {d}")

# Install fresh torch
result = subprocess.run([venv_python, "-m", "pip", "install", "torch==2.4.1+cu121", 
                        "--index-url", "https://download.pytorch.org/whl/cu121"],
                       capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("ERROR:", result.stderr)

# Check torch
result2 = subprocess.run([venv_python, "-c", "import torch; print(torch.__version__, torch.version.cuda)"],
                         capture_output=True, text=True)
print("Torch check:", result2.stdout, result2.stderr)
