import subprocess
import shutil
import os

venv = r"D:\PROJECTS\AUTOAUTO\venv"

# 1. Uninstall ALL torch packages
packages = ["torch", "torchvision", "torchaudio", "xformers", "numpy"]
for pkg in packages:
    subprocess.run([f"{venv}\\Scripts\\pip.exe", "uninstall", pkg, "-y"], capture_output=True)

# 2. Remove ALL torch directories
dirs_to_remove = [
    f"{venv}\\Lib\\site-packages\\torch",
    f"{venv}\\Lib\\site-packages\\torchvision",
    f"{venv}\\Lib\\site-packages\\torchaudio", 
    f"{venv}\\Lib\\site-packages\\xformers",
    f"{venv}\\Lib\\site-packages\\~torch",
]

for d in dirs_to_remove:
    if os.path.exists(d):
        shutil.rmtree(d)
        print(f"Removed: {d}")

print("Cleaned torch. Now installing...")

# 3. Install numpy first (but not numpy 2!)
subprocess.run([f"{venv}\\Scripts\\pip.exe", "install", "numpy<2"], capture_output=True)

# 4. Install torch with CUDA
subprocess.run([f"{venv}\\Scripts\\pip.exe", "install", "torch==2.4.1+cu121", "--index-url", "https://download.pytorch.org/whl/cu121"], capture_output=True)

# 5. Install torchvision
subprocess.run([f"{venv}\\Scripts\\pip.exe", "install", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu121"], capture_output=True)

print("Done. Checking...")

# 6. Check
result = subprocess.run([f"{venv}\\Scripts\\python.exe", "-c", "import torch; print('OK:', torch.__version__, 'CUDA:', torch.version.cuda)"],
                       capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("ERROR:", result.stderr[:500])
