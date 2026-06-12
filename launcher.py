import os
import sys
import subprocess
import shutil

if getattr(sys, 'frozen', False):
    BASE = os.path.dirname(os.path.abspath(sys.executable))
else:
    BASE = os.path.dirname(os.path.abspath(__file__))

VENV_PYTHON = os.path.join(BASE, "venv", "Scripts", "python.exe")


def rename_lora(name):
    path = os.path.join(BASE, "ui", "tob", "ComfyUI", "models", "loras", name)
    if os.path.exists(path):
        os.rename(path, path + ".incompatible")
        print(f"[OK] {name} renombrado a .incompatible")


def main():
    try:
        os.chdir(BASE)
        rename_lora("Flux%20Klein%20-%20NSFW%20v2.safetensors")
        rename_lora("flux2klein_nsfw.safetensors")
        if not os.path.exists(VENV_PYTHON):
            input(f"ERROR: No se encuentra {VENV_PYTHON}\nPresiona Enter para salir...")
            return 1
        subprocess.run([VENV_PYTHON, "run.py"], cwd=BASE)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        input(f"ERROR: {e}\nPresiona Enter para salir...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
