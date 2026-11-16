"""Arranca el ComfyUI local (8188) para las pruebas reales de esta sesion."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ui.tabs.comfy_launcher import start  # noqa: E402

if __name__ == "__main__":
    ok, msg, port = start(port=8188, directly_run=True)
    print(f"[diag] start -> ok={ok} msg={msg} port={port}", flush=True)
