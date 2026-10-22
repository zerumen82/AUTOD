"""CPU check of the GLOBAL route with a measured body change (no render).

The global route measured the body's intensity flat (0.20), so the
final-state sentence was not written and the model answered with swimwear
(skin 0.2101). With the per-clause measurement (0.745) the state sentence is
written and the engine is the one that follows the text.

This captures, without a model:
  * the final prompt (the state sentence and the enforcement);
  * the engine recipe (the vision engine, no hole, no anchor);
  * that the geometry of the photo is preserved.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".autodeep_temp" / "42ebfcf365a6c3b5c1dbfdbf39c36c32f25938bfc7ac489c44cd9366e2b27b3a" / "19978049_13-35-33.jpg"
PROMPT = "la mujer esta desnuda, sin zapatos"


class RecordingClient:
    def __init__(self):
        self.loads, self.generations = [], []

    def load(self, *args, **kwargs):
        self.loads.append((args, kwargs))
        return True, "CPU double"

    def generate(self, **kwargs):
        self.generations.append(kwargs)
        photo = kwargs.get("image")
        return SimpleNamespace(image=photo.copy()), "CPU generation"


def main():
    photo = Image.open(SOURCE).convert("RGB")
    manager = ImgEditorManager()
    client = RecordingClient()
    with (patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
          patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
          patch.object(manager, "_release_local_gpu_before_comfy"),
          patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
                return_value=client)):
        output, message, _mask = manager.generate_intelligent(
            image=photo, prompt=PROMPT, engine="kontext",
            use_semantic=True, use_rewriter=True, enhance_faces=False,
            auto_upscale=False, seed=424242,
        )
    print(f"[Prompt] loads={client.loads}", flush=True)
    print(f"[Prompt] generations={len(client.generations)}", flush=True)
    if not client.generations:
        print("[Prompt] the route did not reach the engine", flush=True)
        return 1
    request = client.generations[0]
    prompt = str(request.get("prompt", ""))
    print(f"[Prompt] final prompt={prompt}", flush=True)
    checks = {
        "el motor con visión se pide explícitamente": any(
            kwargs.get("allow_vision_engine") for _args, kwargs in client.loads),
        "la frase de estado final está escrita": "is fully nude" in prompt,
        "el enforcement está escrito": "must be clearly visible" in prompt,
        "la instrucción está": "naked" in prompt,
        "sin ancla de latentes": bool(request.get("no_kontext")),
        "el texto manda": bool(request.get("prompt_authority")),
        "sin agujero (el marco se reinterpreta)": not request.get("mask_image"),
    }
    failures = [name for name, ok in checks.items() if not ok]
    for name, ok in checks.items():
        print(f"[Prompt] {'PASS' if ok else 'FAIL'} {name}", flush=True)
    print(f"[Prompt] salida {output.size if output is not None else None}, "
          f"mensaje: {message}", flush=True)
    print(f"[Prompt] {len(checks) - len(failures)}/{len(checks)} PASS; failures={failures}",
          flush=True)
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
