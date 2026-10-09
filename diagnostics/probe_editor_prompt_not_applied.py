"""Reproduce the reported regression: the Image Editor ignores the instruction.

The user reports that neither a simple nor a compound instruction changes the
photo any more (default LongCat, no extra options). This probe runs the real
dispatch with the real segmenter and the real face analyser, and only the image
generator is a stub that always answers with a clearly changed sample.

Failing today means the pipeline discards a generated change (an early return
or a paste that keeps the source), which is the reported bug.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from roop.img_editor.img_editor_manager import ImgEditorManager


ROOT = Path(__file__).resolve().parents[1]
PHOTO = ROOT / "ui/tob/ComfyUI/input/fast_1790795484.png"

CASES = (
    ("el hombre debe ser calvo", "The man must be bald"),
    (
        "la mujer debe ir sin medias, descalza, vestida solo con un tanga",
        "the woman must go without stockings, barefoot, dressed only in a thong",
    ),
)


class ChangedClient:
    """Generator stub: answers every request with a visibly changed sample."""

    def __init__(self):
        self.calls = []

    def load(self, version):
        self.calls.append(("load", version))
        return True, "OK (stub)"

    def generate(self, **kwargs):
        self.calls.append(("generate", kwargs))
        image = kwargs.get("image")
        sample = np.asarray(image.convert("RGB"), dtype=np.int16).copy()
        mask = kwargs.get("mask_image")
        if mask is not None:
            # A compliant model: repaints exactly the requested region.
            region = np.asarray(mask.convert("L").resize(image.size)) >= 96
        else:
            region = np.ones(sample.shape[:2], dtype=bool)
        sample[region] = 255 - sample[region]
        print(f"[STUB] zona cambiada: {region.mean():.2%} de la muestra")
        return SimpleNamespace(image=Image.fromarray(sample.astype(np.uint8))), "OK (stub)"


def _changed(source: Image.Image, result: Image.Image, label: str) -> bool:
    a = np.asarray(source.convert("RGB"), dtype=np.int16)
    b = np.asarray(result.convert("RGB").resize(source.size), dtype=np.int16)
    diff = np.abs(a - b).max(axis=2)
    share = float((diff > 10).mean())
    print(f"[CHECK] {label}: {share:.2%} de píxeles con cambio >10 niveles")
    return share > 0.005


def main():
    source = Image.open(PHOTO).convert("RGB")
    print(f"[CHECK] foto {PHOTO.name} {source.size}")
    manager = ImgEditorManager()
    failures = []

    for prompt, translated in CASES:
        client = ChangedClient()
        with (
            patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
            patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
            patch("roop.img_editor.prompt_translator.translate_prompt", return_value=translated),
            patch(
                "roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
                return_value=client,
            ),
            patch.object(manager, "_release_local_gpu_before_comfy"),
        ):
            output, message, mask = manager.generate_intelligent(
                image=source,
                prompt=prompt,
                engine="longcat",
                use_rewriter=False,
                enhance_faces=False,
                auto_upscale=False,
            )

        generations = sum(name == "generate" for name, _ in client.calls)
        print(f"[CHECK] prompt={prompt!r}")
        print(f"[CHECK]   generaciones llamadas: {generations}")
        print(f"[CHECK]   mensaje: {message}")
        if output is None:
            failures.append(f"{prompt}: sin resultado ({message})")
            continue
        if generations == 0:
            failures.append(
                f"{prompt}: el modelo nunca se llamó y se devolvió la foto ({message})"
            )
            continue
        if not _changed(source, output, prompt):
            failures.append(
                f"{prompt}: hubo generación pero el resultado es la foto original ({message})"
            )

    if failures:
        print("\nFALLO:")
        for item in failures:
            print(f"  - {item}")
        raise SystemExit(1)
    print("\nOK: las instrucciones simple y compuesta cambian la foto")


if __name__ == "__main__":
    main()
