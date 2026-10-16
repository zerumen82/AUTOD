# -*- coding: utf-8 -*-
"""Depuración: ¿de dónde sale la máscara de 1.608 px en el pie?

Caso: fast_1792015141.png + «Man must be bald». El probe
probe_editor_bald_backhair_20261016 midió que el primer pase envía hoy una
máscara de 1.608 px en foto (159-203, 388-434) — sobre los DEDOS del pie —
mientras que la elipse de la cabeza del 15/10 está en (593-766, 242-528).
Este script cuelga spies de _head_mask_from_faces, _appearance_head_pass,
_narrow_appearance_passes y _head_zone_for_subject e imprime bbox/px de cada
máscara intermedia + las caras detectadas (insightface CPU real).

CPU sin GPU. Sale 0 salvo error real.
"""
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

SRC = os.path.join("ui", "tob", "ComfyUI", "input", "fast_1792015141.png")


def stats(label, mask):
    if mask is None:
        print(f"    {label}: None")
        return
    arr = np.asarray(mask.convert("L")) >= 128
    ys, xs = np.where(arr)
    if not len(xs):
        print(f"    {label}: VACÍA")
        return
    print(
        f"    {label}: {int(arr.sum())} px, bbox "
        f"({int(xs.min())}, {int(ys.min())}, {int(xs.max())}, {int(ys.max())})"
    )


def main() -> int:
    from roop.face_util import get_face_analyser_cpu
    from roop.img_editor.img_editor_manager import ImgEditorManager

    manager = ImgEditorManager()

    class CaptureClient:
        def __init__(self):
            self.calls = []

        def load(self, version):
            return True, "OK (stub)"

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(image=Image.new("RGB", (32, 20), "red")), "OK (stub)"

    client = CaptureClient()

    orig_head_mask = ImgEditorManager._head_mask_from_faces

    def spy_head_mask(self, image, faces, *args, **kwargs):
        print("  [SPY] _head_mask_from_faces:")
        for index, face in enumerate(faces or []):
            print(f"    cara {index}: bbox={tuple(round(float(v), 1) for v in face.bbox)}")
        result = orig_head_mask(self, image, faces, *args, **kwargs)
        stats("head_mask →", result)
        return result

    orig_head_pass = ImgEditorManager._appearance_head_pass

    def spy_head_pass(self, *args, **kwargs):
        result = orig_head_pass(self, *args, **kwargs)
        print("  [SPY] _appearance_head_pass →", "None" if result is None else "pasada")
        if result is not None:
            stats("    mask", result.get("mask"))
            stats("    person", result.get("person"))
        return result

    orig_narrow = ImgEditorManager._narrow_appearance_passes

    def spy_narrow(self, image, passes, *args, **kwargs):
        print(f"  [SPY] _narrow_appearance_passes entrada ({len(passes)}):")
        for index, item in enumerate(passes):
            stats(f"    [{index}] mask", item.get("mask"))
            stats(f"    [{index}] person", item.get("person"))
        result = orig_narrow(self, image, passes, *args, **kwargs)
        print(f"  [SPY] _narrow_appearance_passes salida ({len(result)}):")
        for index, item in enumerate(result):
            stats(f"    [{index}] mask", item.get("mask"))
            stats(f"    [{index}] person", item.get("person"))
            print(
                f"    [{index}] measured_request={item.get('measured_request')} "
                f"deliver_region={item.get('deliver_region')}"
            )
        return result

    orig_head_zone = ImgEditorManager._head_zone_for_subject

    def spy_head_zone(self, image, subject_mask, faces, announce=True):
        print("  [SPY] _head_zone_for_subject entrada:")
        stats("    subject_mask", subject_mask)
        result = orig_head_zone(self, image, subject_mask, faces, announce=announce)
        stats("    head_zone →", result)
        return result

    orig_complete = ImgEditorManager._complete_request_surface

    def spy_complete(self, image, seed_mask, subject_mask, faces, *args, **kwargs):
        print("  [SPY] _complete_request_surface:")
        stats("    seed", seed_mask)
        stats("    subject", subject_mask)
        result = orig_complete(self, image, seed_mask, subject_mask, faces, *args, **kwargs)
        stats("    completed →", result)
        return result

    with (
        patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
        patch("roop.face_util.get_face_analyser", new=lambda *a, **k: get_face_analyser_cpu()),
        patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
        patch.object(manager, "_release_local_gpu_before_comfy"),
        patch(
            "roop.img_editor.prompt_translator.translate_prompt",
            new=lambda text, *a, **k: text,
        ),
        patch(
            "roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
            return_value=client,
        ),
        patch.object(ImgEditorManager, "_head_mask_from_faces", spy_head_mask),
        patch.object(ImgEditorManager, "_appearance_head_pass", spy_head_pass),
        patch.object(ImgEditorManager, "_narrow_appearance_passes", spy_narrow),
        patch.object(ImgEditorManager, "_head_zone_for_subject", spy_head_zone),
        patch.object(ImgEditorManager, "_complete_request_surface", spy_complete),
    ):
        output, message, _ = manager.generate_intelligent(
            image=Image.open(SRC).convert("RGB"),
            prompt="Man must be bald.",
            engine="longcat",
        )
    print(f"output={'OK' if output is not None else 'None'} | {message}")
    for index, kw in enumerate(client.calls):
        zone = kw.get("mask_image")
        work = kw.get("image")
        print(f"  generate[{index}]: imagen={None if work is None else work.size}")
        stats("    mask_image", zone)
    return 0


if __name__ == "__main__":
    sys.exit(main())
