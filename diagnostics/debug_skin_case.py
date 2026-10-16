"""Depura el caso skin de compound_fast_dispatch: ¿dónde pega y dónde no?

Vuelca fuente, salida, body_zone y subject mask del caso «la mujer debe ir
sin zapatos» para localizar por qué el píxel (40,60) queda intacto.
"""
import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

spec = importlib.util.spec_from_file_location(
    "cfd", ROOT / "diagnostics" / "probe_editor_compound_fast_dispatch.py")
cfd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfd)

import re  # noqa: E402

from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402

OUT = ROOT / "diagnostics" / "out" / "halo_stages_20261013" / "skin_debug"
OUT.mkdir(parents=True, exist_ok=True)


def _install_spies():
    real_vrq = ImgEditorManager._visual_request_queries
    real_rros = ImgEditorManager._request_region_on_subject

    def vrq_spy(self, texts):
        queries = real_vrq(self, texts)
        print(f"[VRQ] texts={[str(t)[:40] for t in texts]} -> {queries}", flush=True)
        return queries

    def rros_spy(self, image, subject_mask, texts):
        result = real_rros(self, image, subject_mask, texts)
        print(f"[RROS] texts={[str(t)[:40] for t in texts]} -> "
              f"{'None' if result is None else 'region'}", flush=True)
        return result

    ImgEditorManager._visual_request_queries = vrq_spy
    ImgEditorManager._request_region_on_subject = rros_spy


def main():
    _install_spies()
    skin_arr = np.full((160, 220, 3), (30, 40, 70), np.uint8)
    skin_arr[20:150, 40:120] = (190, 140, 110)
    skin_arr[110:145, 48:92] = (22, 22, 24)
    skin_photo = Image.fromarray(skin_arr, "RGB")
    class _Face:
        def __init__(self, bbox):
            self.bbox = bbox

    skin_face = _Face((55, 28, 100, 72))

    class _SkinSpeckleMasker:
        def generate_mask(self, image, prompt, threshold=0.4):
            arr = np.zeros((image.height, image.width), np.uint8)
            text = (prompt or "").lower()
            if re.search(r"\b(?:woman|mujer)\b", text):
                arr[20:150, 40:120] = 255
            return Image.fromarray(arr, "L")

    manager = ImgEditorManager()
    skin_client = cfd.FakeClient()

    captured = {}
    real_zone = manager._body_paste_zone

    def spy_zone(*args, **kwargs):
        zone = real_zone(*args, **kwargs)
        if zone is not None:
            za = np.asarray(zone.convert("L"))
            captured["body_zone"] = za.copy()
            print(f"[DEBUG] body_zone: {(za >= 160).sum()} px; "
                  f"en (40,60)={za[40, 60]}; (130,60)={za[130, 60]}")
        else:
            print("[DEBUG] body_zone = None")
        return zone

    with (patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
          patch("roop.face_util.get_face_analyser", return_value=object()),
          patch("roop.face_util.detect_faces_robust", return_value=[skin_face]),
          patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
          patch.object(manager, "_release_local_gpu_before_comfy"),
          patch.object(manager, "_get_clipseg_masker", return_value=_SkinSpeckleMasker()),
          patch.object(manager, "_body_paste_zone", side_effect=spy_zone),
          patch("roop.img_editor.prompt_translator.translate_prompt",
                return_value="the woman must go without shoes"),
          patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
                return_value=skin_client)):
        output, message, _ = manager.generate_intelligent(
            image=skin_photo, prompt="la mujer debe ir sin zapatos",
            engine="longcat",
        )

    print("message:", message)
    src = np.asarray(skin_photo.convert("RGB"))
    out = np.asarray(output.convert("RGB"))
    diff = np.abs(out.astype(np.int16) - src.astype(np.int16)).max(axis=2)
    changed = diff > 2
    ys, xs = np.nonzero(changed)
    print(f"cambiados: {changed.sum()} px; bbox="
          f"{(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if changed.any() else None}")
    print(f"(40,60) cambiado={bool(changed[40, 60])}; (130,60) cambiado={bool(changed[130, 60])}; "
          f"(5,5) cambiado={bool(changed[5, 5])}")
    Image.fromarray(src).save(OUT / "source.png")
    Image.fromarray(out).save(OUT / "output.png")
    Image.fromarray((changed * 255).astype(np.uint8)).save(OUT / "changed.png")
    if "body_zone" in captured:
        Image.fromarray(captured["body_zone"]).save(OUT / "body_zone.png")
    print("generaciones:", [(n, a.get('keep_source_pixels'), a.get('mask_image') is not None,
                            a.get('no_kontext')) for n, a in skin_client.calls if n == "generate"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
