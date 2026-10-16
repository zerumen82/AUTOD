# -*- coding: utf-8 -*-
"""Repro y contrato del ancla de edición sobre la foto «calvo» del 15/10 (H-EDIT-003).

Caso real: ``ui/tob/ComfyUI/input/fast_1792015141.png`` + «Man must be bald»
→ ``FastEdit_00566_.png`` / ``img_editor/edit_1792015316.png``. Diagnóstico
visual previo en ``diagnostics/out/edit_1792015316_review/``.

Lo que se midió (2026-10-16, repro en
``diagnostics/debug_bald_zone_20261016.py`` con CLIPSeg e insightface CPU):

* El detector devuelve el hombre (det 0.69, 100×137 px) y un falso positivo
  de 27×22 px en sus DEDOS (det 0.30). En el flujo anterior ese pie pasaba
  al ancla (``source_faces``), el calor «Man» lo prefería a la cara real
  (0.225 vs 0.097) y toda la edición se anclaba a los dedos: la máscara
  enviada medía 1.608 px en foto (159-203, 388-434) — sobre el pie — y la
  cabeza ni se tocaba.
* Con la guarda ``_credible_faces`` (score ≥ 0.5×mejor; caras reales
  medidas ≥ 0.57×, el pie 0.43×) el ancla vuelve a ser la cabeza del
  hombre: máscara de 36.345 px en (595, 241, 761, 529), casi la misma
  elipse que envió la UI del 15/10 ((593, 242, 766, 528), 37.426 px).

Contrato de este probe (CPU, sin GPU):

1. La máscara del primer pase, mapeada a coordenadas de foto, cubre la
   cabeza de referencia (≥70% de la elipse del 15/10).
2. No toca la zona del pie (≤200 px sobre la banda (159-203, 388-434)).

Informa (sin fallar) la banda de pelo trasero (x≥768): el agujero sigue
siendo la elipse de cabeza, igual que el 15/10; extender el pelo trasero es
el hueco abierto de H-EDIT-003 (decisión ``_complete_hair_surface`` con
validación visual del usuario pendiente).
"""
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import cv2
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

SRC = os.path.join("ui", "tob", "ComfyUI", "input", "fast_1792015141.png")
REF = os.path.join("ui", "tob", "ComfyUI", "input", "mask_1792015141.png")
OUT_DIR = os.path.join("diagnostics", "out", "bald_backhair_20261016")
PROMPT = "Man must be bald."


class CaptureClient:
    """Cliente falso: guarda cada kwargs de generate (incluida la máscara)."""

    def __init__(self):
        self.calls = []

    def load(self, version):
        self.calls.append(("load", version))
        return True, "OK (stub)"

    def generate(self, **kwargs):
        self.calls.append(("generate", kwargs))
        return SimpleNamespace(image=Image.new("RGB", (32, 20), "red")), "OK (stub)"


def mask_stats(mask) -> dict:
    arr = np.asarray(mask.convert("L")) >= 128
    ys, xs = np.where(arr)
    if not len(xs):
        return {"px": 0, "arr": arr}
    return {
        "px": int(arr.sum()),
        "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
        "arr": arr,
    }


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    photo = Image.open(SRC).convert("RGB")
    ref = mask_stats(Image.open(REF).convert("L"))
    print(f"foto: {SRC} {photo.size}")
    print(
        "máscara del 15/10: "
        f"{ref['px']} px, bbox {ref.get('bbox')} (solo elipse de cabeza)"
    )

    from roop.face_util import get_face_analyser_cpu
    from roop.img_editor.img_editor_manager import ImgEditorManager

    manager = ImgEditorManager()
    client = CaptureClient()
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
    ):
        output, message, out_mask = manager.generate_intelligent(
            image=photo, prompt=PROMPT, engine="longcat",
        )
    print(f"generate_intelligent → output={'OK' if output is not None else 'None'}")
    print(f"mensaje: {message}")

    gen_calls = [kw for name, kw in client.calls if name == "generate"]
    print(f"pases generate: {len(gen_calls)}")
    first = None
    first_image = None
    for index, kw in enumerate(gen_calls):
        work = kw.get("image")
        zone = kw.get("mask_image")
        stats = None if zone is None else mask_stats(zone)
        extra = ""
        if stats is not None and zone is not None:
            soft = np.asarray(zone.convert("L"))
            extra = (
                f" máscara={stats['px']} px>=128, {int((soft > 0).sum())} px>0, "
                f"rango {int(soft.min())}-{int(soft.max())}, "
                f"bbox>=128 {stats.get('bbox')}"
            )
        elif zone is None:
            extra = " máscara=SIN (pase global)"
        print(
            f"  [{index}] imagen={None if work is None else work.size} "
            f"steps={kw.get('num_inference_steps')} "
            f"denoise={kw.get('denoise')} no_kontext={kw.get('no_kontext')} "
            f"keep_source={kw.get('keep_source_pixels')}{extra}"
        )
        if first is None and stats is not None:
            first = stats
            first_image = work

    if first is None:
        print("VEREDICTO: el manager no envió máscara en ningún pase")
        return 1

    # Mapa ventana → foto: la ventana puede ser un recorte o la foto a otra
    # escala; se localiza por igualdad de píxeles sobre un parche central.
    photo_arr = np.asarray(photo)
    work_arr = np.asarray(first_image.convert("RGB"))
    offset = None
    scaled = False
    if work_arr.shape[0] <= photo_arr.shape[0] and work_arr.shape[1] <= photo_arr.shape[1]:
        probe_patch = work_arr[work_arr.shape[0] // 2:work_arr.shape[0] // 2 + 24,
                         work_arr.shape[1] // 2:work_arr.shape[1] // 2 + 24]
        match = cv2.matchTemplate(photo_arr, probe_patch, cv2.TM_SQDIFF)
        _, _, loc, _ = cv2.minMaxLoc(match)
        probe_y = work_arr.shape[0] // 2
        probe_x = work_arr.shape[1] // 2
        ox, oy = int(loc[0]) - probe_x, int(loc[1]) - probe_y
        if 0 <= oy and oy + work_arr.shape[0] <= photo_arr.shape[0] and \
           0 <= ox and ox + work_arr.shape[1] <= photo_arr.shape[1]:
            window = photo_arr[oy:oy + work_arr.shape[0], ox:ox + work_arr.shape[1]]
            if np.array_equal(window, work_arr):
                offset = (ox, oy)
    if offset is None:
        # ¿La ventana es la foto a otra escala?
        resized = np.asarray(
            photo.resize((work_arr.shape[1], work_arr.shape[0]), Image.Resampling.LANCZOS)
        )
        diff = np.abs(resized.astype(np.int16) - work_arr.astype(np.int16)).mean()
        scaled = diff < 6.0
        print(f"ventana no es recorte píxel-a-píxel; prueba de escala: diff media {diff:.2f}")
    print(f"ventana: {first_image.size} → offset {offset}, escalada={scaled}")

    today_win = first["arr"]
    ref_arr = ref["arr"]
    if offset is not None:
        today = np.zeros_like(ref_arr)
        ox, oy = offset
        h, w = today_win.shape
        today[oy:oy + h, ox:ox + w] = today_win
        coord_note = f"recorte offset ({ox}, {oy})"
    elif scaled:
        today = np.asarray(
            Image.fromarray((today_win * 255).astype(np.uint8)).resize(
                photo.size, Image.Resampling.NEAREST
            )
        ) > 127
        coord_note = "foto escalada a tamaño completo (NEAREST)"
    else:
        print("VEREDICTO: no se pudo mapear la ventana a la foto")
        return 1
    print(f"mapeo: {coord_note}")

    # Zona del falso positivo del 15/10: la banda medida sobre la foto donde
    # el detector devolvió una "cara" de 27×22 px en los dedos del hombre.
    foot = np.zeros_like(today)
    foot[388:434, 159:203] = True
    foot_px = int((today & foot).sum())
    overlap = int((today & ref_arr).sum())
    coverage = overlap / max(1, int(ref_arr.sum()))
    print(
        f"cubre {coverage:.1%} de la elipse de cabeza del 15/10 "
        f"({overlap}/{int(ref_arr.sum())} px); máscara total {int(today.sum())} px"
    )
    print(f"zona del pie (159-203, 388-434): {foot_px} px tocados hoy")

    # Banda de pelo trasero: se informa sin fallar. El agujero sigue siendo
    # la elipse de cabeza (brecha abierta de H-EDIT-003: la decisión sobre
    # _complete_hair_surface espera validación visual del usuario).
    band = np.zeros_like(today)
    band[240:560, 768:] = True
    band_back_today = int((today & band).sum())
    band_back_ref = int((ref_arr & band).sum())
    print(
        f"pelo trasero (x>=768, y 240-560): ref 15/10 {band_back_ref} px → "
        f"hoy {band_back_today} px (brecha abierta H-EDIT-003)"
    )

    base = np.array(photo)
    base[ref_arr] = (base[ref_arr] * 0.55 + np.array([255, 0, 0]) * 0.45).astype(np.uint8)
    base[today] = (base[today] * 0.55 + np.array([0, 255, 0]) * 0.45).astype(np.uint8)
    foot_vis = base.copy()
    foot_vis[foot] = np.array([0, 255, 255])
    Image.fromarray(np.hstack([base, foot_vis])).save(
        os.path.join(OUT_DIR, "mask_today_vs_1510.png")
    )
    print(
        f"hoja: {os.path.join(OUT_DIR, 'mask_today_vs_1510.png')} "
        "(rojo=elipse 15/10, verde=máscara de hoy, amarillo=banda del pie)"
    )

    failed = False
    if coverage < 0.70:
        print(f"FAIL: la máscara cubre solo {coverage:.1%} de la cabeza de referencia")
        failed = True
    if foot_px > 200:
        print(f"FAIL: la máscara toca la zona del pie ({foot_px} px): el ancla volvió al falso positivo")
        failed = True
    if not failed:
        print("OK: la máscara de hoy se ancla en la cabeza del hombre y no en el pie")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
