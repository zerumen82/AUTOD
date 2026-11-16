#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe v5.1351 — estado «descalza»: zona de los pies y verificación por piel.

Sin GPU ni ComfyUI: el segmentador y el motor se sustituyen por stubs medidos
sobre una imagen sintética con persona, torso y calzado.
"""
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw  # noqa: E402

from roop.img_editor import selected_region_edit as sre  # noqa: E402
from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402

W = H = 400
BLOCK_PERSON = (80, 40, 320, 380)
BLOCK_TORSO = (90, 100, 310, 300)
BLOCK_SHOES = (140, 330, 260, 380)
BLOCK_STOCKINGS = (140, 250, 260, 330)


def _rect(shape, box):
    mask = np.zeros(shape, dtype=bool)
    mask[box[1]:box[3], box[0]:box[2]] = True
    return mask


def build_image(shoes_color=(40, 30, 25)):
    img = Image.new("RGB", (W, H), (128, 128, 128))
    draw = ImageDraw.Draw(img)
    draw.rectangle(BLOCK_PERSON, fill=(200, 150, 130))
    draw.rectangle(BLOCK_TORSO, fill=(240, 240, 240))
    draw.rectangle(BLOCK_SHOES, fill=shoes_color)
    return img


class FakeMasker:
    def generate_mask(self, image, query):
        q = (query or "").lower()
        mask = Image.new("L", (W, H), 0)
        draw = ImageDraw.Draw(mask)
        if any(word in q for word in ("woman", "mujer", "subject")):
            draw.rectangle(BLOCK_PERSON, fill=255)
        elif any(word in q for word in ("barefoot", "descalza")):
            draw.rectangle(BLOCK_SHOES, fill=255)
        elif any(word in q for word in ("stocking", "medias", "tights")):
            draw.rectangle(BLOCK_STOCKINGS, fill=255)
        else:
            return None
        return mask


class FakeClient:
    def __init__(self, result_image=None, solid_color=None):
        self.result_image = result_image
        self.solid_color = solid_color
        self.calls = []

    def load(self, **kwargs):
        self.calls.append(("load", kwargs))
        return True, "OK"

    def generate(self, **kwargs):
        self.calls.append(("generate", kwargs))
        if self.solid_color is not None:
            crop = kwargs.get("image")
            size = crop.size if crop is not None else (W, H)
            return SimpleNamespace(
                image=Image.new("RGB", size, self.solid_color)), "OK"
        return SimpleNamespace(image=self.result_image), "OK"


def noop(*args, **kwargs):
    return None


def run_described(manager, captured, raw_prompt="la mujer debe ir descalza",
                  translated="The woman must go barefoot"):
    image = build_image()
    client = FakeClient()
    clothing = _rect((H, W), BLOCK_TORSO)

    def fake_pass(mgr, cli, img, zone, support, prompt, *args, **kwargs):
        captured["zone"] = np.asarray(zone, dtype=bool).copy()
        captured["kwargs"] = kwargs
        return img.copy(), "ok"

    with patch.object(manager, "_get_clipseg_masker", return_value=FakeMasker()), \
            patch.object(manager, "_release_local_gpu_before_comfy", noop), \
            patch.object(manager, "_request_region_on_subject", return_value=None), \
            patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True), \
            patch("roop.img_editor.selected_region_edit.allow_model",
                  return_value=(True, "")), \
            patch("roop.img_editor.selected_region_edit.permit_automatic_model", noop), \
            patch("roop.img_editor.selected_region_edit._measured_clothing_zone",
                  return_value=clothing), \
            patch("roop.img_editor.selected_region_edit._vision_zone_pass",
                  fake_pass), \
            patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
                  return_value=client):
        result, message, mask = sre.edit_described_subject(
            manager, image, raw_prompt, translated,
            [SimpleNamespace(bbox=np.array([170, 50, 230, 110], dtype=float))],
            engine="longcat", automatic=True,
        )
    return captured, result, message


def check_state_skin_verification(manager, shoes_result_color):
    image = build_image()
    client = FakeClient(solid_color=shoes_result_color)
    zone = _rect((H, W), BLOCK_SHOES)
    support = _rect((H, W), BLOCK_PERSON)
    settings = {
        "crop_context": 0.25, "crop_max_area": 1.0,
        "minimum_reference_side": 64, "maximum_reference_side": 768,
    }
    with patch.object(manager, "_release_local_gpu_before_comfy", noop), \
            patch.object(manager, "_covered_surface_view", return_value=None), \
            patch("roop.img_editor.selected_region_edit._harmonize_patch",
                  side_effect=lambda patch_img, *a, **k: patch_img), \
            patch("cv2.seamlessClone",
                  side_effect=cv2.error("skip")):
        composed, note = sre._vision_zone_pass(
            manager, client, image, zone, support, "barefoot skin",
            "", 424242, settings, noop, lambda: False,
            recipe={"steps": 4, "guidance": 3.5, "denoise": 1.0,
                    "no_kontext": True},
            label="probe", check_state_skin=True,
        )
    return composed, note


def main():
    manager = ImgEditorManager()
    ok = []

    def check(name, condition, detail=""):
        ok.append(bool(condition))
        print(f"{'PASS' if condition else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))

    captured, result, message = run_described(manager, {})
    zone = captured.get("zone")
    check("descalza: la ruta llega al motor (no conserva la foto)", zone is not None,
          message)
    if zone is not None:
        shoes = _rect((H, W), BLOCK_SHOES)
        torso = _rect((H, W), BLOCK_TORSO)
        shoes_share = float((zone & shoes).sum()) / max(1, int(shoes.sum()))
        torso_share = float((zone & torso).sum()) / max(1, int(torso.sum()))
        check(
            "descalza: la zona es el calzado medido («go barefoot»), no todo el torso",
            shoes_share >= 0.60 and torso_share <= 0.10,
            f"calzado={shoes_share:.0%} torso={torso_share:.0%}",
        )
        check(
            "descalza: la verificación por piel queda activa",
            captured.get("kwargs", {}).get("check_state_skin") is True,
        )

    captured, result, message = run_described(
        manager, {},
        raw_prompt="la mujer debe ir descalza y sin medias",
        translated="the woman must go barefoot and without stockings",
    )
    zone = captured.get("zone")
    if zone is not None:
        stockings = _rect((H, W), BLOCK_STOCKINGS)
        shoes = _rect((H, W), BLOCK_SHOES)
        stockings_share = float((zone & stockings).sum()) / max(1, int(stockings.sum()))
        shoes_share = float((zone & shoes).sum()) / max(1, int(shoes.sum()))
        check(
            "sin medias: la zona es la prenda medida",
            stockings_share >= 0.60,
            f"medias={stockings_share:.0%}",
        )
        check(
            "sin medias: la zona une objetivo y estado (medias + pies)",
            shoes_share >= 0.60,
            f"calzado={shoes_share:.0%}",
        )
    check(
        "sin medias: una prenda puesta se verifica por piel (no por otra prenda)",
        captured.get("kwargs", {}).get("check_state_skin") is True,
        message[:90],
    )

    captured, result, message = run_described(
        manager, {},
        raw_prompt="la mujer debe ir descalza y con el pelo suelto",
        translated="the woman must go barefoot and with her hair loose",
    )
    zone = captured.get("zone")
    if zone is not None:
        shoes = _rect((H, W), BLOCK_SHOES)
        shoes_share = float((zone & shoes).sum()) / max(1, int(shoes.sum()))
        check(
            "estado compuesto: la cláusula medible da la zona (pies, no frase entera)",
            shoes_share >= 0.60,
            f"calzado={shoes_share:.0%}",
        )

    composed, note = check_state_skin_verification(manager, (20, 20, 120))
    check(
        "estado sin piel tras el render: no se declara cumplido",
        composed is not None and "no muestra piel" in (note or "").lower(),
        (note or "")[:110],
    )

    composed, note = check_state_skin_verification(manager, (205, 155, 135))
    check(
        "estado con piel descubierta tras el render: pasa",
        composed is not None and "no muestra piel" not in (note or ""),
        (note or "")[:110],
    )

    passed = sum(ok)
    print(f"\n{passed}/{len(ok)} comprobaciones")
    return 0 if passed == len(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
