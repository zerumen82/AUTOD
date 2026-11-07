"""CPU check of a body change with no manual selection (no render).

v5.1244: the prompt names one person. The route writes that change on the
measured person and keeps the rest of the photo. Redrawing the frame
(edit_1792709573.png, identity 0/2) is the failure this probe guards.

This captures, without a diffusion model:
  * the final prompt (the state sentence and the enforcement);
  * one img2img pass on the measured clothing zone, with no latent anchor
    (`kontext_img2img_denoise`, so the client builds no ReferenceLatent);
  * pixels outside that zone stay the source photo.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402

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
        return SimpleNamespace(image=Image.new("RGB", photo.size, (12, 240, 35))), "CPU generation"


class ClothingSegmenter:
    """Stub of the garment measurement: returns the given box."""

    def __init__(self, box):
        self.box = box

    def is_available(self):
        return True

    def segment_clothing(self, image, threshold=0.45):
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).rectangle(self.box, fill=255)
        return mask, np.asarray(mask)


class PersonMasker:
    def __init__(self, box, heat_box=None):
        self.box = box
        self.heat_box = heat_box

    def generate_mask(self, image, prompt, threshold=0.4):
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).rectangle(self.box, fill=255)
        return mask

    def probability(self, image, prompt):
        heat = np.zeros((image.height, image.width), dtype=np.float32)
        box = self.heat_box or self.box
        x1, y1, x2, y2 = box
        heat[y1:y2, x1:x2] = 0.85
        return heat


def scene():
    photo = Image.new("RGB", (640, 420), (20, 30, 40))
    ImageDraw.Draw(photo).rectangle((280, 30, 460, 400), fill=(180, 140, 120))
    ImageDraw.Draw(photo).rectangle((40, 40, 160, 400), fill=(90, 90, 140))
    woman = SimpleNamespace(bbox=(300, 40, 430, 150), age=30, det_score=0.9)
    return photo, woman


def main():
    photo, woman = scene()
    manager = ImgEditorManager()
    client = RecordingClient()
    with (patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
          patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
          patch("roop.face_util.get_face_analyser", return_value=object()),
          patch("roop.face_util.detect_faces_robust", return_value=[woman]),
          patch.object(manager, "_release_local_gpu_before_comfy"),
          patch.object(manager, "_get_clipseg_masker",
                       return_value=PersonMasker((280, 30, 460, 400))),
          # La zona se mide con el segmentador de ropa. Sin este doble, un
          # fixture de rectángulos no tiene ropa que CLIPSeg reconozca y la
          # ruta se queda sin zona (falso FAIL del contrato, no del producto).
          patch.object(manager, "_get_clothing_segmenter",
                       return_value=ClothingSegmenter((290, 40, 450, 400))),
          patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
                return_value=client)):
        output, message, frame_mask = manager.generate_intelligent(
            image=photo, prompt=PROMPT, engine="kontext",
            use_semantic=True, use_rewriter=False, enhance_faces=False,
            auto_upscale=False, seed=424242,
        )
    print(f"[Prompt] loads={client.loads}", flush=True)
    print(f"[Prompt] generations={len(client.generations)}", flush=True)
    if not client.generations:
        print(f"[Prompt] the route did not reach the engine: {message}", flush=True)
        return 1
    request = client.generations[0]
    prompt = str(request.get("prompt", ""))
    print(f"[Prompt] final prompt={prompt}", flush=True)
    hole = request.get("mask_image")
    # v5.1252 — la zona que se escribe se devuelve en coordenadas de la foto:
    # el motor con visión recibe un RECORTE de la persona (a 768), así que la
    # máscara del request ya no es la de la foto.
    zone = (np.asarray(frame_mask.convert("L")) > 0
            if hasattr(frame_mask, "convert") else None)
    source = np.asarray(photo)
    result = np.asarray(output) if output is not None else None
    request_image = request.get("image")
    cropped = request_image is not None and max(request_image.size) > max(photo.size)
    outside_kept = (
        zone is not None and result is not None
        and np.array_equal(result[~zone], source[~zone])
    )
    inside_changed = (
        zone is not None and result is not None and zone.any()
        and np.any(result[zone] != source[zone])
    )
    checks = {
        "una sola generación": len(client.generations) == 1,
        # v5.1252 — medido 2026-10-26 (9 renders, foto del usuario): Kontext
        # responde a la zona de ropa medida con OTRA prenda (bañador) con
        # denoise 0,80-1,00, guidance 3,5/4,6, estado afirmativo y a doble
        # resolución. El motor con visión es el único que quita la prenda y
        # conserva la pose, así que esta ruta lo pide de forma explícita.
        "se pide el motor con visión": any(
            kwargs.get("allow_vision_engine") for _args, kwargs in client.loads),
        "sin ancla de Kontext (no_kontext, denoise 1.00)": (
            bool(request.get("no_kontext"))
            and float(request.get("denoise", 0) or 0) >= 0.99),
        "el motor recibe el recorte de la persona y no el marco": cropped,
        "la frase de estado final está escrita": "is fully nude" in prompt,
        "el enforcement está escrito": "must be clearly visible" in prompt,
        "la instrucción está": "naked" in prompt,
        "hay agujero de la persona medida": hole is not None and bool(request.get("keep_source_pixels")),
        "fuera de la zona la foto no cambia": outside_kept,
        "dentro de la zona el cambio se escribe": inside_changed,
    }
    failures = [name for name, ok in checks.items() if not ok]
    for name, ok in checks.items():
        print(f"[Prompt] {'PASS' if ok else 'FAIL'} {name}", flush=True)
    print(f"[Prompt] salida {output.size if output is not None else None}, "
          f"mensaje: {message}", flush=True)
    print(f"[Prompt] {len(checks) - len(failures)}/{len(checks)} PASS; failures={failures}",
          flush=True)
    return bool(failures) or two_people() or barefoot_without_pants()


def two_people():
    """A sentence that names the left person edits her without a click."""
    photo = Image.new("RGB", (640, 420), (20, 30, 40))
    ImageDraw.Draw(photo).rectangle((40, 30, 250, 400), fill=(180, 140, 120))
    ImageDraw.Draw(photo).rectangle((390, 30, 600, 400), fill=(90, 90, 140))
    left = SimpleNamespace(bbox=(80, 40, 180, 150), age=30, det_score=0.9)
    right = SimpleNamespace(bbox=(430, 40, 540, 150), age=28, det_score=0.88)
    manager = ImgEditorManager()
    client = RecordingClient()
    wide = PersonMasker((40, 30, 600, 400), heat_box=(40, 30, 250, 400))
    with (patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
          patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
          patch("roop.face_util.get_face_analyser", return_value=object()),
          patch("roop.face_util.detect_faces_robust", return_value=[left, right]),
          patch.object(manager, "_release_local_gpu_before_comfy"),
          patch.object(manager, "_get_clipseg_masker", return_value=wide),
          patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
                return_value=client)):
        output, message, frame_mask = manager.generate_intelligent(
            image=photo,
            prompt="la mujer de la izquierda debe ir desnuda",
            engine="kontext", use_semantic=True, use_rewriter=False,
            enhance_faces=False, auto_upscale=False, seed=424242,
        )
    if len(client.generations) != 1 or output is None:
        print(f"[Prompt] FAIL dos personas: no hubo una pasada ({message})", flush=True)
        return 1
    zone = np.asarray(frame_mask.convert("L")) > 0
    source, result = np.asarray(photo), np.asarray(output)
    right_face = np.zeros(zone.shape, dtype=bool)
    right_face[40:150, 430:540] = True
    left_body = np.zeros(zone.shape, dtype=bool)
    left_body[160:400, 40:250] = True
    ok = (
        np.array_equal(result[right_face], source[right_face])
        and left_body.any() and np.any(result[left_body] != source[left_body])
        and message.lower().startswith("sin selección")
    )
    print(f"[Prompt] {'PASS' if ok else 'FAIL'} el prompt de la izquierda no exige marcar",
          flush=True)
    if not ok:
        print(f"[Prompt] mensaje={message}", flush=True)
    return 0 if ok else 1


def barefoot_without_pants():
    """«descalza, sin pantalón, sin blusa»: ropa quitada por partes.

    Medido 2026-10-23 en el log del usuario: la frase entera mide
    body/uncover, pero sus cláusulas («without pants», «without blouse»)
    no miden body y vetaban la pasada medida; el router global encontraba
    la guarda y devolvía la foto idéntica (edit_1792748448.png). Aquí se
    exige que se edite la persona nombrada sin marcar nada.

    v5.1248 — también la forma coordinada con «y» (log 12:06: «la mujer de
    la izquierda debe ir descalza y sin pantalones»), que no se partía en
    cláusulas y dejaba la zona en el fragmento de ropa (19 850 px).
    """
    failures = 0
    for prompt in ("la mujer debe ir descalza, sin pantalon, sin blusa",
                   "la mujer debe ir descalza y sin pantalones"):
        failures += _barefoot_case(prompt)
    return 1 if failures else 0


def _barefoot_case(prompt):
    photo = Image.new("RGB", (640, 420), (20, 30, 40))
    ImageDraw.Draw(photo).rectangle((40, 30, 250, 400), fill=(180, 140, 120))
    ImageDraw.Draw(photo).rectangle((390, 30, 600, 400), fill=(90, 90, 140))
    left = SimpleNamespace(bbox=(80, 40, 180, 150), age=30, det_score=0.9)
    right = SimpleNamespace(bbox=(430, 40, 540, 150), age=28, det_score=0.88)
    manager = ImgEditorManager()
    client = RecordingClient()
    wide = PersonMasker((40, 30, 600, 400), heat_box=(40, 30, 250, 400))
    with (patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
          patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
          patch("roop.face_util.get_face_analyser", return_value=object()),
          patch("roop.face_util.detect_faces_robust", return_value=[left, right]),
          patch.object(manager, "_release_local_gpu_before_comfy"),
          patch.object(manager, "_get_clipseg_masker", return_value=wide),
          # Solo el chaleco, como el CLIPSeg real en la foto del usuario
          # (21568 px medidos): el pantalón queda fuera del fragmento.
          patch.object(manager, "_get_clothing_segmenter",
                       return_value=ClothingSegmenter((45, 150, 245, 215))),
          patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
                return_value=client)):
        output, message, frame_mask = manager.generate_intelligent(
            image=photo,
            prompt=prompt,
            engine="kontext", use_semantic=True, use_rewriter=False,
            enhance_faces=False, auto_upscale=False, seed=424242,
        )
    if len(client.generations) != 1 or output is None:
        print(f"[Prompt] FAIL sin pantalon: no hubo una pasada ({message})", flush=True)
        return 1
    request = client.generations[0]
    request_prompt = str(request.get("prompt", ""))
    zone = np.asarray(frame_mask.convert("L")) > 0 if frame_mask is not None else None
    source, result = np.asarray(photo), np.asarray(output)
    if zone is None or not zone.any():
        print("[Prompt] FAIL sin pantalon: no hay zona medida", flush=True)
        return 1
    # Por debajo del chaleco medido (que acaba en y=215): solo entra si la
    # zona se ensanchó a la persona.
    left_pants = np.zeros(zone.shape, dtype=bool)
    left_pants[260:400, 60:240] = True
    right_face = np.zeros(zone.shape, dtype=bool)
    right_face[40:150, 430:540] = True
    # Las dos formas piden pies descalzos y pantalón; solo la de comas pide
    # también la blusa.
    asked = ["barefoot", "pants"] + (
        ["blouse"] if "blusa" in prompt.lower() else [])
    checks = {
        "la instruccion completa llega al motor": bool(asked) and all(
            word in request_prompt.lower() for word in asked),
        "una sola generacion": True,
        "fuera de la zona la foto no cambia": np.array_equal(result[~zone], source[~zone]),
        "la zona cubre el pantalon de la mujer": bool((left_pants & zone).any()),
        "el pantalon cambia": bool(np.any(result[left_pants] != source[left_pants])),
        "la zona es la persona, no el fragmento de ropa": (
            "persona medida menos la cabeza" in message.lower()),
        "la otra persona no cambia": np.array_equal(result[right_face], source[right_face]),
        "el mensaje lo declara": message.lower().startswith("sin selección"),
    }
    failures = [name for name, ok in checks.items() if not ok]
    for name, ok in checks.items():
        print(f"[Prompt] {'PASS' if ok else 'FAIL'} {name}", flush=True)
    if failures:
        print(f"[Prompt] mensaje={message}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
