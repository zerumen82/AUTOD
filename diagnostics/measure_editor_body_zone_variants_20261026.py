"""A/B medido de la zona de ropa en el motor con visión (foto real 2026-10-26).

El render real `img_editor/edit_1793042423.png` quita la ropa y conserva la
pose, pero deja cuatro defectos medibles:

  1. el antebrazo izquierdo desaparece dentro del cuerpo generado;
  2. el pegado deja escalón de tono y costura recta (clavícula, cadera, codo);
  3. queda una cuña del pantalón azul marino;
  4. el pecho se inventa alto (sobre el hombro).

Hipótesis a medir, una por variante (misma foto, misma semilla, mismo texto
del log real, mismo ComfyUI del usuario):

  V0  receta vigente: ropa dilatada 12 px + 14 px, sin excluir piel, fundido
      de 6 px hacia dentro. Es la referencia y debe reproducir el defecto.
  V1  piel medida fuera de la zona + una sola dilatación proporcional.
  V2  V1 + armonización (tono de la piel del parche al anillo original y
      seamlessClone en el borde).
  V3  V2 + umbral de ropa más bajo (la ropa en sombra entra) + margen de
      contexto en la máscara que ve el modelo, escribiendo solo la zona.

No reinicia la UI ni ComfyUI: usa el servidor que ya está corriendo y el
modelo residente. Guarda resultados y métricas en diagnostics/out.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roop.img_editor.img_editor_manager import (  # noqa: E402
    ImgEditorManager, _skin_pixels,
)
from roop.img_editor.selected_region_edit import (  # noqa: E402
    VISION_BODY_MODEL, _described_subject_mask, _dilate,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagnostics" / "out" / "editor_body_zone_variants_20261026"
SOURCE = (
    ROOT / ".autodeep_temp"
    / "e85e04ee7ba6ba4834d455b3a9b34f51a39ac58730fcf040f384a9d431b64aad"
    / "f828cbc6-e645-405d-b80f-2c1576641e71.jpg"
)
RAW_PROMPT = "la mujer debe ir sin ropa"
INSTRUCTION = "The woman must go without clothes"
# Texto exacto que envió la ruta de producción en el render del 2026-10-26.
PROMPT = (
    "Instruction: The woman must go without clothes. Keep the same room, "
    "objects, lighting and the exact same framing, composition and camera. "
    "The face of the edited person may change. Apply the requested change as "
    "completely, strongly and obviously as possible. The woman is fully nude, "
    "bare skin, no clothing at all. The requested change must be clearly "
    "visible in the result; do not preserve the previous state of what was "
    "asked to change. Leave every other person in the photo exactly as they are."
)
SEED = 424242
# El antebrazo izquierdo, medido sobre la foto: del codo a la muñeca.
ARM_BOX = (188, 300, 220, 425)
# La cadera generada y el hombro original, para el salto de tono.
TONES = {
    "hombro_original": (222, 205, 250, 225),
    "pecho_generado": (280, 220, 325, 265),
}

VARIANTS = [
    {"name": "V0_baseline", "threshold": 0.45, "dilate": 12, "second": 14,
     "skin": False, "margin": 0, "blend": False},
    {"name": "V1_skin_prop", "threshold": 0.45, "dilate": 6, "second": 0,
     "skin": True, "margin": 0, "blend": False},
    {"name": "V2_harmonized", "threshold": 0.45, "dilate": 6, "second": 0,
     "skin": True, "margin": 0, "blend": True},
    {"name": "V3_cover_margin", "threshold": 0.30, "dilate": 8, "second": 0,
     "skin": True, "margin": 28, "blend": True},
    # V4: el agujero grande de la receta vigente (el modelo obedece: es lo que
    # le da una prenda entera que borrar) y la piel original protegida SOLO al
    # pegar. El brazo está delante de la prenda, así que recuperarlo es
    # geométricamente correcto; el cuerpo generado queda detrás.
    {"name": "V4_big_hole_skin_write", "threshold": 0.45, "dilate": 12,
     "second": 14, "skin": False, "margin": 0, "blend": True,
     "protect_skin_write": True},
]


def lab(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2LAB)


def lab_mean(arr: np.ndarray, mask: np.ndarray):
    if not mask.any():
        return None
    return arr[mask].mean(axis=0)


def lab_distance(a, b) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.linalg.norm(a - b))


def skin_guard(lab_img: np.ndarray, clothes, subject, skin) -> bool:
    """La prenda del mismo color que la piel no se excluye de la zona.

    Sin esta guarda, un vestido color piel quedaría fuera de la zona y el
    pedido de desnudo no lo tocaría. La comparación es de color medido, no de
    vocabulario.
    """
    garment_skin = clothes & skin & subject
    if int(garment_skin.sum()) < 400:
        return True
    garment = lab_mean(lab_img, clothes & subject)
    skin_mean = lab_mean(lab_img, skin & subject)
    distance = lab_distance(garment, skin_mean)
    print(f"[Variants] color prenda={garment} piel={skin_mean} d={distance:.1f}",
          flush=True)
    return distance < 22.0


def build_zone(manager, image, subject, variant):
    segmenter = manager._get_clothing_segmenter()
    mask, _ = segmenter.segment_clothing(image, threshold=variant["threshold"])
    if mask is None:
        raise RuntimeError("la ropa no se pudo medir")
    clothes = np.asarray(mask.convert("L")) > 127
    skin = _skin_pixels(np.asarray(image.convert("RGB")))
    lab_img = lab(image)
    exclude = bool(variant["skin"])
    if exclude and skin_guard(lab_img, clothes, subject, skin):
        exclude = False
        print("[Variants] la prenda es del color de la piel: no se excluye piel",
              flush=True)
    zone = _dilate(clothes, variant["dilate"]) & subject
    if variant["second"]:
        zone = _dilate(zone, variant["second"]) & subject
    write_zone = zone
    if exclude:
        write_zone = zone & ~skin
    if variant.get("protect_skin_write"):
        write_zone = zone & ~skin
    if int(zone.sum()) < 200:
        raise RuntimeError("zona vacía")
    delivered = zone
    if variant["margin"]:
        delivered = _dilate(zone, variant["margin"]) & subject
    return zone, write_zone, delivered, clothes, skin


def tone_match(patch: Image.Image, image: Image.Image, box, zone_crop,
               subject, skin, ring_px=14) -> Image.Image:
    """Iguala la piel del parche al anillo de piel original que lo rodea.

    El modelo entrega una piel más clara y menos cálida que la foto (medido:
    L 207 frente a 178 en el hombro). Un desplazamiento medido en Lab, acotado,
    quita el escalón sin inventar color.
    """
    full_zone = np.zeros(subject.shape, dtype=bool)
    full_zone[box[1]:box[3], box[0]:box[2]] = zone_crop
    ring = _dilate(full_zone, ring_px) & ~full_zone & subject & skin
    gen = np.asarray(patch.convert("RGB"))
    gen_zone = zone_crop
    gen_skin = _skin_pixels(gen) & gen_zone
    ring_mean = lab_mean(lab(image), ring)
    patch_mean = lab_mean(cv2.cvtColor(gen, cv2.COLOR_RGB2LAB).astype(np.float32),
                          gen_skin)
    if ring_mean is None or patch_mean is None:
        return patch
    shift = np.clip(ring_mean - patch_mean, -18.0, 18.0)
    print(f"[Variants] armonización Lab shift={shift.round(1).tolist()}", flush=True)
    patch_lab = cv2.cvtColor(gen, cv2.COLOR_RGB2LAB).astype(np.float32) + shift
    patch_lab = np.clip(patch_lab, 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(patch_lab, cv2.COLOR_LAB2RGB))


def compose_plain(image, patch, box, zone_crop, subject, skin, soft_px=6):
    radius = max(2, min(soft_px, int(min(patch.size) * 0.02)))
    feathered = cv2.GaussianBlur(
        zone_crop.astype(np.float32) * 255.0, (radius * 2 + 1, radius * 2 + 1), 0)
    feathered[~zone_crop] = 0.0
    soft = Image.fromarray(feathered.astype(np.uint8), "L")
    out = image.copy()
    out.paste(Image.composite(patch, image.crop(box), soft), box)
    return out


def compose_blend(image, patch, box, zone_crop, subject, skin, soft_px=10):
    """Funde el borde con Poisson sobre una máscara crecida y difuminada.

    El fundido de la receta vigente solo va hacia dentro (6 px) y deja el
    escalón de tono en el borde. Poisson iguala el borde; el exterior sigue
    siendo la foto original fuera de la máscara.
    """
    plain = compose_plain(image, patch, box, zone_crop, subject, skin, soft_px)
    full_zone = np.zeros(subject.shape, dtype=bool)
    full_zone[box[1]:box[3], box[0]:box[2]] = zone_crop
    effective = _dilate(full_zone, 2)
    alpha = cv2.GaussianBlur(
        effective.astype(np.float32) * 255.0, (17, 17), 0)
    alpha[~effective] = 0.0
    alpha_img = Image.fromarray(alpha.astype(np.uint8), "L")
    alpha_arr = np.asarray(alpha_img) >= 128
    xs, ys = np.nonzero(alpha_arr)
    if not xs.size:
        return plain
    center = (int((xs.min() + xs.max()) // 2), int((ys.min() + ys.max()) // 2))
    try:
        blended = cv2.seamlessClone(
            cv2.cvtColor(np.asarray(plain.convert("RGB")), cv2.COLOR_RGB2BGR),
            cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR),
            alpha.astype(np.uint8), center, cv2.NORMAL_CLONE,
        )
    except cv2.error as error:
        print(f"[Variants] seamlessClone omitido: {error}", flush=True)
        return plain
    candidate = Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    return Image.composite(candidate, image, alpha_img)


def crop_for(image, zone, context=0.35, maximum=768, minimum=64):
    ys, xs = np.nonzero(zone)
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    pad_x = int(round((x2 - x1) * context))
    pad_y = int(round((y2 - y1) * context))
    box = (max(0, x1 - pad_x), max(0, y1 - pad_y),
           min(image.width, x2 + pad_x), min(image.height, y2 + pad_y))
    crop = image.crop(box)
    scale = maximum / max(1, max(crop.size))
    size = (max(minimum, int(round(crop.width * scale / 16)) * 16),
            max(minimum, int(round(crop.height * scale / 16)) * 16))
    return box, crop, size


def resize_mask(mask, box, size):
    crop = mask[box[1]:box[3], box[0]:box[2]]
    return np.asarray(
        Image.fromarray(crop.astype(np.uint8) * 255, "L").resize(
            size, Image.Resampling.NEAREST)) > 127


def seam_ratio(image: Image.Image, zone: np.ndarray) -> float:
    grad = cv2.Laplacian(
        np.asarray(image.convert("L")).astype(np.float32), cv2.CV_32F)
    grad = np.abs(grad)
    boundary = _dilate(zone, 1) & ~zone
    inner = zone & ~_dilate(~zone, 2)
    if not boundary.any() or not inner.any():
        return 0.0
    return float(grad[boundary].mean()) / max(1e-6, float(grad[inner].mean()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    image = Image.open(SOURCE).convert("RGB")
    manager = ImgEditorManager()
    queries = manager._subject_queries(RAW_PROMPT, INSTRUCTION)
    subject, query = _described_subject_mask(manager, image, queries)
    if subject is None:
        print("[Variants] persona no medida", flush=True)
        return 1
    print(f"[Variants] persona «{query}»={int(subject.sum())} px", flush=True)

    source_arr = np.asarray(image).astype(np.int16)
    skin_full = _skin_pixels(np.asarray(image))
    report = {"source": str(SOURCE), "prompt": PROMPT, "seed": SEED,
              "variants": []}
    masks_view = image.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 255, 0)]
    images = {}

    only = None
    for arg in sys.argv[1:]:
        if arg.startswith("--only="):
            only = arg.split("=", 1)[1]
    prepared = []
    for index, variant in enumerate(VARIANTS):
        if only and variant["name"] != only:
            continue
        zone_gen, zone_write, delivered, clothes, skin = build_zone(
            manager, image, subject, variant)
        box, crop, size = crop_for(image, zone_gen)
        print(f"[Variants] {variant['name']}: agujero={int(zone_gen.sum())} px "
              f"escritura={int(zone_write.sum())} px "
              f"mascara_modelo={int(delivered.sum())} px crop={crop.size}->{size}",
              flush=True)
        contour = _dilate(zone_write, 1) & ~zone_write
        overlay = np.asarray(masks_view).copy()
        overlay[contour] = colors[index % len(colors)]
        masks_view = Image.fromarray(overlay)
        prepared.append(
            (variant, zone_gen, zone_write, delivered, clothes, box, crop, size))

    masks_view.resize((masks_view.width * 2, masks_view.height * 2),
                      Image.Resampling.LANCZOS).save(OUT / "zones_overlay.png")
    if "--masks" in sys.argv:
        manager._release_local_gpu_before_comfy()
        return 0

    from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
    client = get_flux_edit_comfy_client()
    client._loaded = True
    client._flux_version = VISION_BODY_MODEL
    ok, note = client.load(flux_version=VISION_BODY_MODEL, allow_vision_engine=True)
    if not ok:
        print(f"[Variants] motor no disponible: {note}", flush=True)
        return 1
    print(f"[Variants] motor: {note}", flush=True)

    for variant, zone_gen, zone, delivered, clothes, box, crop, size in prepared:
        manager._release_local_gpu_before_comfy()
        started = time.time()
        generated, note = client.generate(
            prompt=PROMPT,
            negative_prompt="",
            num_inference_steps=12,
            guidance_scale=1.0,
            denoise=1.0,
            seed=SEED,
            progress_callback=None,
            prompt_authority=True,
            no_kontext=True,
            image=crop.resize(size, Image.Resampling.LANCZOS),
            mask_image=Image.fromarray(
                resize_mask(delivered, box, size).astype(np.uint8) * 255, "L"),
            keep_source_pixels=True,
            preserve_frame=True,
        )
        elapsed = time.time() - started
        if generated is None or generated.image is None:
            print(f"[Variants] {variant['name']} falló: {note}", flush=True)
            continue
        patch = generated.image.convert("RGB").resize(crop.size, Image.Resampling.LANCZOS)
        patch.save(OUT / f"{variant['name']}_patch.png")
        np.save(OUT / f"{variant['name']}_zone_gen.npy", zone_gen)
        np.save(OUT / f"{variant['name']}_zone_write.npy", zone)
        np.save(OUT / f"{variant['name']}_delivered.npy", delivered)
        zone_crop = zone[box[1]:box[3], box[0]:box[2]]
        if variant["blend"]:
            patch = tone_match(patch, image, box, zone_crop, subject, skin_full)
            result = compose_blend(image, patch, box, zone_crop, subject, skin_full)
        else:
            result = compose_plain(image, patch, box, zone_crop, subject, skin_full)
        result = result.convert("RGB")
        path = OUT / f"{variant['name']}.png"
        result.save(path)
        images[variant["name"]] = result

        out_arr = np.asarray(result).astype(np.int16)
        diff = np.abs(out_arr - source_arr).mean(axis=2)
        arm_delta = float(diff[ARM_BOX[1]:ARM_BOX[3], ARM_BOX[0]:ARM_BOX[2]].mean())
        outside = float(((diff > 6) & ~subject).sum()) * 100.0 / max(1, int((~subject).sum()))
        # Ropa residual: píxeles de la persona, fuera de la zona, con el color
        # medido de la prenda (Lab).
        garment_mean = lab_mean(lab(image), clothes & subject)
        lab_out = lab(result)
        if garment_mean is not None:
            dist = np.linalg.norm(lab_out - garment_mean, axis=2)
            residual = int(((dist < 26.0) & subject & ~zone).sum())
        else:
            residual = -1
        def box_mean(arr, b):
            return arr[b[1]:b[3], b[0]:b[2]].reshape(-1, 3).mean(axis=0)
        tone = {
            "hombro_original": box_mean(lab(image), TONES["hombro_original"]).round(1).tolist(),
            "pecho_generado": box_mean(lab(result), TONES["pecho_generado"]).round(1).tolist(),
        }
        tone_delta = lab_distance(
            np.array(tone["hombro_original"]), np.array(tone["pecho_generado"]))
        item = {
            "variant": variant["name"],
            "seconds": round(elapsed, 1),
            "zone_px": int(zone.sum()),
            "delivered_px": int(delivered.sum()),
            "arm_mean_delta": round(arm_delta, 2),
            "outside_changed_pct": round(outside, 4),
            "residual_garment_px": residual,
            "seam_ratio": round(seam_ratio(result, zone), 2),
            "tone_lab": tone,
            "tone_de": round(tone_delta, 1),
        }
        print(f"[Variants] {variant['name']} {elapsed:.0f}s -> {item}", flush=True)
        report["variants"].append(item)

    (OUT / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if images:
        width = max(img.width for img in images.values())
        height = sum(img.height for img in images.values())
        sheet = Image.new("RGB", (width, height), (20, 20, 20))
        y = 0
        for img in images.values():
            sheet.paste(img, (0, y))
            y += img.height
        sheet.save(OUT / "contact_sheet.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
