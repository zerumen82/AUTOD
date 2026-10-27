"""Explicit measured selection: one Kontext pass, restricted composition.

Called before textual routing so neither pose, body nor inferred quality can
override the region chosen by the user. An invalid choice never falls back to
a global edit. The original photo and measured support remain authoritative
after optional classical finishing.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageFilter

from roop.img_editor import element_selection as selection
from roop.img_editor.gen_prompt_config import get_kontext_model_config
from roop.img_editor.edit_execution import allow_model
from roop.img_editor.img_editor_manager import _skin_pixels

# El motor CON VISIÓN para un cambio corporal medido que reinterpreta el
# marco. Medido en FLUX.1/Kontext: la participación de piel en la zona de
# prenda medida baja de 0,2675 a 0,0676 (el modelo viste más) y el negative
# es inerte porque el muestreo va a CFG=1. El motor con visión (Qwen-VL ve la
# foto y el texto es la única autoridad) es el único del proyecto medido que
# sigue una instrucción de estado final: piel 0,5299. La excepción de motor
# sigue siendo estrecha: solo la pide la ruta global medida.
VISION_BODY_MODEL = "LongCat-Image-Edit-Turbo-Q4_K_S.gguf"


def _measured_clothing_zone(manager, image, support, analysis=None):
    """Measured clothing of the selected subject, inside the selection.

    The zone the model regenerates is measured, never a word list: the
    clothing segmenter locates the garments of the photo and they are
    intersected with the measured selection. Returns None when the garments
    cannot be located, so the caller falls back to the measured support.
    """
    try:
        segmenter = manager._get_clothing_segmenter()
    except Exception:
        return None
    if not segmenter or not segmenter.is_available():
        return None
    try:
        mask, _ = segmenter.segment_clothing(image, threshold=0.45)
    except Exception:
        return None
    if mask is None:
        return None
    clothes = np.asarray(mask.convert("L")) > 127
    if not clothes.any():
        return None
    # The garment border is part of the garment: the lining that survives
    # around the edges is what the model leaves behind. A small dilation
    # covers it without giving the model room to paint a new garment.
    clothes = _dilate(clothes, 12)
    zone = clothes & support
    # The head of the selected subject is not part of the garment: leaving it
    # out of the zone keeps the face of the photo, which the model otherwise
    # re-invents (measured: the whole silhouette as zone changed the face and
    # the hair).
    if analysis is not None:
        for region in analysis.regions:
            if region.kind == "head" and isinstance(region.mask, Image.Image):
                head = np.asarray(region.mask.convert("L")) > 127
                if head.shape == zone.shape:
                    zone = zone & ~_head_with_hair(head, support)
    # A garment zone smaller than a few pixels cannot carry the change.
    if int(zone.sum()) < 200:
        return None
    return zone


def _dilate(mask, radius):
    """Grow a boolean mask by a radius of pixels."""
    if radius <= 0:
        return mask
    import cv2
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def _lab_mean(image, mask):
    """Mean Lab of the pixels of `image` inside `mask`, or None."""
    if mask is None or not mask.any():
        return None
    import cv2
    arr = np.asarray(image.convert("RGB"))
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
    return lab[mask].mean(axis=0)


def _harmonize_patch(patch, image, box, write_crop, skin, ring_px=14):
    """Match the patch's skin tone to the ring of original skin around it.

    Measured on the user's photo (2026-10-26, V2/V4): the model delivers a
    skin that is lighter and less warm than the original (L 207 against 178 on
    the shoulder, ΔE 19). A feathered ellipse alone leaves that step. A
    measured Lab shift, bounded to ±18 per channel, moves the whole patch
    toward the original without inventing color, and it only touches skin
    pixels: the generated torso keeps its own texture.
    """
    import cv2
    if skin is None or not skin.any():
        return patch
    full_write = np.zeros((image.height, image.width), dtype=bool)
    full_write[box[1]:box[3], box[0]:box[2]] = write_crop
    ring = _dilate(full_write, ring_px) & ~full_write & skin
    if not ring.any():
        return patch
    ring_mean = _lab_mean(image, ring)
    if ring_mean is None:
        return patch
    gen = np.asarray(patch.convert("RGB"))
    gen_lab = cv2.cvtColor(gen, cv2.COLOR_RGB2LAB).astype(np.float32)
    gen_skin = _skin_pixels(gen) & write_crop
    patch_mean = _lab_mean(Image.fromarray(gen), gen_skin)
    if patch_mean is None:
        return patch
    shift = np.clip(ring_mean - patch_mean, -18.0, 18.0)
    if float(np.linalg.norm(shift)) > 1.5:
        print(
            f"[ImgEditor] Armonización de tono Lab shift="
            f"{shift.round(1).tolist()}",
            flush=True,
        )
    gen_lab = np.clip(gen_lab + shift, 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(gen_lab, cv2.COLOR_LAB2RGB))


def _vision_zone_pass(manager, client, image, zone, support, prompt,
                      negative_prompt, seed, settings, progress_callback,
                      cancel_check, recipe=None, label="motor con visión"):
    """Measured zone on the vision engine, at the person's own resolution.

    Measured on the user's photo (2026-10-26, nine real renders, seed
    424242): Kontext answers the measured garment zone with ANOTHER garment
    - a leopard body, an orange top, two red bikinis, a green swimsuit -
    at denoise 0.80-1.00, guidance 3.5/4.6, with the affirmative end state
    and at twice the resolution. In the whole 640x640 frame the woman is
    ~250x540 px, so the garment zone has almost no latent room: the fabric
    is re-resolved as fabric whatever the text says. The vision engine does
    remove it, and on the crop of the person (768 px, 480x768) it keeps the
    pose (silhouette IoU 0.93) with the exterior at 0 px. The crop is the
    measured zone with the configured context, never a fixed box.

    Returns the composed photo; the caller keeps the measured zone as its
    support, so nothing outside it is written.
    """
    if recipe is None:
        recipe = {"steps": 12, "guidance": 1.0, "denoise": 1.0, "no_kontext": True}
    region = support if support is not None else zone
    # The garment border is part of the garment, and the first real render of
    # this recipe left a pale band of the sleeve outside the measured zone
    # (measured 16:0x: the strap edge survived as a ghost over the shoulder).
    # Grow the zone before writing, never past the measured person.
    zone = _dilate(np.asarray(zone, dtype=bool), 14)
    if support is not None:
        zone &= np.asarray(support, dtype=bool)
    # v5.1253 — el modelo debe recibir el agujero grande para obedecer (medido:
    # zona 48k px ≠ bañador; zona 30k px = otra prenda). Pero lo que se ESCRIBE
    # es la ropa medida, no todo el agujero: el antebrazo delante de la prenda
    # está dentro del agujero (el modelo lo pinta de torso) y debe recuperarse
    # de la foto original al pegar. Sin esta guarda el brazo desaparece
    # (medido: brazo Δ 69.7 → 0.0 en V4 vs V0).
    skin = _skin_pixels(np.asarray(image.convert("RGB")))
    garment = zone & ~skin
    write_zone = (zone & ~skin) if (skin is not None and skin.any()
                                    and int(garment.sum()) >= 400) else zone
    if skin is not None and skin.any() and int(garment.sum()) < 400:
        print("[ImgEditor] la prenda es del color de la piel: "
              "no se excluye piel al escribir", flush=True)
    # Calculate crop box from the actual edit zone (clothing), not the whole person.
    # This gives the model more resolution on the area being edited.
    ys, xs = np.nonzero(zone)
    if not xs.size:
        return None, "la zona medida no tiene píxeles"
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    context = float(settings.get("crop_context_fraction", 0.35))
    if not math.isfinite(context) or context < 0:
        return None, "contexto de recorte inválido"
    pad_x = math.ceil((x2 - x1) * context)
    pad_y = math.ceil((y2 - y1) * context)
    box = (max(0, x1 - pad_x), max(0, y1 - pad_y),
           min(image.width, x2 + pad_x), min(image.height, y2 + pad_y))
    minimum = max(64, int(settings.get("minimum_reference_side", 64)))
    maximum = max(minimum, int(settings.get("maximum_reference_side", 768)))
    crop = image.crop(box)
    scale = maximum / max(1, max(crop.size))
    size = (max(minimum, int(round(crop.width * scale / 16)) * 16),
            max(minimum, int(round(crop.height * scale / 16)) * 16))
    crop_big = crop.resize(size, Image.Resampling.LANCZOS)
    zone_crop = np.asarray(zone, dtype=bool)[box[1]:box[3], box[0]:box[2]]
    if not zone_crop.any():
        return None, "la zona medida quedó fuera del recorte"
    zone_big = np.asarray(
        Image.fromarray(zone_crop.astype(np.uint8) * 255, "L").resize(
            size, Image.Resampling.NEAREST)
    ) > 127
    print(
        f"[ImgEditor] Sin selección: {label} en el recorte de la "
        f"zona medida {crop.size} → {size}, denoise="
        f"{float(recipe.get('denoise', 1.0)):.2f}",
        flush=True,
    )
    manager._release_local_gpu_before_comfy()
    generate_options = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": int(recipe.get("steps", 12)),
        "guidance_scale": float(recipe.get("guidance", 1.0)),
        "denoise": float(recipe.get("denoise", 1.0)),
        "seed": seed,
        "progress_callback": progress_callback,
        "prompt_authority": True,
        "no_kontext": bool(recipe.get("no_kontext", True)),
        "image": crop_big,
        "mask_image": Image.fromarray(zone_big.astype(np.uint8) * 255, "L"),
        "keep_source_pixels": True,
        "preserve_frame": True,
        "cancel_check": cancel_check,
    }
    if recipe.get("kontext_img2img_denoise") is not None:
        generate_options["kontext_img2img_denoise"] = float(
            recipe["kontext_img2img_denoise"])
        generate_options["step_cap"] = int(recipe.get("steps", 12))
    generated, note = client.generate(**generate_options)
    if cancel_check and cancel_check():
        return None, "Cancelado"
    if generated is None or generated.image is None:
        return None, f"el motor no produjo una imagen: {note}"
    import cv2
    patch = generated.image.convert("RGB").resize(crop.size, Image.Resampling.LANCZOS)
    write_crop = write_zone[box[1]:box[3], box[0]:box[2]]
    if not write_crop.any():
        write_crop = zone_crop
    # v5.1253 — el parche se escribe sobre la ZONA MEDIDA (la ropa), no sobre
    # todo el agujero: el antebrazo delante de la prenda vuelve de la foto
    # original. La máscara del pegado es la write_zone, no el agujero entero.
    patch = _harmonize_patch(patch, image, box, write_crop, skin)
    soft_zone = Image.fromarray(write_crop.astype(np.uint8) * 255, "L")
    radius = max(2, min(6, int(min(crop.size) * 0.02)))
    feathered = cv2.GaussianBlur(
        write_crop.astype(np.float32) * 255.0,
        (radius * 2 + 1, radius * 2 + 1), 0)
    feathered[~write_crop] = 0.0
    soft = Image.fromarray(feathered.astype(np.uint8), "L")
    composed = image.copy()
    composed.paste(Image.composite(patch, crop, soft), box)
    # v5.1253 — Poisson sobre una máscara crecida del borde: el fundido solo
    # hacia dentro dejaba el escalón de tono en la clavícula y la cadera
    # (medido: seam_ratio 2.07 en V0 → 1.08 en V4). El exterior de la write_zone
    # sigue siendo la foto original fuera de la máscara crecida.
    try:
        full_write = np.zeros((image.height, image.width), dtype=bool)
        full_write[box[1]:box[3], box[0]:box[2]] = write_crop
        effective = _dilate(full_write, 2)
        alpha = cv2.GaussianBlur(
            effective.astype(np.float32) * 255.0, (17, 17), 0)
        alpha[~effective] = 0.0
        alpha_arr = np.asarray(Image.fromarray(alpha.astype(np.uint8), "L"))
        ys, xs = np.nonzero(alpha_arr >= 128)
        if xs.size:
            center = (int((xs.min() + xs.max()) // 2),
                      int((ys.min() + ys.max()) // 2))
            blended = cv2.seamlessClone(
                cv2.cvtColor(np.asarray(composed.convert("RGB")),
                             cv2.COLOR_RGB2BGR),
                cv2.cvtColor(np.asarray(image.convert("RGB")),
                             cv2.COLOR_RGB2BGR),
                alpha.astype(np.uint8), center, cv2.NORMAL_CLONE,
            )
            composed = Image.fromarray(
                cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
            print("[ImgEditor] Borde del cuerpo armonizado con la foto", flush=True)
    except cv2.error as blend_error:
        print(f"[ImgEditor] Ajuste de borde omitido: {blend_error}", flush=True)
    del soft_zone
    return composed, note


def _person_without_head(support, analysis):
    """The measured person, with the head that belongs to them removed.

    The head stays out of the hole so the sampler keeps that face. Heads
    that do not overlap this person are not part of the zone anyway.
    """
    person = np.asarray(support, dtype=bool)
    zone = person.copy()
    if analysis is not None:
        for region in analysis.regions:
            if region.kind != "head" or not isinstance(region.mask, Image.Image):
                continue
            head = np.asarray(region.mask.convert("L")) > 127
            if head.shape != zone.shape or not head.any():
                continue
            if float((head & zone).sum()) / float(head.sum()) < 0.30:
                continue
            zone &= ~_head_with_hair(head, person)
    if int(zone.sum()) < 200:
        return None
    return zone


def _head_ellipse(face, shape):
    """Face box grown into the head, in the same proportions used elsewhere."""
    height, width = shape
    try:
        x1, y1, x2, y2 = [float(value) for value in face.bbox]
    except Exception:
        return None
    fw, fh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    canvas = np.zeros((height, width), dtype=np.uint8)
    import cv2
    cv2.ellipse(
        canvas,
        (int(round((x1 + x2) * 0.5)), int(round(y1 + fh * 0.42))),
        (max(1, int(round(fw * 0.58))), max(1, int(round(fh * 0.65)))),
        0, 0, 360, 255, -1,
    )
    return canvas > 0


def _head_hair_margin() -> float:
    """Margen bajo el borde inferior de la cabeza, en fracción de su alto.

    Read from config; falls back to 0.15 (measured on the real photo: the
    nape and ponytail hang below the trimmest possible head ellipse).
    """
    try:
        value = float(selection.load_settings()["edit"].get(
            "head_hair_margin_ratio", 0.15))
        if 0.0 <= value <= 0.5:
            return value
    except Exception:
        pass
    return 0.15


def _head_with_hair(head, person):
    """The measured head plus the hair that belongs to that same head.

    The detector fits its box to the *visible* face. On a profile that leaves
    the ponytail out of any face-fitted ellipse, so the nape and the jaw stay
    inside the generation zone and the sampler paints a second head there.
    Measured on the user's photo (27/10, 640x640): 61 % of the ponytail and
    72 % of the jaw/neck were inside the zone, and the delivery came back
    with a duplicated translucent head.

    The hair of that head is the part of THAT person, above the head's own
    lower edge plus the configured margin, that touches the head. No wording,
    no colour list: the person support the editor already measured decides.
    """
    if head is None:
        return None
    head = np.asarray(head, dtype=bool)
    if not head.any() or person is None:
        return head
    person = np.asarray(person, dtype=bool)
    if person.shape != head.shape:
        return head
    import cv2

    ys = np.nonzero(head)[0]
    span = int(ys.max() - ys.min() + 1)
    height = head.shape[0]
    limit = int(round(ys.max() + _head_hair_margin() * span))
    limit = max(0, min(height - 1, limit))
    band = person & (np.arange(height)[:, None] <= limit)
    if not band.any():
        return head
    count, labels = cv2.connectedComponents(
        band.astype(np.uint8), connectivity=8)
    keep = head.copy()
    for index in range(1, count):
        component = labels == index
        if (component & head).any():
            keep |= component
    return keep


def _faces_inside(mask, faces):
    """Faces whose center falls inside the measured subject."""
    if mask is None or not faces:
        return []
    height, width = mask.shape
    inside = []
    for face in faces:
        try:
            x1, y1, x2, y2 = [float(value) for value in face.bbox]
        except Exception:
            continue
        cx = int(round((x1 + x2) * 0.5))
        cy = int(round((y1 + y2) * 0.5))
        if 0 <= cx < width and 0 <= cy < height and bool(mask[cy, cx]):
            inside.append(face)
    return inside


def _face_center(face):
    try:
        x1, y1, x2, y2 = [float(value) for value in face.bbox]
    except Exception:
        return None
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def _zone_of_face(mask, faces, chosen):
    """Pixels of a shared mask that sit closer to one measured face."""
    if mask is None or chosen is None or not mask.any():
        return None
    chosen_center = _face_center(chosen)
    if chosen_center is None:
        return None
    others = []
    for face in faces or []:
        if face is chosen:
            continue
        center = _face_center(face)
        if center is not None:
            others.append(center)
    if not others:
        return mask
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    chosen_d = (xs - chosen_center[0]) ** 2 + (ys - chosen_center[1]) ** 2
    other_d = None
    for center in others:
        distance = (xs - center[0]) ** 2 + (ys - center[1]) ** 2
        other_d = distance if other_d is None else np.minimum(other_d, distance)
    zone = np.zeros(mask.shape, dtype=bool)
    zone[ys[chosen_d <= other_d], xs[chosen_d <= other_d]] = True
    if int(zone.sum()) < 200:
        return None
    return zone


def _zone_measured_for_subject(manager, image, faces, query):
    """The region the segmenter assigns to one face for this subject phrase.

    Each face is scored by the mean of the segmenter's own map inside its
    box. The winner has to lead the next face by the same margin the concept
    scorer already uses, and the map is cut with the segmenter's own
    threshold. No word list decides the side.
    """
    query = (query or "").strip()
    if not query or len(faces or []) < 2:
        return None, None
    try:
        masker = manager._get_clipseg_masker()
    except Exception:
        return None, None
    if masker is None or not hasattr(masker, "probability"):
        return None, None
    try:
        heat = masker.probability(image, query)
    except Exception:
        return None, None
    if heat is None:
        return None, None
    heat = np.asarray(heat, dtype=float)
    if heat.shape[:2] != (image.height, image.width):
        return None, None
    scored = []
    for face in faces:
        try:
            x1, y1, x2, y2 = [int(round(float(value))) for value in face.bbox]
        except Exception:
            continue
        x1, x2 = max(0, min(image.width, x1)), max(0, min(image.width, x2))
        y1, y2 = max(0, min(image.height, y1)), max(0, min(image.height, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        scored.append((float(heat[y1:y2, x1:x2].mean()), face))
    if len(scored) < 2:
        return None, None
    scored.sort(key=lambda item: item[0], reverse=True)
    best, second = scored[0][0], scored[1][0]
    if best < 0.40 or best < second * 1.25:
        return None, None
    answered = heat >= 0.40
    zone = _zone_of_face(answered, faces, scored[0][1])
    if zone is None:
        return None, None
    return zone, scored[0][1]


def _described_subject_mask(manager, image, queries, max_coverage=0.70):
    """CLIPSeg mask of a subject the prompt already named.

    A speck is not a person. A mask of the whole frame is not one person
    either, unless the caller is about to split it between measured faces.
    """
    try:
        masker = manager._get_clipseg_masker()
    except Exception:
        return None, ""
    if masker is None:
        return None, ""
    for query in queries:
        query = (query or "").strip()
        if not query:
            continue
        try:
            mask = masker.generate_mask(image, query)
        except Exception:
            continue
        if mask is None:
            continue
        arr = np.asarray(mask.convert("L").resize(image.size, Image.Resampling.NEAREST)) > 127
        coverage = float(arr.mean()) if arr.size else 0.0
        if arr.any() and 0.02 <= coverage <= max_coverage:
            return arr, query
    return None, ""


def _zone_prompt(manager, settings, instruction, regime, negative_prompt):
    """The measured text for a change inside the selected zone.

    The template of the route, the shared enforcement clause, the measured
    recipe of the bald request (the one that avoids the documented "cap") and
    the measured negative inside the text (with CFG=1 the negative has no
    effect of its own).
    """
    text = settings["prompt_template"].format(instruction=instruction)
    if regime["enforce_clause"]:
        text += regime["enforce_clause"]
    try:
        text += manager._bald_recipe_tail(instruction)
    except Exception:
        pass
    negative = negative_prompt or regime["negative_prompt"]
    if negative:
        text += f" Avoid: {negative}."
    return text, negative


def _blend_head_boundary(image, patch, effective, box):
    """Match the patch boundary to the photo while retaining source gradients.

    The model can brighten the background around an otherwise correct head.
    A feathered ellipse still displays that bright oval. Poisson composition
    matches the measured boundary; exact exterior restoration remains final.
    """
    import cv2

    support = np.asarray(effective.convert("L")) >= 128
    ys, xs = np.where(support)
    if not xs.size:
        raise ValueError("La cabeza no tiene soporte de integración")
    center = ((int(xs.min()) + int(xs.max()) + 1) // 2,
              (int(ys.min()) + int(ys.max()) + 1) // 2)
    blended = cv2.seamlessClone(
        cv2.cvtColor(np.asarray(patch.convert("RGB")), cv2.COLOR_RGB2BGR),
        cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR),
        np.asarray(effective.crop(box).convert("L")), center, cv2.NORMAL_CLONE,
    )
    candidate = Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    return Image.composite(candidate, image, effective)


def edit_selected_region(
    manager, image, prompt, analysis, ids, *, engine, negative_prompt="",
    num_inference_steps=None, guidance_scale=None, seed=None,
    progress_callback=None, quality_mode=False, auto_upscale=False,
    enhance_tier="hd", quality_preserve_faces=True, quality_steps=None,
    cancel_check=None,
):
    """Return a protected edit or the original with an explicit omission."""
    print(f"[ImgEditor] DEBUG edit_selected_region: ids={ids}, engine={engine}", flush=True)
    def omitted(reason):
        print(f"[ImgEditor] DEBUG omitted: {reason}", flush=True)
        return image.copy(), f"Omitida: {reason}; se conserva la imagen", None

    try:
        valid, reason = selection.validate_selection(image, analysis, ids)
        if not valid:
            return omitted(reason)
        if engine != "kontext":
            return omitted("la selección visual requiere el motor Kontext")
        if not (prompt or "").strip():
            return omitted("escribe el cambio para las zonas seleccionadas")

        settings = selection.load_settings()["edit"]
        mask = selection.union_mask(analysis, ids)
        if mask is None:
            return omitted("la selección no tiene una máscara medible")
        support = np.asarray(mask) > 0
        protected = selection.protection_mask(analysis, ids)
        if protected is not None:
            support &= np.asarray(protected) == 0
        if not support.any():
            return omitted("la región coincide completamente con cabezas protegidas")
        ys, xs = np.where(support)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        context = float(settings["crop_context_fraction"])
        # A detected head is already a bounded anatomical reference. Object
        # segmenters can return only a small visible part: show enough of the
        # source around that part for the editor to recognize the whole item.
        # The write mask remains exactly the measured user selection.
        selected_kinds = {region.kind for region in analysis.regions if region.id in set(ids)}
        if "head" in selected_kinds:
            # A head crop with wide context can contain a second person, and
            # the editor then applies the instruction to the wrong face. The
            # head is already bounded, so it only needs a narrow margin.
            head_context = float(settings["head_crop_context_fraction"])
            if not math.isfinite(head_context) or head_context < 0:
                return omitted("configuración de contexto de cabeza inválida")
            context = min(context, head_context)
        if "object" in selected_kinds:
            object_context = float(settings["object_context_fraction"])
            if not math.isfinite(object_context) or object_context < 0:
                return omitted("configuración de contexto de objeto inválida")
            context = max(context, object_context)
        minimum = int(settings["minimum_reference_side"])
        if not math.isfinite(context) or context < 0 or minimum < 1:
            return omitted("configuración de recorte inválida")
        pad_x, pad_y = math.ceil((x2 - x1) * context), math.ceil((y2 - y1) * context)
        x1, x2 = max(0, x1 - pad_x), min(image.width, x2 + pad_x)
        y1, y2 = max(0, y1 - pad_y), min(image.height, y2 + pad_y)
        # Expand context for narrow objects. Never resize a mask or feed a
        # zero-sized latent to the client (it floors dimensions to 64).
        for axis in (0, 1):
            lo, hi, limit = (x1, x2, image.width) if axis == 0 else (y1, y2, image.height)
            extra = max(0, minimum - (hi - lo))
            lo = max(0, lo - (extra + 1) // 2)
            hi = min(limit, max(hi, lo + minimum))
            lo = max(0, min(lo, hi - minimum))
            if axis == 0:
                x1, x2 = lo, hi
            else:
                y1, y2 = lo, hi
        if x2 - x1 < minimum or y2 - y1 < minimum:
            return omitted("la foto no permite un recorte válido para Kontext")
        maximum = int(settings["maximum_reference_side"])
        if maximum < minimum:
            return omitted("configuración de resolución inválida")
        # The client's thumbnail then floors both axes to latent blocks. Add
        # real context for very tall/wide selections instead of a zero axis.
        target_width = max(minimum, math.ceil((y2 - y1) * minimum / maximum))
        target_height = max(minimum, math.ceil((x2 - x1) * minimum / maximum))
        for axis, target in ((0, target_width), (1, target_height)):
            lo, hi, limit = (x1, x2, image.width) if axis == 0 else (y1, y2, image.height)
            extra = max(0, target - (hi - lo))
            lo = max(0, lo - (extra + 1) // 2)
            hi = min(limit, max(hi, lo + target))
            lo = max(0, min(lo, hi - target))
            if hi - lo < target:
                return omitted("el formato de la foto no permite un latente válido")
            if axis == 0:
                x1, x2 = lo, hi
            else:
                y1, y2 = lo, hi
        if (x2 - x1) * (y2 - y1) > image.width * image.height * float(settings["crop_max_area"]):
            return omitted("el contexto supera el límite de recorte configurado")
        box = (x1, y1, x2, y2)
        original_crop = image.crop(box)
        if cancel_check and cancel_check():
            return None, "Cancelado", None

        # A nearby unselected person inside the crop makes the target
        # ambiguous, and the editor applies the instruction to that face.
        # The reference is blinded for the unselected heads; the original
        # pixels of those heads are restored by the protected composition.
        blind_radius = int(settings["blind_dilate_radius"])
        if blind_radius < 0:
            return omitted("configuración de cegado inválida")
        selected_mask = selection.union_mask(analysis, ids)
        blinded = selection.blind_unselected_in_reference(
            analysis, ids, image, dilate_radius=blind_radius, keep_mask=selected_mask)
        reference = blinded.crop(box)
        blind_note = ""
        if not np.array_equal(np.asarray(reference.convert("L")),
                              np.asarray(original_crop.convert("L"))):
            blind_note = f" | halo gris de {blind_radius}px en las cabezas no seleccionadas"
        if cancel_check and cancel_check():
            return None, "Cancelado", None

        from roop.img_editor.prompt_translator import translate_prompt
        instruction = translate_prompt(prompt.strip())
        model = get_kontext_model_config()
        # v5.1227 — el régimen de la ruta regional se MIDE con el mismo
        # analizador de conceptos que la ruta global. Antes esta ruta usaba un
        # régimen fijo (EDIT + ancla 'offset'), y con tokens limpios el ancla
        # copia la foto: el modelo devolvía la misma imagen con el zapato
        # quitado y la ropa intacta (log real 21:03). Ahora la ruta regional
        # recibe el mismo tratamiento medido: reference method, negative por
        # dirección y frase de estado final.
        regime = manager.measured_body_regime(
            prompt.strip(), instruction, engine=engine)
        # v5.1232 — la misma medición, para la APARIENCIA sobre una cabeza
        # seleccionada. Medido en el caso real del usuario (log 22:14, foto
        # con dos personas, cabeza seleccionada + «debe ser calvo»): la ruta
        # con ancla (`EDIT ref=offset`, 20 pasos) NO editó nada — el ancla
        # copia la foto y el pelo se quedó igual. El motor que sigue el
        # texto es el mismo con visión que en el cambio corporal.
        appearance_facts = {}
        try:
            appearance_facts = manager._prompt_authority_facts(
                prompt.strip(), instruction)
        except Exception:
            appearance_facts = {}
        head_selected = any(
            region.kind == "head" and region.id in set(ids)
            for region in analysis.regions)
        # People from the visual inventory have kind "object". The same
        # measured containment already makes their head editable/unblinded;
        # use it here too, rather than sending the entire scene to Kontext.
        contained_head = None
        if (not head_selected and len(ids) == 1 and not regime["body"]
                and appearance_facts.get("local_appearance")):
            contained = [
                region for region in analysis.regions
                if region.kind == "head" and selection._head_inside_selection(
                    region.mask, selected_mask,
                    float(settings["head_inclusion_coverage"]))
            ]
            if len(contained) == 1:
                contained_head = contained[0]
        # v5.1234 — el cambio de apariencia sobre la cabeza seleccionada usa
        # la RECETA FOCAL de la ruta global: el recorte con su contexto (sin
        # el lienzo regional), la edición anclada y el texto medido de
        # apariencia (con la receta de calvicie, que evita el «gorrito»).
        # Medido: esa receta SÍ ejecuta (pelo 139,4 → 196,9, cabeza calva
        # coherente) y respeta la selección (solo se escribe el soporte
        # medido). Descartadas las otras: el ancla con el lienzo no cambia
        # nada (copy machine), el agujero con el motor con visión sustituye la
        # cabeza por una inventada (el usuario, 01:40: «una edición pésima»)
        # y reinterpretar el marco ignora la zona marcada (el usuario:
        # «lo de la selección de formas detectadas no funciona»).
        appearance_head = bool(
            not regime["body"] and (head_selected or contained_head is not None)
            and appearance_facts.get("local_appearance"))
        # v5.1238 — cambio de POSE/ÁNGULO medido («CAMBIA EL ANGULO, IMAGEN
        # DE LATERAL», caso real del usuario 22/10 que no hizo nada). Un
        # cambio de postura no es un parche local: el cuerpo se mueve entero,
        # así que solo puede rehacerlo el motor que SIGUE EL TEXTO (el de
        # visión, sin ancla y sin agujero). La ruta ordinaria con lienzo +
        # ancla está medida como «copia la foto y no cambia nada» (copy
        # machine), que es exactamente lo que falló. El marco se reinterpreta
        # y las caras medidas de las demás personas se conservan de la foto
        # original (la regla de protección del proyecto).
        pose_change = bool(
            not regime["body"] and not appearance_head
            and appearance_facts.get("pose"))
        if appearance_head:
            print(
                "[ImgEditor] Selección: cambio de apariencia medido sobre la "
                "cabeza seleccionada → recorte con contexto y texto medido "
                "(la zona marcada es lo que se escribe)",
                flush=True,
            )
        # v5.1233 — SEÑAL CERO = SEÑAL, pero AVISADO, no secuestrado: una
        # instrucción que el analizador no mide no es un pedido débil, es un
        # pedido que su vocabulario no cubre (caso real del usuario, 00:09:
        # «debe ser cxalvo»). La ruta global regenera el marco y puede
        # tratarlo como señal fuerte; la ruta de selección escribe solo la
        # zona medida, así que secuestrar su motor rompería los casos
        # validados (calidad, ángulo, postura). Lo que sí se hace es **decirlo**:
        # un no-op silencioso se convierte en un aviso.
        measured_signal = bool(
            appearance_facts.get("body")
            or appearance_facts.get("local_appearance")
            or appearance_facts.get("local_region")
            or appearance_facts.get("pose")
            or appearance_facts.get("authority")
            or appearance_facts.get("structural_remove")
            or appearance_facts.get("scenario"))
        unmeasured_note = ""
        if not measured_signal and not regime["body"]:
            unmeasured_note = (
                " | sin señal medida en la instrucción (revisa la ortografía "
                "o el vocabulario: puede que el motor no la ejecute)")
            print(
                "[ImgEditor] Selección: la instrucción no tiene ninguna señal "
                "medida; se avisa en el resultado",
                flush=True,
            )
        # v5.1228 — la zona que el modelo REGENERA se mide aparte de la zona
        # que se ESCRIBE. Medido sobre la foto real: con la silueta entera como
        # zona de generación, el modelo re-inventa a la persona (cara y pelo
        # distintos). La ropa medida es la superficie de un cambio local.
        # v5.1244 — una exposición de todo el cuerpo no cabe en el fragmento
        # de ropa (log 00:54: 21568 px, el torso sí y el pantalón no). La
        # zona es la persona medida menos la cabeza.
        generation_zone = None
        zone_label = "zona medida"
        if appearance_head and contained_head is not None:
            generation_zone = support & (np.asarray(contained_head.mask.convert("L")) >= 128)
            head_mask = Image.fromarray(generation_zone.astype(np.uint8) * 255, "L")
            focal = manager._crop_focal_context(
                image, head_mask, max_area=float(settings["crop_max_area"]))
            if focal is None:
                return omitted("la cabeza contenida no admite un recorte fiable")
            box = focal[2]
            original_crop = image.crop(box)
            reference = blinded.crop(box)
        if regime["body"]:
            uncover = str(regime["direction"]).lower() == "uncover"
            full_body = bool(
                uncover and manager.measured_full_body_exposure(instruction))
            clothing = None
            if not full_body:
                clothing = _measured_clothing_zone(
                    manager, image, support, analysis)
                # v5.1247 — el vocabulario no siempre dice hasta dónde llega
                # un «quitar» («sin pantalón, sin blusa» puntúa 0). Entonces
                # decide la foto: si la ropa medida es una parte pequeña de
                # la persona, escribir ahí no quita la ropa (medido: 21568
                # px, el chaleco se volvió camiseta y el pantalón quedó).
                if (clothing is not None and uncover
                        and manager.measured_uncover_is_whole_person(
                            instruction,
                            int(np.asarray(support, dtype=bool).sum()),
                            int(clothing.sum()))):
                    full_body = True
                    print(
                        "[ImgEditor] Selección: la ropa medida cubre poco de "
                        "la persona; la zona es la persona medida menos la "
                        "cabeza",
                        flush=True,
                    )
            if full_body:
                true_full_body = manager.is_true_full_body_exposure(instruction)
                if manager.measured_full_body_exposure(instruction) and not true_full_body:
                    # v5.125x — para exposición completa medida ("sin ropa",
                    # "naked"), probar primero la zona de ropa medida. Si la
                    # ropa cubre suficiente de la persona, regenerar solo la
                    # ropa (cabeza excluida) en vez de todo el cuerpo. Si la
                    # ropa es un fragmento (p. ej. 33 %), regenerar el cuerpo
                    # sin cabeza para preservar el rostro en vez de repintar
                    # la persona entera (cara + pelo + cuerpo).
                    clothing = _measured_clothing_zone(
                        manager, image, support, analysis)
                    person_px = int(
                        np.asarray(support, dtype=bool).sum())
                    min_ratio = manager._clothing_zone_min_ratio()
                    if (clothing is not None
                            and int(clothing.sum())
                            >= min_ratio * max(1, person_px)):
                        generation_zone = clothing
                        zone_label = "zona de ropa medida"
                        print(
                            "[ImgEditor] Selección: exposición completa "
                            "medida pero la ropa cubre suficiente de la "
                            f"persona → zona de ropa medida "
                            f"({int(clothing.sum())} px)",
                            flush=True,
                        )
                    elif clothing is not None:
                        generation_zone = _person_without_head(
                            support, analysis)
                        zone_label = "persona medida menos la cabeza"
                        print(
                            "[ImgEditor] Selección: exposición completa "
                            f"medida pero la ropa ({int(clothing.sum())} px) "
                            f"es un fragmento de la persona ({person_px} px) → "
                            "persona medida menos la cabeza",
                            flush=True,
                        )
                    else:
                        generation_zone = np.asarray(
                            support, dtype=bool).copy()
                        zone_label = "persona medida completa"
                else:
                    # True full-body exposure (naked/sin ropa/desnuda) or
                    # non-full-body: always use person without head to include legs.
                    generation_zone = _person_without_head(support, analysis)
                    zone_label = "persona medida menos la cabeza"
                    if true_full_body:
                        print(
                            "[ImgEditor] Selección: exposición completa real "
                            f"(body_exposure_full) → {zone_label}",
                            flush=True,
                        )
            if generation_zone is None:
                if clothing is None:
                    clothing = _measured_clothing_zone(
                        manager, image, support, analysis)
                generation_zone = clothing
                if generation_zone is not None:
                    zone_label = "zona de ropa medida"
            if generation_zone is not None:
                print(
                    f"[ImgEditor] Selección: cambio corporal medido "
                    f"(dirección={regime['direction']}, "
                    f"intensidad={regime['intensity']:.2f}) → {zone_label} "
                    f"({int(generation_zone.sum())} px)",
                    flush=True,
                )
            else:
                print(
                    f"[ImgEditor] Selección: cambio corporal medido "
                    f"(dirección={regime['direction']}, "
                    f"intensidad={regime['intensity']:.2f}) → zona medida completa",
                    flush=True,
                )
            vision_body = bool(
                regime["body"] and generation_zone is not None
                and str(regime["direction"]).lower() == "uncover")
        # v5.1251 — la ruta regional se ejecuta siempre con el Kontext
        # configurado, así que el modelo planificado tiene que existir TAMBIÉN
        # cuando la instrucción no es un cambio corporal. Antes se asignaba
        # solo dentro de la rama corporal y cualquier otra edición con
        # selección moría con «local variable 'planned_model' referenced
        # before assignment» (probe de régimen: 3/11). Igual con `vision_body`:
        # solo existe cuando la instrucción es un cambio corporal medido.
        vision_body = False
        planned_model = model["model_file"]
        allowed, policy_note = allow_model(planned_model)
        if not allowed:
            return None, policy_note, None
        from ui.tabs.comfy_launcher import is_comfyui_running, start
        if not is_comfyui_running():
            ok, note, _ = start(directly_run=True)
            if not ok:
                return omitted(f"ComfyUI no disponible: {note}")
        manager._release_local_gpu_before_comfy()
        from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
        client = get_flux_edit_comfy_client()
        ok, note = client.load(flux_version=model["model_file"])
        if not ok:
            return omitted(f"Kontext no disponible: {note}")
        engine_label = "Kontext"
        if cancel_check and cancel_check():
            return None, "Cancelado", None
        if vision_body:
            # Cambio corporal medido con Kontext en modo img2img puro:
            # kontext_img2img_denoise alto (0.90) parte del latente de la foto
            # con mucho ruido para permitir quitar la ropa, sin ReferenceLatent
            # que ancle demasiado y devuelva la ropa.
            body_prompt = manager._compose_generation_prompt(
                instruction, engine="kontext",
                magnitude=max(0.75, float(regime["magnitude"])),
                body_transform=True,
                body_direction=regime["direction"],
                body_intensity=regime["intensity"],
            )
            if regime["enforce_clause"] and regime["enforce_clause"] not in body_prompt:
                body_prompt += regime["enforce_clause"]
            measured_negative = negative_prompt or regime["negative_prompt"]
            if measured_negative:
                body_prompt += f" Avoid: {measured_negative}."
            gen_options = {
                "prompt": body_prompt,
                "negative_prompt": measured_negative,
                "num_inference_steps": 24,
                "step_cap": 24,
                "guidance_scale": 3.5,
                "denoise": 0.88,
                "reference_latents_method": "uxo/uno",
                "seed": seed,
                "progress_callback": progress_callback,
                "prompt_authority": True,
                "no_kontext": False,
                "keep_source_pixels": True,
                "preserve_frame": True,
                "cancel_check": cancel_check,
            }
            gen_options.update({
                "image": reference,
                "mask_image": Image.fromarray(
                    (generation_zone if generation_zone is not None else support).astype(np.uint8) * 255, "L").crop(box),
                "reference_image": reference,
            })
            print(
                f"[ImgEditor] Selección: cambio corporal medido → Kontext img2img puro, "
                f"el prompt se escribe en la {zone_label} "
                "(el resto de la foto se conserva)",
                flush=True,
            )
        elif appearance_head:
            # v5.1235 — el cambio de apariencia sobre la cabeza seleccionada:
            # la RECETA FOCAL de la ruta global, que es la única medida que
            # edita una cabeza de forma coherente: el recorte con su contexto
            # a **buena resolución** (una cabeza de 136 px en una foto de 2841
            # no le da nada al modelo: medido, inventa el fragmento), la
            # edición ANCLADA (Kontext es un modelo de edición: con el ancla
            # la foto se conserva y el cambio se aplica encima; sin ancla
            # pinta una escena nueva) y el texto medido de apariencia (con la
            # receta de calvicie, que evita el «gorrito»). Lo que se escribe
            # sigue siendo solo el soporte medido: la selección manda.
            head_prompt, measured_negative = _zone_prompt(
                manager, settings, instruction, regime, negative_prompt)
            _focal = manager._scale_kontext_focal_reference(reference)
            gen_options = {
                "image": _focal,
                "prompt": head_prompt,
                "negative_prompt": measured_negative,
                "num_inference_steps": model["steps"] if num_inference_steps is None else num_inference_steps,
                "guidance_scale": model["cfg"] if guidance_scale is None else guidance_scale,
                "seed": seed,
                "progress_callback": progress_callback,
                "reference_latents_method": model["reference_latents_method"],
                "cancel_check": cancel_check,
            }
            engine_options = dict(regime["engine"])
            engine_options.pop("measured_zone_mask", None)
            engine_options.pop("img2img_denoise", None)
            gen_options.update(engine_options)
        elif pose_change:
            # Cambio de postura/ángulo medido con Kontext en modo img2img puro:
            # el muestreador parte del latente de la foto con denoise alto y
            # SIN ReferenceLatent, porque el ancla con tokens limpios conserva
            # la postura vieja (copy machine medido).
            zone_prompt, measured_negative = _zone_prompt(
                manager, settings, instruction, regime, negative_prompt)
            gen_options = {
                "image": image,
                "prompt": zone_prompt,
                "negative_prompt": measured_negative,
                "num_inference_steps": 24,
                "step_cap": 24,
                "guidance_scale": 3.5,
                "denoise": 0.88,
                # v5.1251 — sin este kwarg el cliente mantiene el
                # ReferenceLatent y el ancla devuelve la postura vieja: el
                # mensaje decía «img2img puro» y el workflow iba anclado.
                "kontext_img2img_denoise": 0.90,
                "seed": seed,
                "progress_callback": progress_callback,
                "prompt_authority": True,
                "no_kontext": False,
                "keep_source_pixels": True,
                "preserve_frame": False,
                "cancel_check": cancel_check,
            }
            print(
                "[ImgEditor] Selección: cambio de postura/ángulo medido → "
                "Kontext img2img puro, marco reinterpretado (el texto manda; "
                "las caras de las demás personas se conservan)",
                flush=True,
            )
        else:
            # En Kontext el prompt negativo es inerte (la guía va en
            # FluxGuidance y el muestreo va a CFG=1): en un cambio corporal
            # medido, el «avoid» medido viaja dentro del texto positivo.
            _negative_in_text = (
                f" Avoid: {regime['negative_prompt']}."
                if regime["body"] and regime["negative_prompt"] else ""
            )
            gen_options = {
                "image": reference,
                # v5.1228 — la frase de estado va pegada a la instrucción, antes
                # de la cláusula de contexto: el modelo lee el estado final como
                # parte de lo que tiene que hacer en esa zona.
                "prompt": settings["prompt_template"].format(
                    instruction=instruction + regime["state_suffix"])
                + regime["enforce_clause"] + _negative_in_text,
                "negative_prompt": negative_prompt or regime["negative_prompt"],
                "num_inference_steps": model["steps"] if num_inference_steps is None else num_inference_steps,
                "guidance_scale": model["cfg"] if guidance_scale is None else guidance_scale,
                "seed": seed,
                "progress_callback": progress_callback,
                "reference_latents_method": model["reference_latents_method"],
                "reference_canvas": {"max_side": settings["maximum_reference_side"],
                                     "multiple": settings["reference_canvas_multiple"]},
                "cancel_check": cancel_check,
            }
            # v5.1228 — el cambio corporal medido se ejecuta sobre la ZONA MEDIDA
            # (modo «foto»): el modelo regenera solo el interior de la máscara y la
            # foto de alrededor se conserva. Sin la máscara, el ancla copia la foto
            # entera (medido: registro 0,999 y ropa intacta) y el cambio no se
            # ejecuta; con la foto regenerada entera el resultado no encaja con el
            # original (medido: registro 0,625, desplazamiento 24 px).
            engine_options = dict(regime["engine"])
            measured_zone = bool(engine_options.pop("measured_zone_mask", False))
            if measured_zone:
                zone = support if generation_zone is None else generation_zone
                measured_mask = Image.fromarray(zone.astype(np.uint8) * 255, "L")
                gen_options["mask_image"] = measured_mask.crop(box)
                gen_options["keep_source_pixels"] = True
                # v5.1228 — sin el ancla, el modelo no puede copiar la foto: solo
                # se regenera la zona medida (el resto lo conserva el muestreador).
                # Medido sobre la foto real: con ancla la ropa quedaba intacta
                # (torso 5,2 niveles); sin ancla y con la zona de ropa medida la
                # ropa cambia y la persona sobrevive.
                engine_options["kontext_img2img_denoise"] = float(
                    engine_options.pop("img2img_denoise", 1.0))
            # El régimen medido manda sobre el valor por defecto de la config.
            gen_options.update(engine_options)
        generated, note = client.generate(**gen_options)
        if cancel_check and cancel_check():
            return None, "Cancelado", None
        if generated is None or generated.image is None:
            return omitted(f"{engine_label} no produjo una imagen: {note}")
        patch = generated.image.convert("RGB").resize(reference.size, Image.Resampling.LANCZOS)
        radius = float(settings["feather_radius"])
        if not math.isfinite(radius) or radius < 0:
            return omitted("configuración de borde inválida")
        if pose_change:
            # v5.1238 — la postura reinterpreta el marco: el cuerpo se mueve
            # entero. Las caras medidas de las demás personas se conservan.
            # El cambio corporal no entra aquí: se escribe en su zona.
            patch = generated.image.convert("RGB").resize(
                image.size, Image.Resampling.LANCZOS)
            if protected is not None and np.asarray(protected).any():
                effective = Image.fromarray(
                    (np.asarray(protected) == 0).astype(np.uint8) * 255, "L")
            else:
                effective = Image.new("L", image.size, 255)
            result = patch
        else:
            # The patch may come from the no-op retry (the vision engine).
            compose_zone = generation_zone if generation_zone is not None else support
            effective = Image.fromarray(compose_zone.astype(np.uint8) * 255, "L")
            alpha = np.asarray(effective.filter(ImageFilter.GaussianBlur(radius))).copy()
            alpha[~compose_zone] = 0
            alpha = Image.fromarray(alpha, "L")
            # The fallback inside the crop is the original photo, never the
            # blinded reference: the grey patch must never reach the result.
            result = image.copy()
            result.paste(Image.composite(patch, original_crop, alpha.crop(box)), box)
            if appearance_head and settings.get("head_blend_method") == "normal_clone":
                try:
                    matched = _blend_head_boundary(image, patch, effective, box)
                    if not manager._edit_is_nearly_unchanged(original_crop, matched.crop(box)):
                        result = matched
                        print("[ImgEditor] Borde de cabeza armonizado con la foto", flush=True)
                    else:
                        print("[ImgEditor] Ajuste de borde descartado: anulaba el cambio", flush=True)
                except Exception as blend_error:
                    print(f"[ImgEditor] Ajuste de borde omitido: {blend_error}", flush=True)
        finish_note = ""
        if quality_mode or auto_upscale:
            from roop.img_editor.top_tier_finish import run_top_tier_quality
            result, finish_note, quality_analysis = run_top_tier_quality(
                result, enhance_tier=enhance_tier, use_generative=False,
                preserve_faces=quality_preserve_faces, quality_steps=quality_steps,
                progress_callback=progress_callback, cancel_check=cancel_check,
            )
            manager._last_quality_analysis = quality_analysis
        if cancel_check and cancel_check():
            return None, "Cancelado", None
        if result is None:
            return omitted("el acabado no devolvió una imagen")
        # Keep the source geometry so exterior protection is pixel-exact even
        # when a finisher enlarges the image. Restore after every finish step.
        result = result.convert("RGB").resize(image.size, Image.Resampling.LANCZOS)
        result = Image.composite(result, image, effective)
        manager._last_instruction_compliance = {
            "compliant": None, "source": "explicit_selection",
            "note": ("Marco reinterpretado (cambio de postura); "
                     "cumplimiento visual pendiente de inspección")
            if pose_change else
            "Zona medida; el resto de la foto se conserva; "
            "cumplimiento visual pendiente de inspección",
        }
        labels = ", ".join(region.label for region in analysis.regions if region.id in set(ids))
        if pose_change:
            message = (f"Selección: {labels} | {engine_label}: {note} | "
                       "marco reinterpretado (cambio de postura; las caras "
                       f"de las demás personas se conservan){unmeasured_note}")
        elif vision_body:
            message = (f"Selección: {labels} | {engine_label}: {note} | "
                       f"prompt escrito en la {zone_label}; el resto "
                       "de la foto se conserva")
        else:
            message = (f"Selección: {labels} | {engine_label}: {note} | "
                       f"exterior protegido{blind_note}{unmeasured_note}")
        if finish_note:
            message += f" | {finish_note} (tamaño de entrada conservado para proteger el exterior)"
        return result, message, effective
    except Exception as error:
        return omitted(f"no se pudo editar la región ({error})")


def edit_described_subject(
    manager, image, raw_prompt, translated_prompt, faces, *, engine,
    negative_prompt="", seed=None, progress_callback=None, quality_mode=False,
    auto_upscale=False, enhance_tier="hd", quality_preserve_faces=True,
    quality_steps=None, cancel_check=None,
):
    """One vision pass on the person the prompt names, when nobody is selected.

    Returns None when this request is not a body uncover, so the caller keeps
    its own route. A body uncover that cannot be placed on one person keeps
    the photo: the global route was redrawing the whole frame.
    """
    if engine != "kontext":
        return None
    raw_prompt = (raw_prompt or "").strip()
    translated_prompt = (translated_prompt or "").strip()
    if not raw_prompt and not translated_prompt:
        return None
    try:
        facts = manager._prompt_authority_facts(raw_prompt, translated_prompt)
    except Exception:
        return None
    if facts.get("pose") or not facts.get("body"):
        return None
    if str(facts.get("body_direction", "change")).lower() != "uncover":
        return None
    instruction = translated_prompt or raw_prompt
    clauses = manager._split_compound_instruction(instruction) or [instruction]
    if len(clauses) > 1:
        for clause in clauses:
            try:
                clause_facts = manager._prompt_authority_facts(clause, clause)
            except Exception:
                return None
            # v5.1246 — «sin pantalón» y «sin blusa» no miden body por sí
            # solas: son la misma frase corporal cortada por comas. El veto
            # por falta de body anulaba la pasada medida y la guarda del
            # router global devolvía la foto idéntica (log del usuario
            # 2026-10-23, edit_1792748448.png). Una cláusula neutra no aporta
            # otro régimen y el texto completo llega igual al motor; sí veta
            # la que pide postura, escena, región, apariencia o quitar un
            # objeto, porque esa pasada no la escribe.
            if (clause_facts.get("pose") or clause_facts.get("scenario")
                    or clause_facts.get("local_region")
                    or clause_facts.get("local_appearance")
                    or clause_facts.get("structural_remove")):
                return None

    # This route writes the change on the measured zone of one person with
    # Kontext. If it cannot place the change on one person, it keeps the photo
    # and no model is loaded.
    from roop.img_editor.gen_prompt_config import get_kontext_model_config
    _km = get_kontext_model_config()
    _km_file = _km["model_file"]
    allowed, policy_note = allow_model(_km_file)
    if not allowed:
        return None, policy_note, None

    def kept(reason):
        print(f"[ImgEditor] Sin selección: {reason}", flush=True)
        return image.copy(), f"Sin selección: {reason}", None

    queries = []
    try:
        queries = manager._subject_queries(raw_prompt, instruction)
    except Exception:
        queries = []
    subject, query = _described_subject_mask(manager, image, queries)
    people = list(faces or [])
    inside = _faces_inside(subject, people)
    located = False
    if subject is not None:
        if len(people) >= 2 and len(inside) == 1:
            located = True
        elif len(people) == 1 and len(inside) == 1:
            located = True
        elif len(people) == 0:
            located = True
    # The binary mask covered every face. The same subject phrase is scored
    # on each face with the segmenter's map. The face that answers wins.
    if not located and len(people) >= 2:
        for candidate in queries:
            measured, chosen = _zone_measured_for_subject(
                manager, image, people, candidate)
            if measured is None or chosen is None:
                continue
            subject = measured
            inside = [chosen]
            query = candidate
            located = True
            break
    if not located:
        return kept(
            "no se pudo localizar a una sola persona descrita; "
            "se conserva la foto"
        )

    full_body = manager.measured_full_body_exposure(instruction)
    clothing = None
    zone = None
    zone_label = None
    if not full_body:
        clothing = _measured_clothing_zone(manager, image, subject, None)
        # v5.1247 — mismo criterio de la ruta de selección: cuando el
        # vocabulario no mide hasta dónde llega el «quitar», decide la foto.
        if (clothing is not None
                and manager.measured_uncover_is_whole_person(
                    instruction, int(subject.sum()), int(clothing.sum()))):
            full_body = True
            print(
                "[ImgEditor] Sin selección: la ropa medida cubre poco de la "
                "persona; la zona es la persona medida menos la cabeza",
                flush=True,
            )
    # v5.125x — cuando el vocabulario dice exposición completa
    # («sin ropa», «naked»), probar primero la zona de ropa medida. Si la
    # ropa cubre suficiente de la persona, regenerar solo la ropa en vez de
    # todo el cuerpo. Si la ropa es un fragmento, regenerar el cuerpo sin
    # cabeza para preservar el rostro.
    if full_body and manager.measured_full_body_exposure(instruction):
        clothing = _measured_clothing_zone(manager, image, subject, None)
        person_px = int(subject.sum())
        min_ratio = manager._clothing_zone_min_ratio()
        if (clothing is not None
                and int(clothing.sum())
                >= min_ratio * max(1, person_px)):
            full_body = False
            zone = clothing
            zone_label = "zona de ropa medida"
            print(
                "[ImgEditor] Sin selección: exposición completa medida pero "
                "la ropa cubre suficiente de la persona → "
                f"zona de ropa medida ({int(clothing.sum())} px)",
                flush=True,
            )
        elif clothing is not None:
            full_body = False
            zone = subject.copy()
            for face in inside:
                head = _head_ellipse(face, zone.shape)
                if head is not None:
                    zone &= ~_head_with_hair(head, subject)
            zone_label = "persona medida menos la cabeza"
            print(
                "[ImgEditor] Sin selección: exposición completa medida pero "
                f"la ropa ({int(clothing.sum())} px) es un fragmento de la "
                f"persona ({person_px} px) → persona medida menos la cabeza",
                flush=True,
            )
        else:
            full_body = False
            zone = subject.copy()
            for face in inside:
                head = _head_ellipse(face, zone.shape)
                if head is not None:
                    zone &= ~_head_with_hair(head, subject)
            zone_label = "persona medida menos la cabeza"
            print(
                "[ImgEditor] Sin selección: exposición completa medida pero "
                "no se detectó ropa; la zona es la persona medida menos la "
                "cabeza",
                flush=True,
            )
    if full_body:
        zone = subject.copy()
        if manager.measured_full_body_exposure(instruction):
            zone_label = "persona medida completa"
        else:
            for face in inside:
                head = _head_ellipse(face, zone.shape)
                if head is not None:
                    zone &= ~_head_with_hair(head, subject)
            zone_label = "persona medida menos la cabeza"
        if int(zone.sum()) < 200:
            zone = None
    elif zone is None:
        # v5.1251 — sin decisión previa (no se midió exposición completa) la
        # zona es la ropa medida. Antes esta rama se ejecutaba SIEMPRE y
        # borraba la zona ya decidida arriba: con la ropa no localizada
        # (`clothing=None`) la ruta devolvía la foto idéntica con el mensaje
        # «la zona medida quedó vacía».
        zone = clothing
        zone_label = "zona de ropa medida"
    if zone is None:
        return kept("la zona medida quedó vacía; se conserva la foto")

    print(
        f"[ImgEditor] Sin selección: «{query}» localizada "
        f"({int(subject.sum())} px) → {zone_label} ({int(zone.sum())} px)",
        flush=True,
    )
    if cancel_check and cancel_check():
        return None, "Cancelado", None
    from ui.tabs.comfy_launcher import is_comfyui_running, start
    if not is_comfyui_running():
        ok, note, _ = start(directly_run=True)
        if not ok:
            return kept(f"ComfyUI no disponible: {note}")
    manager._release_local_gpu_before_comfy()
    from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
    client = get_flux_edit_comfy_client()
    _km = get_kontext_model_config()
    regime = manager.measured_body_regime(raw_prompt, instruction, engine=engine)
    # v5.1252 — el cambio corporal medido sobre la ZONA DE ROPA va al motor con
    # visión. Medido en la foto del usuario (9 renders, semilla 424242):
    # Kontext responde a esa zona con otra prenda (bañador) con denoise
    # 0,80-1,00, guidance 3,5/4,6, estado afirmativo y hasta a doble resolución;
    # el motor con visión la quita y conserva la pose. El resto de casos
    # (cambio de una prenda, zona de cuerpo entero) sigue con Kontext, que es
    # lo medido hasta ahora para ellos.
    # Paso A (2026-10-26): respetar engine=kontext explícito del usuario.
    # La excepción vision_zone (LongCat) solo aplica cuando el usuario NO
    # ha pedido kontext explícito. Si pide kontext, usar Kontext con
    # denoise 0.80-0.90 según _kontext_body_regime.
    vision_zone = (
        zone_label == "zona de ropa medida"
        and bool(regime.get("body"))
        and str(regime.get("direction", "")).lower() == "uncover"
        # Paso A (2026-10-26): el usuario elegía Kontext y la excepción se
        # respetaba. A partir de 2026-10-27 la decisión la toma el prompt, no
        # el desplegable: si pide quitar ropa sobre una zona de ropa medida, se
        # usa el motor con visión aunque el desplegable diga Kontext.
    )
    if vision_zone:
        ok, note = client.load(flux_version=VISION_BODY_MODEL,
                               allow_vision_engine=True)
        if not ok:
            return kept(f"el motor con visión no está disponible: {note}")
    else:
        ok, note = client.load(flux_version=_km["model_file"])
        if not ok:
            return kept(f"Kontext no disponible: {note}")
    if cancel_check and cancel_check():
        return None, "Cancelado", None
    body_prompt = manager._compose_generation_prompt(
        instruction, engine="kontext",
        magnitude=max(0.75, float(regime["magnitude"])),
        body_transform=True,
        body_direction=regime["direction"],
        body_intensity=regime["intensity"],
    )
    if regime["enforce_clause"] and regime["enforce_clause"] not in body_prompt:
        body_prompt += regime["enforce_clause"]
    measured_negative = negative_prompt or regime["negative_prompt"]
    if measured_negative and not vision_zone:
        # Con CFG=1 (motor con visión y Kontext) el negative no tiene efecto
        # propio: en Kontext el «avoid» viaja en el texto positivo mediado.
        body_prompt += f" Avoid: {measured_negative}."
    print(
        "[ImgEditor] Sin selección: cambio corporal medido → "
        f"{'motor con visión (LongCat)' if vision_zone else 'Kontext img2img puro'}, "
        f"zona={zone_label}, cuerpo={regime.get('body')}, dirección={regime.get('direction')}, "
        f"motor del desplegable={engine} (el resto de la foto se conserva)",
        flush=True,
    )
    import cv2
    effective = Image.fromarray(zone.astype(np.uint8) * 255, "L")
    radius = max(2, min(6, int(min(image.size) * 0.006)))
    feathered = cv2.GaussianBlur(zone.astype(np.float32) * 255.0, (radius * 2 + 1, radius * 2 + 1), 0)
    feathered[~zone] = 0.0
    effective_soft = Image.fromarray(feathered.astype(np.uint8), "L")
    if vision_zone:
        # El recorte de la persona le da al muestreador la resolución que el
        # marco completo le niega (ver `_vision_zone_pass`). La escritura
        # sigue siendo solo la zona medida: el exterior no se toca.
        settings = selection.load_settings()["edit"]
        vision_recipe = {
            "steps": 12,
            "guidance": 1.0,
            "denoise": 1.0,
            "no_kontext": True,
        }
        result, note = _vision_zone_pass(
            manager, client, image, zone, subject, body_prompt,
            measured_negative, seed, settings, progress_callback, cancel_check,
            recipe=vision_recipe, label="motor con visión (LongCat)",
        )
        if result is None:
            return kept(note)
    else:
        # v5.1251 — SIN ancla: la receta medida para «quitar» es img2img puro
        # sobre el latente de la foto con el ruido acotado a la zona medida.
        # Esta llamada no pasaba `kontext_img2img_denoise`, así que el cliente
        # mantenía el ReferenceLatent (`6r`/`6m`) y el ancla con tokens limpios
        # devolvía la foto vestida (log 12:00, edit_1793012554.png: ropa
        # intacta y 0 px cambiados en la zona). El régimen lo mide el manager,
        # igual que la ruta de selección.
        _zone_denoise = float(regime["engine"].get("img2img_denoise", 0.90))
        if not math.isfinite(_zone_denoise) or _zone_denoise <= 0:
            _zone_denoise = 0.90
        _zone_denoise = max(0.80, min(0.90, _zone_denoise))
        print(
            f"[ImgEditor] Sin selección: img2img puro sin ReferenceLatent, "
            f"denoise={_zone_denoise:.2f} en la {zone_label}",
            flush=True,
        )
        generated, note = client.generate(
            prompt=body_prompt,
            negative_prompt=measured_negative,
            num_inference_steps=24,
            guidance_scale=3.5,
            denoise=0.88,
            kontext_img2img_denoise=_zone_denoise,
            seed=seed,
            progress_callback=progress_callback,
            prompt_authority=True,
            no_kontext=False,
            step_cap=24,
            image=image,
            mask_image=Image.fromarray(zone.astype(np.uint8) * 255, "L"),
            keep_source_pixels=True,
            preserve_frame=True,
            cancel_check=cancel_check,
        )
        if cancel_check and cancel_check():
            return None, "Cancelado", None
        if generated is None or generated.image is None:
            return kept(f"el motor no produjo una imagen: {note}")
        patch = generated.image.convert("RGB").resize(image.size, Image.Resampling.LANCZOS)
        result = Image.composite(patch, image, effective_soft)
    if quality_mode or auto_upscale:
        from roop.img_editor.top_tier_finish import run_top_tier_quality
        result, finish_note, quality_analysis = run_top_tier_quality(
            result, enhance_tier=enhance_tier, use_generative=False,
            preserve_faces=quality_preserve_faces, quality_steps=quality_steps,
            progress_callback=progress_callback, cancel_check=cancel_check,
        )
        manager._last_quality_analysis = quality_analysis
        if finish_note:
            note = f"{note} | {finish_note}"
    if cancel_check and cancel_check():
        return None, "Cancelado", None
    result = result.convert("RGB").resize(image.size, Image.Resampling.LANCZOS)
    result = Image.composite(result, image, effective_soft)
    manager._last_instruction_compliance = {
        "compliant": None, "source": "described_subject",
        "note": f"{zone_label}; el resto de la foto se conserva",
    }
    return result, (
        f"Sin selección: «{query}» | prompt escrito en la {zone_label}; "
        f"el resto de la foto se conserva | {note}"
    ), effective
