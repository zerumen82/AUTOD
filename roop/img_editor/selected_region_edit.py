"""Explicit measured selection: one edit pass, restricted composition.

Called before textual routing so neither pose, body nor inferred quality can
override the region chosen by the user. An invalid choice never falls back to
a global edit. The original photo and measured support remain authoritative
after optional classical finishing.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from roop.img_editor import element_selection as selection
from roop.img_editor.config.editor_selection import get_config
from roop.img_editor.gen_prompt_config import (
    get_kontext_model_config,
    get_longcat_model_config,
)
from roop.img_editor.edit_execution import allow_model, permit_automatic_model, current_execution
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
        mask, _ = segmenter.segment_clothing(image, threshold=get_config()["instruction_scope"]["body_detection_threshold"])
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
    clothes = _dilate(clothes, 14)
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


def _harmonize_patch(patch, image, box, write_crop, skin, ring_px=28, person_mask=None):
    """Match the patch's skin tone to the ring of original skin around it.

    Measured on the user's photo (2026-10-26, V2/V4): the model delivers a
    skin that is lighter and less warm than the original (L 207 against 178 on
    the shoulder, ΔE 19). A feathered ellipse alone leaves that step. A
    measured Lab shift, bounded to ±40 per channel (was ±30), moves the whole
    patch toward the original without inventing color. The shift is measured
    on skin pixels (reliable reference) but applied to the ENTIRE patch so
    shadows, transitions and non-detected skin areas also match.

    For full-person zones (person_mask provided), use the original person's
    skin tone directly instead of a boundary ring (which would be background).
    """
    import cv2
    if skin is None or not skin.any():
        return patch
    full_write = np.zeros((image.height, image.width), dtype=bool)
    full_write[box[1]:box[3], box[0]:box[2]] = write_crop
    
    # For full-person zones, use person's own skin as reference
    if person_mask is not None and person_mask.any():
        # Use original person's skin pixels as reference
        person_skin = person_mask & skin
        if person_skin.any():
            ring_mean = _lab_mean(image, person_skin)
        else:
            ring = _dilate(full_write, ring_px) & ~full_write & skin
            ring_mean = _lab_mean(image, ring) if ring.any() else None
    else:
        # Standard boundary ring for partial zones
        ring = _dilate(full_write, ring_px) & ~full_write & skin
        ring_mean = _lab_mean(image, ring) if ring.any() else None
    
    if ring_mean is None:
        return patch
    gen = np.asarray(patch.convert("RGB"))
    gen_lab = cv2.cvtColor(gen, cv2.COLOR_RGB2LAB).astype(np.float32)
    gen_skin = _skin_pixels(gen) & write_crop
    # La media del parche solo cuenta la piel que está POR ENCIMA de la
    # referencia: la alfombra que cuela el detector (L 82 frente a
    # referencia 102) tiraba de la media hacia abajo y el shift quedaba
    # corto justo para los pies. Con la puerta, patch_mean mide los pies.
    _above = gen_skin & (gen_lab[:, :, 0] > float(ring_mean[0]) + 5.0)
    patch_mean = _lab_mean(Image.fromarray(gen), _above if _above.any() else gen_skin)
    if patch_mean is None:
        return patch
    # Medido 2026-10-30 («sin zapatos», edit_1793353906): la referencia
    # pedía −54,7 niveles de L (piel generada L 156,4 dentro de la zona
    # frente a L 101,7 de la piel de la foto fuera de ella) y el recorte de
    # ±30 cortaba 25 de ellos: la zona salía +20,4 L sobre el resto de la foto
    # (104,8 contra 84,4), que es el descuadre de color que se ve. El tope
    # sube a ±50, que es el que proyecta la zona sobre el entorno (104,8 −
    # 20 ≈ 85 frente a 84,5) y sigue dejando margen para que un anillo de
    # referencia sin contenido no pueda disparar el tono.
    # Medido 2026-11-13 (log «ir sin sujetador», edición final del usuario):
    # la referencia pedía exactamente −50,0 en L —el tope— y el resultado
    # seguía con el tono equivocado («no era el color»). El recorte a ±50
    # truncaba el shift real, así que la zona quedaba a decenas de niveles del
    # entorno. El tope sube a ±80: sigue acotando un anillo de referencia sin
    # contenido, pero ya no corta un desplazamiento medido de piel real.
    shift = np.clip(ring_mean - patch_mean, -80.0, 80.0)
    if float(np.linalg.norm(shift)) > 1.5:
        print(
            f"[ImgEditor] Armonización de tono Lab shift="
            f"{shift.round(1).tolist()}",
            flush=True,
        )
    # Apply shift ONLY to detected skin pixels in the generated patch, not to
    # the whole patch. When the model leaves fabric in the zone, shifting the
    # entire patch by a skin delta makes the fabric look off-colour and the
    # result looks wrong. Skin pixels are the only ones that need tone matching
    # to the original body; everything else stays as the model generated it.
    #
    # Medido 2026-10-30: aplicarlo sobre la máscara BINARIA del detector
    # moteaba la armonización. |blur(skin) − skin| dentro de la zona daba
    # 0,167, o sea 1 de cada 6 píxeles caía en el filo de la máscara y recibía
    # un trato distinto a su vecino: visión describía la salida como
    # «anomalías granuladas/pixeladas alrededor de tobillos y espinillas» y
    # «un anillo moteado superpuesto» sobre el mismo pie que el motor
    # devolvía liso. El peso pasa a la versión suavizada de la máscara, así la
    # transición es un gradiente continuo en vez de salpicadura por píxel.
    gen_lab = cv2.cvtColor(gen, cv2.COLOR_RGB2LAB).astype(np.float32)
    weight = cv2.GaussianBlur(gen_skin.astype(np.float32), (0, 0), 2.0)
    weight = np.clip(weight, 0.0, 1.0) * write_crop.astype(np.float32)
    # Medido 2026-10-30 (edit_1793355228): el 63 % de la alfombra que hay
    # dentro de la zona pasaba el detector de piel, así que la alfombra
    # recibía el mismo oscurecimiento que los pies: L 61,3 dentro de la zona
    # frente a L 72,4 de la MISMA alfombra fuera, 11,1 niveles de diferencia.
    # Esa es la mancha circular translúcida bajo los pies que ve visión.
    #
    # Un shift negativo solo puede bajar lo que ya está POR ENCIMA de la
    # referencia; lo que está por debajo no tiene dónde ir, y bajándolo se
    # separa del mismo material de alrededor. La puerta mide cuánto por encima
    # de la referencia de piel está cada píxel y entra en rampa sobre 20
    # niveles, así no se abre un filo en la línea de corte. La piel del pie
    # (L 144-154 frente a referencia 102) entra al 100 %; la alfombra (L 82)
    # queda intacta.
    gate = np.clip((gen_lab[:, :, 0] - float(ring_mean[0])) / 20.0, 0.0, 1.0)
    weight = weight * gate
    gen_lab = np.clip(
        gen_lab + weight[..., None] * shift.reshape(1, 1, 3), 0, 255,
    )
    # Fondo neutro: el modelo regenera también el fondo y lo deja con otro
    # tono (medido 2026-10-30, edit_1793355228: la alfombra generada sale a
    # L 72,1 dentro de la zona frente a L 54,6 de la MISMA alfombra fuera,
    # +17,5 niveles que se ven como disco translúcido bajo los pies). El
    # mismo principio que la piel: cada material se acerca a su anillo. Solo
    # píxeles neutros (saturación Lab < 25: alfombra, pared, sombra gris) y
    # no-piel, con peso complementario al de piel para no tocar ni pies ni
    # ropa saturada. Referencia: fondo neutro del anillo exterior. (Ojo: el
    # Lab de OpenCV en uint8 trae a/b desplazados +128; hay que restarlo.)
    _sat = np.linalg.norm(gen_lab[:, :, 1:3] - 128.0, axis=2)
    # Ojo: ~gen_skin NO sirve aquí: el detector mete alfombra en gen_skin
    # (55 % de la zona) y esos píxeles quedarían sin rama que los ajuste.
    # Lo que excluye es el peso de piel REALMENTE aplicado (ya con puerta:
    # la alfombra mal clasificada tiene peso 0 y entra; el pie peso ~1 y no).
    _skin_w = np.clip(weight, 0.0, 1.0)
    _bg_like = write_crop & (_sat < 25.0) & (_skin_w < 0.5)
    if _bg_like.any():
        _bg_med = np.median(gen_lab[_bg_like], axis=0)
        _ring_bg = _dilate(full_write, ring_px) & ~full_write
        _ring_bg = _ring_bg & ~skin
        _ring_lab = np.asarray(image.convert("RGB"))
        _ring_lab = cv2.cvtColor(_ring_lab, cv2.COLOR_RGB2LAB).astype(np.float32)
        _ring_sat = np.linalg.norm(_ring_lab[:, :, 1:3] - 128.0, axis=2)
        _ring_bg = _ring_bg & (_ring_sat < 25.0)
        if _ring_bg.any():
            _bg_ref = _ring_lab[_ring_bg].mean(axis=0)
            _bg_shift = np.clip(_bg_ref - _bg_med, -80.0, 80.0)
            if float(np.linalg.norm(_bg_shift)) > 1.5:
                print(
                    f"[ImgEditor] Armonización de fondo Lab shift="
                    f"{_bg_shift.round(1).tolist()}",
                    flush=True,
                )
            _d = np.linalg.norm(gen_lab - _bg_med.reshape(1, 1, 3), axis=2)
            _bg_w = np.clip(1.0 - _d / 30.0, 0.0, 1.0)
            _bg_w = _bg_w * (1.0 - _skin_w)
            _bg_w = _bg_w * _bg_like.astype(np.float32)
            gen_lab = np.clip(
                gen_lab + _bg_w[..., None] * _bg_shift.reshape(1, 1, 3), 0, 255,
            )
    harmonized = Image.fromarray(cv2.cvtColor(gen_lab.astype(np.uint8), cv2.COLOR_LAB2RGB))
    # Smooth the boundary transition to avoid visible seam
    return _smooth_boundary_transition(harmonized, image, box, write_crop, blend_px=12)


def _smooth_boundary_transition(patch, image, box, write_crop, blend_px=8):
    """Blend patch edges into original image over a feathered boundary.

    The Poisson blend in the caller only works inside the write zone.
    This adds an explicit cross-boundary feather so the tone shift doesn't
    create a hard edge at the mask boundary.
    """
    import cv2
    full_write = np.zeros((image.height, image.width), dtype=bool)
    full_write[box[1]:box[3], box[0]:box[2]] = write_crop
    if not full_write.any():
        return patch
    # La rampa va del parche (1, dentro de la zona) al original (0, fuera) y se
    # mide como distancia al EXTERIOR de la zona: 0 en el borde y crece al
    # salir. Así el interior nunca baja de 1 y la foto original no se cuela
    # dentro de la zona.
    #
    # Medido 2026-10-30 («sin zapatos», edit_1793345379): la versión anterior
    # construía la banda como `dilatada & ~erodida` y medía la distancia SOBRE
    # esa banda. En el aro interior de 1 px eso da 0,95 px, luego alpha =
    # 0,95/12 = 0,08, o sea que el aro se rellenaba con un 92 % de la FOTO
    # ORIGINAL. El perfil quedaba 100 % → 96 % → 50 % → 68 % → 100 % de
    # original: un pinchazo no monótono justo en el contorno. Los 122 píxeles
    # con desviación > 60 de su mediano 5×5 estaban TODOS a 0 px del borde de
    # la zona (perímetro 346 px) y la salida era irreconocible frente al
    # recorte que sí salió limpio del motor (0 píxeles impulsivos): era el
    # contorno del zapato original dibujado alrededor del pie generado.
    dist_out = cv2.distanceTransform((~full_write).astype(np.uint8),
                                     cv2.DIST_L2, 3)
    alpha_full = np.clip(1.0 - dist_out / float(blend_px), 0.0, 1.0)
    alpha_full[full_write] = 1.0
    patch_arr = np.asarray(patch.convert("RGB")).copy()
    orig_arr = np.asarray(image.crop(box).convert("RGB"))
    if orig_arr.shape != patch_arr.shape:
        return patch
    alpha = alpha_full[box[1]:box[3], box[0]:box[2]][..., None]
    blended = (patch_arr * alpha + orig_arr * (1.0 - alpha)).astype(np.uint8)
    return Image.fromarray(blended)


# Marcador estable: el caller no debe aceptar el resultado como cumplido.
TARGET_RESIDUAL_FAIL = "el motor no lo quitó"


def _measured_residual(manager, image, zone, query):
    """Píxeles del objetivo que siguen medidos DENTRO de la zona medida.

    Se mide con el mismo segmentador y la misma consulta que decidieron la
    zona, así que la comparación antes/después es la misma medida, a la
    misma resolución (352 px del masker). Devuelve None si no se pudo medir:
    sin medida no se inventa un veredicto.
    """
    if not query or zone is None:
        return None
    try:
        masker = manager._get_clipseg_masker()
        mask = masker.generate_mask(image, query)
    except Exception:
        return None
    if mask is None:
        return None
    try:
        arr = np.asarray(
            mask.convert("L").resize(image.size, Image.Resampling.NEAREST)
        ) > 127
    except Exception:
        return None
    if arr.shape != np.asarray(zone).shape:
        return None
    return int((arr & np.asarray(zone, dtype=bool)).sum())


def _residual_fraction(value, name, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1
            or (positive and value == 0)):
        raise ValueError(f"edit.{name} debe ser una fracción finita válida")
    return float(value)


def _residual_policy(settings=None):
    """Read measured retry parameters without silently replacing invalid config."""
    settings = selection.load_settings()["edit"] if settings is None else settings
    try:
        threshold = _residual_fraction(
            settings["residual_failure_fraction"], "residual_failure_fraction", positive=True,
        )
        growth = _residual_fraction(
            settings["residual_retry_growth_fraction"], "residual_retry_growth_fraction",
        )
        growth2 = _residual_fraction(
            settings.get(
                "residual_retry_growth_fraction_2",
                settings["residual_retry_growth_fraction"],
            ),
            "residual_retry_growth_fraction_2",
        )
        radius = settings["residual_retry_min_radius_pixels"]
    except KeyError as error:
        raise ValueError(f"Falta edit.{error.args[0]} en editor_selection.json") from error
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError("edit.residual_retry_min_radius_pixels debe ser un entero no negativo")
    return threshold, growth, growth2, radius


def _residual_ratio_failed(residual, zone, threshold=None):
    """True when the measured target still covers too much of the edit zone."""
    if residual is None or zone is None:
        return False
    threshold = (
        _residual_policy()[0] if threshold is None
        else _residual_fraction(threshold, "residual_failure_fraction", positive=True)
    )
    before = max(1, int(np.asarray(zone, dtype=bool).sum()))
    return float(residual) / float(before) >= float(threshold)


def _note_reports_residual_failure(note):
    return bool(note) and TARGET_RESIDUAL_FAIL in str(note)


# v5.1325 — límites de la comprobación de destrucción corporal, medidos en
# `diagnostics/probe_editor_body_destruction_20261110.py`.
BODY_OUTSIDE_MAX_FRACTION = 0.002
BODY_FLAT_STD_BEFORE = 4.0
BODY_FLAT_STD_AFTER = 1.5
BODY_SKIN_REWRITE_SHARE = 0.15
BODY_SKIN_BLOCK_SHARE = 0.05
BODY_SKIN_MIN_FRACTION = 0.002


def _body_destruction_metrics(original, edited, zone) -> dict:
    """Measure whether a pass destroyed the body instead of editing it.

    v5.1325 — el 2026-11-10 el caso «la mujer debe ir descalza» entregó las
    piernas convertidas en bloques planos y aun así pasó como
    «Integridad visual OK»: la única comprobación miraba caras e identidad.
    Aquí se mide el síntoma real, sin vocabulario y sin motores nuevos:

    - **Fuera de la zona**: píxeles que cambiaron lejos de lo medido. La
      pasada solo puede escribir su zona; el feather y el Poisson viven en
      un anillo alrededor, que se permite.
    - **Piel reescrita**: piel que tenía textura en la foto y queda plana
      (relleno de un color) tras la pasada. Quitar un zapato o una prenda
      cambia piel por piel o ropa por piel, pero no aplana la piel que ya
      estaba: eso es taparla con un bloque.
    """
    if original is None or edited is None or zone is None:
        return {}
    import cv2

    before = np.asarray(original.convert("RGB"))
    after_img = edited.convert("RGB")
    if after_img.size != original.size:
        after_img = after_img.resize(original.size, Image.Resampling.LANCZOS)
    after = np.asarray(after_img)
    mask = np.asarray(zone, dtype=bool)
    if mask.shape != before.shape[:2]:
        mask = np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255, "L").resize(
                original.size, Image.Resampling.NEAREST)
        ) > 127

    # Mismo criterio a <=1024 px: coste acotado y medidas comparables.
    longest = max(before.shape[:2])
    scale = min(1.0, 1024.0 / float(longest))
    if scale < 1.0:
        size = (max(1, int(round(before.shape[1] * scale))),
                max(1, int(round(before.shape[0] * scale))))
        before_s = cv2.resize(before, size, interpolation=cv2.INTER_AREA)
        after_s = cv2.resize(after, size, interpolation=cv2.INTER_AREA)
        mask_s = cv2.resize(mask.astype(np.uint8), size,
                            interpolation=cv2.INTER_NEAREST) > 0
    else:
        before_s, after_s, mask_s = before, after, mask
    if mask_s.shape != before_s.shape[:2]:
        return {}

    total = float(before_s.shape[0] * before_s.shape[1])
    diff = np.abs(after_s.astype(np.int16) - before_s.astype(np.int16)).max(axis=2)
    radius = max(8, int(round(min(before_s.shape[:2]) * 0.03)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1),
    )
    allowed = cv2.dilate(mask_s.astype(np.uint8), kernel) > 0
    outside = (~allowed) & (diff > 12)

    from roop.img_editor.img_editor_manager import _skin_pixels
    skin = _skin_pixels(before_s) & mask_s
    skin_px = int(skin.sum())
    rewritten_px = 0
    rewritten_largest = 0
    if skin_px >= max(400, int(BODY_SKIN_MIN_FRACTION * total)):
        gray_before = cv2.cvtColor(before_s, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray_after = cv2.cvtColor(after_s, cv2.COLOR_RGB2GRAY).astype(np.float32)

        def _local_std(gray):
            mean = cv2.blur(gray, (9, 9))
            square = cv2.blur(gray * gray, (9, 9))
            return np.sqrt(np.maximum(square - mean * mean, 0.0))

        textured = _local_std(gray_before) >= BODY_FLAT_STD_BEFORE
        flattened = _local_std(gray_after) <= BODY_FLAT_STD_AFTER
        rewritten = skin & textured & flattened & (diff > 12)
        rewritten_px = int(rewritten.sum())
        if rewritten_px:
            # v5.1325 — un bloque tapado es CONTIGUO; un cuerpo regenerado
            # suaviza la piel de forma difusa. Exigir un componente grande
            # evita rechazar un «desnudo» legítimo por pérdida de grano.
            count, _, stats, _ = cv2.connectedComponentsWithStats(
                rewritten.astype(np.uint8), connectivity=8,
            )
            if count > 1:
                rewritten_largest = int(stats[1:, cv2.CC_STAT_AREA].max())

    return {
        "outside_px": int(outside.sum()),
        "outside_fraction": float(outside.sum()) / max(1.0, total),
        "skin_px": skin_px,
        "rewritten_px": rewritten_px,
        "rewritten_fraction": float(rewritten_px) / max(1, skin_px),
        "rewritten_largest": rewritten_largest,
        "rewritten_largest_fraction": float(rewritten_largest) / max(1, skin_px),
    }


def _body_destruction_reason(original, edited, zone) -> str:
    """Empty when the pass kept the body; the measured reason when it did not."""
    metrics = _body_destruction_metrics(original, edited, zone)
    if not metrics:
        return ""
    if metrics["outside_fraction"] > BODY_OUTSIDE_MAX_FRACTION:
        return (
            f"escribió {metrics['outside_px']} px fuera de la zona medida "
            f"({metrics['outside_fraction']:.1%} de la foto)"
        )
    if (metrics["rewritten_fraction"] >= BODY_SKIN_REWRITE_SHARE
            and metrics["rewritten_largest_fraction"] >= BODY_SKIN_BLOCK_SHARE):
        return (
            f"la piel con textura quedó plana en {metrics['rewritten_px']} px "
            f"({metrics['rewritten_fraction']:.0%} de la piel de la zona, "
            f"bloque contiguo de {metrics['rewritten_largest_fraction']:.0%}): "
            "el cuerpo se tapó en vez de editarse"
        )
    return ""


def _expand_zone_within_support(zone, support, grow_frac=None):
    """Grow a measured target inside its support without naming garments.

    The zone the model regenerates is measured, never a word list: the
    clothing segmenter locates the garments of the photo and they are
    intersected with the measured selection. Returns None when the garments
    cannot be located, so the caller falls back to the measured support.

    v5.1336: `grow_frac` es el factor de crecimiento REAL (antes se pasaba
    por dos validaciones con nombres distintos y `growth2` se calculaba sin
    usarse nunca). El radio efectivo es `max(minimum_radius, span*factor)`:
    con una zona pequeña el `minimum_radius` domina y dos factores
    distintos producen la MISMA máscara.
    """
    import cv2

    threshold, growth, growth2, minimum_radius = _residual_policy()
    if grow_frac is not None:
        growth = _residual_fraction(grow_frac, "residual_retry_growth_fraction")
    z = np.asarray(zone, dtype=bool)
    if support is None:
        s = np.ones_like(z, dtype=bool)
    else:
        s = np.asarray(support, dtype=bool)
        if s.shape != z.shape:
            return z
    ys, xs = np.nonzero(z)
    if not xs.size:
        return z
    span = max(int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1))
    radius = max(minimum_radius, int(round(span * growth)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1),
    )
    grown = cv2.dilate(z.astype(np.uint8), kernel) > 0
    return grown & s


def _vision_zone_pass(manager, client, image, zone, support, prompt,
                      negative_prompt, seed, settings, progress_callback,
                      cancel_check, recipe=None, label="motor con visión",
                      analysis=None, faces=None, is_vision_zone=True,
                      target_query="", verify_zone=None,
                      check_clothing_residual=False, reference_image=None,
                      no_hole=False, auto_cover=True, write_mask=None):
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
    if target_query:
        _residual_policy(settings)
    if recipe is None:
        recipe = {"steps": 12, "guidance": 1.0, "denoise": 1.0, "no_kontext": True}
    zone = np.asarray(zone, dtype=bool).copy()
    if support is not None:
        measured_support = np.asarray(support, dtype=bool)
        if measured_support.shape != zone.shape:
            return None, "el soporte medido no corresponde a la zona"
        zone &= measured_support
    region = support if support is not None else zone
    # Save original clothing zone for crop positioning and mask
    original_zone = np.asarray(zone, dtype=bool).copy()
    # Crop from clothing zone (measured size), matching _kontext_zone_pass behavior
    crop_source = original_zone
    # FIX para evitar que la cabeza (con pelo) quede dentro del recorte: usar la
    # zona SIN la cabeza para calcular el box, de forma que el modelo no vea la
    # cabeza en el frame del crop aunque la máscara de generación la excluya.
    # La cabeza puede estar en zone antes de la resta de _head_with_hair en
    # _measured_clothing_zone, así que aplicamos la misma lógica aquí.
    zone_no_head = zone.copy()
    # Siempre intentar eliminar la cabeza, tanto con análisis como sin él.
    # Si hay regiones de cabeza en el análisis, usarlas directamente.
    if analysis is not None:
        for region in analysis.regions:
            if region.kind == "head" and isinstance(region.mask, Image.Image):
                head = np.asarray(region.mask.convert("L")) > 127
                if head.shape == zone.shape:
                    zone_no_head = zone_no_head & ~_head_with_hair(head, support)
    # Sin análisis (ruta "Sin selección"): usar las caras detectadas si hay,
    # si no, detectar caras automáticamente para eliminar la cabeza del zona.
    elif faces is not None and len(faces) > 0:
        for face in faces:
            box = face.bbox
            head_box = np.array([
                max(0, box[0]),
                max(0, box[1]),
                min(zone.shape[1], box[0] + box[2] * 0.9),  # 90% del ancho = cabeza
                min(zone.shape[0], box[1] + box[3] * 0.7)   # 70% del alto = parte superior
            ])
            if head_box[2] > head_box[0] and head_box[3] > head_box[1]:
                head_mask = np.zeros(zone.shape, dtype=bool)
                head_mask[int(head_box[1]):int(head_box[3]),
                          int(head_box[0]):int(head_box[2])] = True
                zone_no_head = zone_no_head & ~head_mask
    else:
        # Sin caras detectadas: intentar detectar caras automáticamente
        # para eliminar la cabeza del zona de edición.
        try:
            from insightface.model_zoo import get_model
            import cv2 as _cv2
            import numpy as _np
            det_model = get_model(
                str(Path(__file__).resolve().parents[3] / "models" / "buffalo_l" / "det_10g.onnx"),
                providers=["CPUExecutionProvider"]
            )
            det_model.prepare(ctx_id=-1, input_size=(1024, 1024), det_thresh=0.5)
            bgr = _cv2.cvtColor(_np.asarray(image.convert("RGB")), _cv2.COLOR_RGB2BGR)
            boxes, _ = det_model.detect(bgr)
            del det_model
            if boxes is not None:
                for box in boxes:
                    x1f, y1f, x2f, y2f = box[:4]
                    head_box = _np.array([
                        max(0, x1f),
                        max(0, y1f),
                        min(zone.shape[1], x1f + (x2f - x1f) * 0.9),
                        min(zone.shape[0], y1f + (y2f - y1f) * 0.7),
                    ])
                    if head_box[2] > head_box[0] and head_box[3] > head_box[1]:
                        head_mask = _np.zeros(zone.shape, dtype=bool)
                        head_mask[int(head_box[1]):int(head_box[3]),
                                  int(head_box[0]):int(head_box[2])] = True
                        zone_no_head = zone_no_head & ~head_mask
        except Exception:
            # Si no se pueden detectar caras, dejar zone_no_head como está (con cabeza)
            pass
    # Usar la zona sin cabeza para la máscara de generación, pero el soporte (persona) para posicionar el crop
    ys, xs = np.nonzero(crop_source)
    if not xs.size:
        ys, xs = np.nonzero(original_zone)
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
    if target_query and reference_image is None and auto_cover:
        # v5.1318 — superficie cubierta automática en un QUITAR: Qwen repite
        # la superficie que ve en la foto, así que se le enseña la zona ya
        # pintada con el color de alrededor (el objeto no está). Medido en el
        # caso real «la mujer debe ir sin zapatos» (2026-11-07,
        # diagnostics/out/shoe_removal_variants_20261107): residuo del 26 %
        # con cubierta (PASS, umbral 40 %) vs 74 % sin ella. Sin cubierta
        # el motor devolvía el zapato pintado; ver también write_zone.
        try:
            covered_full = manager._covered_surface_view(
                image,
                Image.fromarray(np.asarray(zone, dtype=np.uint8) * 255, "L"),
                Image.fromarray(
                    np.asarray(support, dtype=bool).astype(np.uint8) * 255, "L"),
            )
            if covered_full is not None:
                reference_image = covered_full.crop(box)
                print(
                    "[ImgEditor] Superficie cubierta para Qwen: el objeto "
                    f"medido «{target_query}» no se le enseña",
                    flush=True,
                )
        except Exception:
            reference_image = None
    minimum = max(64, int(settings.get("minimum_reference_side", 64)))
    maximum = max(minimum, int(settings.get("maximum_reference_side", 768)))
    crop = image.crop(box)
    scale = maximum / max(1, max(crop.size))
    size = (max(minimum, int(round(crop.width * scale / 16)) * 16),
            max(minimum, int(round(crop.height * scale / 16)) * 16))
    crop_big = crop.resize(size, Image.Resampling.LANCZOS)
    # FIX: la máscara de generación usa zone_no_head (sin cabeza),
    # no la zone original, para que no se edite la cara.
    zone_crop = np.asarray(zone_no_head, dtype=bool)[box[1]:box[3], box[0]:box[2]]
    if not zone_crop.any():
        # Fallback: si la zona sin cabeza está vacía, usar la original
        zone_crop = np.asarray(zone, dtype=bool)[box[1]:box[3], box[0]:box[2]]
    if not zone_crop.any():
        return None, "la zona medida quedó fuera del recorte"
    zone_big = np.asarray(
        Image.fromarray(zone_crop.astype(np.uint8) * 255, "L").resize(
            size, Image.Resampling.NEAREST)
    ) > 127
    if no_hole:
        # v5.1318 — sin agujero: el modelo transforma el recorte entero (un
        # agujero con forma de objeto se rellena con otro objeto) y la
        # escritura sigue limitada a la zona medida, así que el resto de la
        # foto queda píxel a píxel. Misma opción que `_kontext_zone_pass`.
        zone_big = np.ones((size[1], size[0]), dtype=bool)
    # Compute write_zone (same logic as _kontext_zone_pass)
    skin = _skin_pixels(np.asarray(image.convert("RGB")))
    garment = zone & ~skin
    if target_query:
        # v5.1318 — la zona medida ES el objeto que hay que quitar: la guarda
        # de color de piel no puede protegerlo. Medido sobre la foto real del
        # caso «sin zapatos» (2026-11-07, diagnostics/out/
        # shoe_removal_variants_20261107): _skin_pixels clasifica 4248 de los
        # 6575 px de la zona (64,6 %) como piel —suela y flores tienen tono
        # cálido—, write_zone se quedaba en 2327 px de CONTORNO y se escribía
        # el suelo alrededor mientras el calzado quedaba intacto: 0 de 4248 px
        # cambiados en las cuatro variantes de render. Kontext ya escribía la
        # zona entera (write_mask=target_zone); con visión, la misma medida.
        write_zone = zone
    else:
        write_zone = (zone & ~skin) if (skin is not None and skin.any() and int(garment.sum()) >= 400) else zone
    if write_mask is not None:
        # v5.1323 — un reintento con la zona ampliada solo puede ESCRIBIR lo
        # que el segmentador midió al principio. El recorte mayor le da al
        # motor contexto y resolución, pero los píxeles nuevos no son el
        # objeto: escribirlos inventa contenido en la pierna o en el fondo
        # (log 2026-11-10: 18848 → 81345 px y el cuerpo reescrito).
        measured_write = np.asarray(write_mask, dtype=bool)
        if measured_write.shape == write_zone.shape:
            write_zone = write_zone & measured_write
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
        # v5.1318 — superficie cubierta opcional: Qwen repite lo que ve en
        # image1, así que para QUITAR un objeto la vista cubierta (la zona
        # pintada con el color de alrededor) evita que el motor vuelva a
        # pintar el objeto que debe desaparecer. Sin referencia se conserva
        # el comportamiento anterior.
        "reference_image": reference_image,
        "keep_source_pixels": True,
        "preserve_frame": True,
        "cancel_check": cancel_check,
    }
    if recipe.get("kontext_img2img_denoise") is not None:
        generate_options["kontext_img2img_denoise"] = float(
            recipe["kontext_img2img_denoise"])
    # v5.1345d — el `steps` de la receta se ignoraba en Turbo: sin `step_cap`
    # el cliente cae a 12. Medido 2026-11-13 (log «sin ropa»/«desnuda»): la
    # pasada corporal iba a 12 pasos aunque la receta pidiera más, y el modelo
    # no completaba la instrucción. El presupuesto de pasos se forwardea
    # siempre; el cliente lo acota a 4..28.
    try:
        generate_options["step_cap"] = int(recipe.get("steps", 12) or 12)
    except (TypeError, ValueError):
        generate_options["step_cap"] = 12
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
    # For vision zone (clothing), use exposed skin (support & ~zone) as reference
    # For vision zone (clothing), use boundary ring (28px) for harmonization reference
    patch = _harmonize_patch(patch, image, box, write_crop, skin, ring_px=28, person_mask=None)
    soft_zone = Image.fromarray(write_crop.astype(np.uint8) * 255, "L")
    # Larger feathering for vision zone clothing edges
    radius = max(4, min(10, int(min(crop.size) * 0.03))) if is_vision_zone else max(2, min(6, int(min(crop.size) * 0.02)))
    # El feather va hacia FUERA de la zona medida, nunca hacia dentro.
    #
    # El anterior era un blur simétrico con `feathered[~write_crop] = 0`, que
    # descarta justo la mitad que sangra hacia fuera y se queda con la que
    # erosiona hacia dentro: en el borde la FOTO ORIGINAL entraba al 50 % y se
    # desvanecía hacia dentro en ~4 px. En un «quitar» eso resucita lo que hay
    # que quitar. Medido en edit_1793348045 («la mujer debe ir sin zapatos»):
    # LongCat devolvió los pies limpios (0 píxeles impulsivos, imagen natural)
    # y la salida traía un cascarón translúcido despegado de la piel con
    # contorno rojo, huecos en la pantorrilla y las hebillas del zapato
    # original alrededor del pie. El tono de la zona salía 42 niveles por
    # debajo del generado pese a un shift medido de -30: los 12 extra eran la
    # sombra del zapato colada por el feather.
    #
    # La zona medida es el contenido que debe DESAPARECER, así que dentro se
    # escribe la generación al 100 % y la transición con la foto ocurre fuera
    # de ella, donde la foto tiene que seguir intacta. La anchura sigue
    # escalando con el recorte (`radius`), igual que antes.
    d_out = cv2.distanceTransform((~write_crop).astype(np.uint8),
                                  cv2.DIST_L2, 3)
    feathered = np.clip(1.0 - (d_out - 1.0) / float(radius), 0.0, 1.0) * 255.0
    feathered[write_crop] = 255.0
    soft = Image.fromarray(feathered.astype(np.uint8), "L")
    composed = image.copy()
    composed.paste(Image.composite(patch, crop, soft), box)
    # Poisson blend also for vision zone: larger alpha/dilation smooths clothing edges
    try:
        full_write = np.zeros((image.height, image.width), dtype=bool)
        full_write[box[1]:box[3], box[0]:box[2]] = write_crop
        effective = _dilate(full_write, 6)
        alpha = cv2.GaussianBlur(
            effective.astype(np.float32) * 255.0, (31, 31), 0)
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
    # Feathering and Poisson operate on a wider context, but the measured zone
    # is the write contract returned to the caller, including holes/neighbors.
    write_limit = np.zeros_like(original_zone)
    write_limit[box[1]:box[3], box[0]:box[2]] = write_crop
    composed = Image.composite(
        composed, image,
        Image.fromarray((write_limit & original_zone).astype(np.uint8) * 255, "L"),
    )
    del soft_zone
    destroyed = _body_destruction_reason(
        image, composed, write_limit & original_zone,
    )
    if destroyed:
        print(f"[ImgEditor] Resultado rechazado: {destroyed}", flush=True)
        return None, f"resultado rechazado: {destroyed}"
    if target_query:
        # v5.1285 — «no cambia bien ropa» no se veía en ninguna parte: el guard
        # de la ruta descrita solo exige que ALGO haya cambiado, así que un
        # render que conservó las medias de rejilla salió como «Validación
        # visual OK» (log 2026-11-03 13:47 y 13:48). Se vuelve a medir el
        # objetivo sobre el resultado, dentro de la zona medida, y se dice con
        # la cifra; el objetivo no se da por quitado porque el píxel cambió.
        # v5.1311 — residual ≥40 % marca TARGET_RESIDUAL_FAIL: el caller no
        # puede aceptar el resultado como cumplido (log 2026-11-06 zapatos).
        _verify_zone = verify_zone if verify_zone is not None else original_zone
        _verify_zone = np.asarray(_verify_zone, dtype=bool)
        residual = _measured_residual(manager, composed, _verify_zone, target_query)
        if residual is not None:
            before = max(1, int(_verify_zone.sum()))
            ratio = float(residual) / float(before)
            if _residual_ratio_failed(residual, _verify_zone):
                message = (
                    f"el objetivo «{target_query}» sigue medido en la zona "
                    f"({residual} px, {100.0 * ratio:.0f} % de lo medido antes): "
                    f"{TARGET_RESIDUAL_FAIL}"
                )
                print(f"[ImgEditor] Verificación del objetivo: {message}", flush=True)
                note = f"{note} | {message}" if note else message
            else:
                print(
                    f"[ImgEditor] Verificación del objetivo: «{target_query}» "
                    f"reducido a {residual} px ({100.0 * ratio:.0f} % de lo "
                    "medido antes)",
                    flush=True,
                )
    if check_clothing_residual:
        try:
            seg = manager._get_clothing_segmenter()
            if seg is not None and seg.is_available():
                before_mask, _ = seg.segment_clothing(image, threshold=0.45)
                after_mask, _ = seg.segment_clothing(composed, threshold=0.45)
                if before_mask is not None and after_mask is not None:
                    bz = np.asarray(before_mask.convert("L")) > 127
                    az = np.asarray(after_mask.convert("L")) > 127
                    if bz.shape == original_zone.shape and az.shape == original_zone.shape:
                        before_count = int((bz & original_zone).sum())
                        after_count = int((az & original_zone).sum())
                        if before_count > 0 and (float(after_count) / float(before_count)) >= 0.4:
                            message = (
                                f"la ropa sigue medida en la zona "
                                f"({after_count} px, "
                                f"{100.0 * after_count / before_count:.0f} % de lo "
                                f"medido antes): {TARGET_RESIDUAL_FAIL}"
                            )
                            print(f"[ImgEditor] Verificación del objetivo: {message}", flush=True)
                            note = f"{note} | {message}" if note else message
        except Exception:
            pass
    return composed, note


def _kontext_zone_pass(manager, client, image, zone, support, prompt,
                      negative_prompt, seed, settings, progress_callback,
                      cancel_check, denoise=0.90, label="Kontext img2img puro",
                      write_mask=None, no_hole=False):
    """Measured zone with Kontext img2img puro (crop-based like vision engine).

    Same crop logic as _vision_zone_pass but with Kontext img2img puro parameters:
    - kontext_img2img_denoise acotado 0.80-0.90
    - steps=24, guidance=3.5
    - keep_source_pixels=True, preserve_frame=True
    - CFG=1 (FluxGuidance), no negative branch effect

    This gives the model resolution on the measured zone while keeping
    the full-frame anchor for color/lighting consistency.
    """
    original_zone = np.asarray(zone, dtype=bool).copy()
    # Grow zone for generation hole (same as vision pass)
    zone = _dilate(original_zone, 14)
    if support is not None:
        zone &= np.asarray(support, dtype=bool)
    skin = _skin_pixels(np.asarray(image.convert("RGB")))
    garment = zone & ~skin
    write_zone = (zone & ~skin) if (skin is not None and skin.any()
                                    and int(garment.sum()) >= 400) else zone
    if skin is not None and skin.any() and int(garment.sum()) < 400:
        print("[ImgEditor] la prenda es del color de la piel: "
              "no se excluye piel al escribir", flush=True)
    # Crop from original zone (not dilated)
    ys, xs = np.nonzero(original_zone)
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
    if write_mask is not None:
        # El modelo transforma el recorte entero (no un agujero con forma de
        # zapato, que rellenaba con otro zapato) y solo se escribe el objeto
        # medido: el resto del recorte vuelve de la foto.
        write_zone = np.asarray(write_mask, dtype=bool)
        write_crop = write_zone[box[1]:box[3], box[0]:box[2]]
        if not write_crop.any():
            return None, "la zona a escribir quedó fuera del recorte"
    if no_hole:
        zone_big = np.ones((size[1], size[0]), dtype=bool)
    else:
        zone_big = np.asarray(
            Image.fromarray(zone_crop.astype(np.uint8) * 255, "L").resize(
                size, Image.Resampling.NEAREST)
        ) > 127
    print(
        f"[ImgEditor] Sin selección: {label} en el recorte de la "
        f"zona medida {crop.size} → {size}, denoise={denoise:.2f}",
        flush=True,
    )
    manager._release_local_gpu_before_comfy()
    generate_options = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": 24,
        "guidance_scale": 3.5,
        "denoise": 0.88,
        "seed": seed,
        "progress_callback": progress_callback,
        "prompt_authority": True,
        "no_kontext": False,
        "kontext_img2img_denoise": max(0.80, min(0.90, denoise)),
        "step_cap": 24,
        "image": crop_big,
        "mask_image": Image.fromarray(zone_big.astype(np.uint8) * 255, "L"),
        "keep_source_pixels": True,
        "preserve_frame": True,
        "cancel_check": cancel_check,
    }
    generated, note = client.generate(**generate_options)
    if cancel_check and cancel_check():
        return None, "Cancelado"
    if generated is None or generated.image is None:
        return None, f"el motor no produjo una imagen: {note}"
    import cv2
    patch = generated.image.convert("RGB").resize(crop.size, Image.Resampling.LANCZOS)
    if write_mask is None:
        write_crop = write_zone[box[1]:box[3], box[0]:box[2]]
        if not write_crop.any():
            write_crop = zone_crop
    # Harmonize tone and paste back (same as vision pass)
    patch = _harmonize_patch(patch, image, box, write_crop, skin, person_mask=support)
    soft_zone = Image.fromarray(write_crop.astype(np.uint8) * 255, "L")
    radius = max(2, min(6, int(min(crop.size) * 0.02)))
    # Mismo criterio que en la ruta con visión: el feather sale HACIA FUERA
    # de la zona medida. Un blur simétrico recortado con `~write_crop = 0`
    # solo deja la erosión hacia dentro, y eso mezcla la foto original dentro
    # de la zona — en un «quitar» devuelve justo el contenido retirado como
    # un cascarón translúcido con contorno (medido en edit_1793348045, pies
    # limpios en el motor y zapatos originales resucitados en la salida).
    d_out = cv2.distanceTransform((~write_crop).astype(np.uint8),
                                  cv2.DIST_L2, 3)
    feathered = np.clip(1.0 - (d_out - 1.0) / float(radius), 0.0, 1.0) * 255.0
    feathered[write_crop] = 255.0
    soft = Image.fromarray(feathered.astype(np.uint8), "L")
    composed = image.copy()
    composed.paste(Image.composite(patch, crop, soft), box)
    # Poisson boundary blend
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
    written = np.zeros((image.height, image.width), dtype=bool)
    written[box[1]:box[3], box[0]:box[2]] = write_crop
    destroyed = _body_destruction_reason(image, composed, written)
    if destroyed:
        print(f"[ImgEditor] Resultado rechazado: {destroyed}", flush=True)
        return None, f"resultado rechazado: {destroyed}"
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
        face_level = float(heat[y1:y2, x1:x2].mean())
        # La prenda que nombra al sujeto ("mujer de azul") no esta en la
        # cara sino en el cuerpo bajo ella (log 2026-10-30: 4 caras, cara
        # 0,008 y cuerpo 0,10; la cara nunca puntua y la foto se devolvia
        # intacta). Se mide tambien la franja del cuerpo bajo cada cara,
        # con geometria de esa misma cara y sin listas de palabras.
        face_w = max(1, x2 - x1)
        face_h = max(1, y2 - y1)
        bx1 = max(0, x1 - face_w // 2)
        bx2 = min(image.width, x2 + face_w // 2)
        by1 = max(0, y2)
        by2 = min(image.height, y2 + 3 * face_h)
        body_level = 0.0
        if bx2 > bx1 and by2 > by1:
            body_level = float(heat[by1:by2, bx1:bx2].mean())
        scored.append((max(face_level, body_level), face))
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


def _measured_target_zone(manager, image, support, query, faces=(),
                          max_ratio=0.5, threshold=0.35):
    """Measured region of the object a LOCAL removal names.

    The whole-photo CLIPSeg pass runs at 352 px, so on a 6395 px photo a shoe
    is ~10 px and the measured person ends at the ankles: the zone never
    contained the object that had to leave (log 2026-10-29 17:53: «sin
    zapatos», mask above the feet, shoes untouched). The query is measured
    on the person's own halves, where the object has room, and the result is
    still the segmenter's mask, not a vocabulary box.
    """
    if not query:
        return None
    try:
        masker = manager._get_clipseg_masker()
    except Exception:
        return None
    if masker is None:
        return None
    support = np.asarray(support, dtype=bool)
    ys, xs = np.nonzero(support)
    if not xs.size:
        return None
    # El objetivo local que pide quitar (zapato, mochila, brazalete...) puede
    # tocar al cuerpo o ser algo que cuelga/colga fuera del cuerpo (un animal,
    # un objeto de la escena) o estar libre en la foto (un segundo personaje,
    # un elemento de fondo). Medir solo a la persona deja fuera lo que quita
    # pero no lleva: el «quitar» local ya no se limita a lo que la persona
    # lleva, porque cualquier objeto medido sirve de zona.
    try:
        body_zone = _measured_body_zone(manager, image, support, query,
                                       threshold, max_ratio)
    except Exception:
        body_zone = None
    zone = body_zone
    if zone is None:
        # Queda fuera del cuerpo: buscar, sobre todo el lienzo, un objeto que
        # la consulta encienda, como mínimo el tamaño mínimo y sin invadir más
        # de la mitad del volumen de la persona.
        zone = _measured_scene_zone(manager, image, support, query,
                                   threshold, max_ratio)
    if zone is None:
        return None
    for face in faces or ():
        head = _head_ellipse(face, zone.shape)
        if head is not None:
            zone &= ~_head_with_hair(head, support)
    pixels = int(zone.sum())
    person = int(support.sum())
    if pixels < 400 or pixels > max_ratio * max(1, person):
        return None
    print(
        f"[ImgEditor] Objetivo local «{query}» medido en la personne: "
        f"{pixels} px ({100.0 * pixels / max(1, person):.1f} % de elle)",
        flush=True,
    )
    return zone


def _measured_body_zone(manager, image, support, query, threshold=0.35,
                        max_ratio=0.5):
    """Objeto local que toca o cuelga del cuerpo de la persona medida.

    Se mantiene el comportamiento cerrado del punto anterior (silueta
    extendida con margen + barrido por mitades + crecimiento del objeto).

    Los tres datos de los que vive esta función (la caja de la persona, el
    masker y las mitades) se miden AQUÍ: sin ellos la primera línea lanza
    NameError, el caller lo traga con `except Exception` y la ruta cae siempre
    al barrido global de 352 px, que devuelve una zona enorme y fuera del
    objeto (medido 2026-10-29: 381870 px con solo 55 % dentro de la caja del
    calzado, frente a 164 k px / 97 % midiendo en las mitades).
    """
    if not query:
        return None
    try:
        masker = manager._get_clipseg_masker()
    except Exception:
        return None
    if masker is None:
        return None
    support = np.asarray(support, dtype=bool)
    ys, xs = np.nonzero(support)
    if not xs.size:
        return None
    margin = max(8, int(round(0.04 * max(
        int(xs.max()) - int(xs.min()) + 1,
        int(ys.max()) - int(ys.min()) + 1))))
    x1 = max(0, int(xs.min()) - margin)
    y1 = max(0, int(ys.min()) - margin)
    x2 = min(image.width, int(xs.max()) + 1 + margin)
    y2 = min(image.height, int(ys.max()) + 1 + margin)
    half = max(1, (y2 - y1) // 2)
    found = np.zeros(support.shape, dtype=bool)
    for by1, by2 in ((y1, min(y2, y1 + half)), (max(y1, y2 - half), y2)):
        if by2 - by1 < 16:
            continue
        try:
            probability = masker.probability(
                image.crop((x1, by1, x2, by2)), query)
        except Exception:
            continue
        if probability is None:
            continue
        binary = np.asarray(probability) > threshold
        if binary.shape != (by2 - by1, x2 - x1):
            binary = np.asarray(
                Image.fromarray(binary.astype(np.uint8) * 255, "L").resize(
                    (x2 - x1, by2 - by1), Image.Resampling.NEAREST)) > 127
        found[by1:by2, x1:x2] |= binary
    if not found.any():
        return None
    fys, fxs = np.nonzero(found)
    object_side = max(int(fxs.max()) - int(fxs.min()) + 1,
                      int(fys.max()) - int(fys.min()) + 1)
    found = _dilate(found, max(3, int(round(0.06 * object_side))))
    grow = max(8, int(round(0.03 * (int(ys.max()) - int(ys.min()) + 1))))
    zone = found & _dilate(support, grow)
    return zone if int(zone.sum()) >= 400 else None


def _measured_scene_zone(manager, image, support, query, threshold=0.35,
                         max_ratio=0.5):
    """Objeto local que no toca al cuerpo: animales, personas, objetos,
    partes de fondo que la instrucción de quitar/ocultar pide.

    La persona sigue como filtro de pertenencia solo cuando el objeto la
    toca (la misma lógica del punto anterior), pero nunca obliga al
    resultado a estar dentro de la silueta.
    """
    if not query:
        return None
    try:
        from roop.img_editor.img_editor_manager import (
            _visual_request_queries_from_query,
        )
        queries = _visual_request_queries_from_query(query)
    except Exception:
        queries = []
    if not queries:
        queries = [query]
    try:
        masker = manager._get_clipseg_masker()
    except Exception:
        return None
    if masker is None or not hasattr(masker, "generate_mask"):
        return None
    found = np.zeros(image.size[::-1], dtype=bool)
    for q in queries:
        try:
            mask = masker.generate_mask(image, q)
        except Exception:
            continue
        if mask is None:
            continue
        arr = np.asarray(mask.convert("L").resize(image.size,
                       Image.Resampling.NEAREST)) > 127
        if not arr.any():
            continue
        found |= arr
    if not found.any():
        return None
    obj_px = int(found.sum())
    person_px = int(support.sum())
    whole_ratio = obj_px / max(1, person_px)
    # Un objeto que ocupa más de la mitad de la persona es el cuerpo a full
    # (no es local). Disparar al cuerpo entero es traición al change local.
    if whole_ratio >= max_ratio:
        return None
    # Crecer ligeramente el objeto medido, nunca al cuerpo, para no dejar
    # restos (mismo razonamiento anterior).
    fys, fxs = np.nonzero(found)
    object_side = max(int(fxs.max()) - int(fxs.min()) + 1,
                      int(fys.max()) - int(fys.min()) + 1)
    found = _dilate(found, max(3, int(round(0.06 * object_side))))
    ys, xs = np.nonzero(support)
    if xs.size:
        grow = max(8, int(round(0.03 * (int(ys.max()) - int(ys.min()) + 1))))
        near_person = _dilate(support, grow)
    else:
        near_person = np.zeros(image.size[::-1], dtype=bool)
    # El objeto pertenece a la persona si toca su silueta extendida;
    # si no la toca, se considera local dentro de la población de la foto.
    if (found & near_person).any():
        zone = found & near_person
    else:
        zone = found
    return zone if int(zone.sum()) >= 400 else None


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
        if engine not in ("kontext", "longcat"):
            return omitted("la selección visual requiere Kontext o LongCat Turbo")
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
        from roop.img_editor.edit_execution import record_execution_scope
        record_execution_scope(
            f"selección visual protegida ({float(support.mean()):.1%} del fotograma)"
        )
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
        model = (
            get_longcat_model_config() if engine == "longcat"
            else get_kontext_model_config()
        )
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
            or appearance_facts.get("region_ambiguous")
            or appearance_facts.get("structural_add")
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
        vision_body = False
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
                # v5.1247 — para cambios locales de ropa («zapatos», «calcetines»)
                # no se expande a cuerpo completo aunque el fragmento sea pequeño;
                # se mantiene la zona local para que el motor localice el cambio.
                if (clothing is not None and uncover
                        and not manager.measured_full_body_exposure(
                            instruction)):
                    # Zona local de ropa, sin expansión a cuerpo completo
                    full_body = False
                    print(
                        "[ImgEditor] Selección: la ropa medida cubre poco de "
                        "la persona; la zona es la persona medida menos la "
                        "cabeza",
                        flush=True,
                    )
            if full_body:
                true_full_body = manager.is_true_full_body_exposure(instruction)
                # Explicit full-body vocabulary ("sin ropa", "naked", etc.) ALWAYS uses full person
                if true_full_body:
                    generation_zone = np.asarray(support, dtype=bool).copy()
                    zone_label = "persona medida completa (con cabeza)"
                    print(
                        "[ImgEditor] Selección: exposición completa EXPLÍCITA "
                        f"→ {zone_label}",
                        flush=True,
                    )
                elif manager.measured_full_body_exposure(instruction):
                    # v5.125x — para exposición completa medida ("sin ropa",
                    # "naked"), probar primero la zona de ropa medida. Si la
                    # ropa cubre suficiente de la persona, regenerar solo la
                    # ropa (cabeza excluida) en vez de todo el cuerpo. Si la
                    # ropa es un fragmento (p. ej. 33 %), regenerar la persona
                    # COMPLETA (con cabeza) para color/escala coherentes.
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
                    else:
                        # Clothing is a fragment or not found: replace entire person
                        # including head for coherent result.
                        generation_zone = np.asarray(support, dtype=bool).copy()
                        zone_label = "persona medida completa (con cabeza)"
                        if clothing is not None:
                            print(
                                "[ImgEditor] Selección: exposición completa "
                                f"medida pero la ropa ({int(clothing.sum())} px) "
                                f"es un fragmento de la persona ({person_px} px) → "
                                "persona medida completa (con cabeza)",
                                flush=True,
                            )
                        else:
                            print(
                                "[ImgEditor] Selección: ropa no medida → "
                                "persona medida completa (con cabeza)",
                                flush=True,
                            )
                else:
                    # Non-full-body: always use person without head to include legs.
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
                engine == "longcat" and regime["body"]
                and generation_zone is not None)
        # v5.1251 — la ruta regional se ejecuta siempre con el Kontext
        # configurado, así que el modelo planificado tiene que existir TAMBIÉN
        # cuando la instrucción no es un cambio corporal. Antes se asignaba
        # solo dentro de la rama corporal y cualquier otra edición con
        # selección moría con «local variable 'planned_model' referenced
        # before assignment» (probe de régimen: 3/11). Igual con `vision_body`:
        # solo existe cuando la instrucción es un cambio corporal medido.
        planned_model = model["model_file"]
        _report = current_execution()
        if _report is not None and getattr(_report, "automatic", False):
            permit_automatic_model(planned_model)
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
        ok, note = client.load(
            flux_version=model["model_file"],
            allow_vision_engine=engine == "longcat",
        )
        if not ok:
            label = "LongCat Turbo" if engine == "longcat" else "Kontext"
            return omitted(f"{label} no disponible: {note}")
        engine_label = "LongCat Turbo" if engine == "longcat" else "Kontext"
        if cancel_check and cancel_check():
            return None, "Cancelado", None
        if vision_body:
            # LongCat sees the measured region and instruction; the masked
            # composition below keeps the rest of the original photo.
            body_prompt = manager._compose_generation_prompt(
                instruction, engine=engine,
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
                "num_inference_steps": model["steps"],
                "step_cap": model["steps"],
                "guidance_scale": model["cfg"],
                "denoise": 1.0,
                "seed": seed,
                "progress_callback": progress_callback,
                "prompt_authority": True,
                "no_kontext": engine == "longcat",
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
                f"[ImgEditor] Selección: cambio corporal medido → {engine_label} con máscara, "
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
            _focal = (
                manager._scale_kontext_focal_reference(reference)
                if engine == "kontext" else reference
            )
            gen_options = {
                "image": _focal,
                "prompt": head_prompt,
                "negative_prompt": measured_negative,
                "num_inference_steps": (
                    model["steps"] if engine == "longcat" or num_inference_steps is None
                    else num_inference_steps
                ),
                "guidance_scale": (
                    model["cfg"] if engine == "longcat" or guidance_scale is None
                    else guidance_scale
                ),
                "seed": seed,
                "progress_callback": progress_callback,
                "reference_latents_method": model.get("reference_latents_method", "index"),
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
                "num_inference_steps": model["steps"] if engine == "longcat" else 24,
                "step_cap": model["steps"] if engine == "longcat" else 24,
                "guidance_scale": model["cfg"] if engine == "longcat" else 3.5,
                "denoise": 1.0 if engine == "longcat" else 0.88,
                # v5.1251 — sin este kwarg el cliente mantiene el
                # ReferenceLatent y el ancla devuelve la postura vieja: el
                # mensaje decía «img2img puro» y el workflow iba anclado.
                "seed": seed,
                "progress_callback": progress_callback,
                "prompt_authority": True,
                "no_kontext": engine == "longcat",
                "keep_source_pixels": True,
                "preserve_frame": False,
                "cancel_check": cancel_check,
            }
            if engine == "kontext":
                gen_options["kontext_img2img_denoise"] = 0.90
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
                "num_inference_steps": (
                    model["steps"] if engine == "longcat" or num_inference_steps is None
                    else num_inference_steps
                ),
                "guidance_scale": (
                    model["cfg"] if engine == "longcat" or guidance_scale is None
                    else guidance_scale
                ),
                "seed": seed,
                "progress_callback": progress_callback,
                "reference_latents_method": model.get("reference_latents_method", "index"),
                "cancel_check": cancel_check,
            }
            if engine == "kontext":
                gen_options["reference_canvas"] = {
                    "max_side": settings["maximum_reference_side"],
                    "multiple": settings["reference_canvas_multiple"],
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
            # v5.1321 — el exterior de una selección conserva píxeles y tamaño:
            # ampliar aquí se pagaba y se tiraba en el resize de abajo.
            selection_steps = dict(quality_steps or {})
            selection_steps["upscale"] = False
            result, finish_note, quality_analysis = run_top_tier_quality(
                result, enhance_tier=enhance_tier, use_generative=False,
                preserve_faces=quality_preserve_faces, quality_steps=selection_steps,
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
    automatic=False, negative_prompt="", seed=None, progress_callback=None, quality_mode=False,
    auto_upscale=False, enhance_tier="hd", quality_preserve_faces=True,
    quality_steps=None, cancel_check=None,
):
    """One vision pass on the person the prompt names, when nobody is selected.

    Returns None when this request is not a body uncover, so the caller keeps
    its own route. A body uncover that cannot be placed on one person keeps
    the photo: the global route was redrawing the whole frame. Auto may use
    this measured route before its general LongCat pass; manual LongCat keeps
    its separate dispatcher.
    """
    automatic_longcat = automatic and engine == "longcat"
    if engine != "kontext" and not automatic_longcat:
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
    from roop.img_editor.gen_prompt_config import get_kontext_model_config, get_longcat_model_config
    _km = (
        get_longcat_model_config() if automatic_longcat
        else get_kontext_model_config()
    )
    _km_file = _km["model_file"]
    if automatic:
        permit_automatic_model(get_longcat_model_config()["model_file"])
        permit_automatic_model(get_kontext_model_config()["model_file"])
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
    # La prenda que nombra al sujeto («mujer de azul») no cubre la cara:
    # la mascara del vestido existe (8-10 % en el log 2026-10-30) pero ningun
    # centro de cara cae dentro y la ruta devolvia la foto intacta. Con varias
    # personas se obedece igual: se asigna la cara mas cercana al centroide
    # del sujeto medido. Sin listas de palabras, solo geometria medida.
    if not located and subject is not None and len(people) >= 2:
        try:
            _ys, _xs = np.nonzero(np.asarray(subject, dtype=bool))
            if _ys.size and _xs.size:
                _cy, _cx = float(_ys.mean()), float(_xs.mean())
                _best, _best_face = None, None
                for _face in people:
                    try:
                        _x1, _y1, _x2, _y2 = [float(v) for v in _face.bbox]
                    except Exception:
                        continue
                    _fx, _fy = (_x1 + _x2) * 0.5, (_y1 + _y2) * 0.5
                    _dist = (_fx - _cx) ** 2 + (_fy - _cy) ** 2
                    if _best is None or _dist < _best:
                        _best, _best_face = _dist, _face
                if _best_face is not None:
                    inside = [_best_face]
                    located = True
                    print(
                        "[ImgEditor] Sin selección: sujeto medido sin cara dentro; "
                        "se usa la cara mas cercana al sujeto para obedecer el prompt",
                        flush=True,
                    )
        except Exception:
            pass
    if not located:
        return kept(
            "no se pudo localizar a una sola persona descrita; "
            "se conserva la foto"
        )

    regime = manager.measured_body_regime(raw_prompt, instruction, engine=engine)
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
    target_zone = None
    target_query = ""
    lower_body_target = False
    initial_target_zone = None
    # v5.1255 (2026-10-29): _requested_removal_target detecta el objeto local
    # a remover («sin zapatos», «sin gafas», …) usando el semantic analyzer
    # local (conceptos, sin hardcode de usuario). Antes la señal uncover
    # lo promovía a exposición completa: prompt «fully nude» y zona de toda
    # la persona, con los zapatos fuera del recorte.
    if (not full_body and zone is None
            and str(regime.get("direction", "")).lower() == "uncover"):
        target_query = manager._requested_removal_target(
            raw_prompt, instruction)
        if target_query:
            target_zone = _measured_target_zone(
                manager, image, subject, target_query, inside)
            if target_zone is not None:
                zone = target_zone
                zone_label = f"zona medida de «{target_query}»"
                initial_target_zone = np.asarray(target_zone, dtype=bool).copy()
                person_ys = np.nonzero(subject)[0]
                if person_ys.size:
                    top, bottom = float(person_ys.min()), float(person_ys.max())
                    _tz_ys = np.nonzero(target_zone)[0]
                    center = float(_tz_ys.mean())
                    _tz_bottom = float(_tz_ys.max())
                    # El centro falla en pose sentada (piernas plegadas: el
                    # zapato queda a media altura de la persona, log 00:17
                    # «sin zapatos» sin frase de pies desnudos -> media orden).
                    # El borde inferior de la zona si llega al bajo de la
                    # persona: geometria medida, sin listas de prendas.
                    lower_body_target = (
                        (center - top) / max(1.0, bottom - top) >= 0.65
                        or (_tz_bottom - top) / max(1.0, bottom - top) >= 0.60)
                print(
                    "[ImgEditor] Sin selección: «%s» localizado y medido en la "
                    "foto → zona local del objetivo%s" % (
                        target_query,
                        " (parte baja de la persona)" if lower_body_target else "",
                    ),
                    flush=True,
                )
    if full_body and manager.measured_full_body_exposure(instruction):
        clothing = _measured_clothing_zone(manager, image, subject, None)
        person_px = int(subject.sum())
        min_ratio = manager._clothing_zone_min_ratio()
        if (clothing is not None
                and int(clothing.sum())
                >= min_ratio * max(1, person_px)):
            zone = clothing
            zone_label = "zona de ropa medida"
            print(
                "[ImgEditor] Sin selección: exposición completa medida pero "
                "la ropa cubre suficiente de la persona → "
                f"zona de ropa medida ({int(clothing.sum())} px)",
                flush=True,
            )
        elif clothing is not None:
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
    if zone is None:
        # v5.1251 — sin decisión previa (no se midió exposición completa) la
        # zona es la ropa medida. Antes esta rama se ejecutaba SIEMPRE y
        # borraba la zona ya decidida arriba: con la ropa no localizada
        # (`clothing=None`) la ruta devolvía la foto idéntica con el mensaje
        # «la zona medida quedó vacía».
        zone = clothing
        zone_label = "zona de ropa medida"
    if zone is None:
        # v5.1257 (2026-10-30): si el usuario pidió explícitamente quitar algo
        # (target_query) pero ni CLIPSeg ni la ropa midieron zona, NO se
        # conserva la foto. Usamos la banda inferior de la persona (pies/pantorrillas)
        # como fallback para que "sin zapatos", "sin medias", etc. SÍ editen.
        # v5.1257 (2026-10-30): si el usuario pidió explícitamente quitar algo
        # (target_query) o no, cuando ni CLIPSeg ni ropa dieron zona,
        # NO conservamos la foto: banda inferior obligatoria para que
        # "sin zapatos", "sin medias", etc. SIEMPRE editen.
        person_ys = np.nonzero(subject)[0]
        if person_ys.size:
            top, bottom = float(person_ys.min()), float(person_ys.max())
            h = bottom - top
            band_top = int(top + 0.65 * h)
            zone = subject.copy()
            zone[:band_top, :] = False
            zone_label = "banda inferior (fallback 'uncover')"
            print(
                "[ImgEditor] Sin selección: fallback a banda inferior "
                f"({int(zone.sum())} px) — editará localmente",
                flush=True,
            )
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
    # For explicit full-body ("persona medida completa (con cabeza)"),
    # Kontext img2img puro works better than vision engine on large zone.
    vision_zone = (
        (zone_label in ("zona de ropa medida", "persona medida menos la cabeza", "banda inferior (fallback 'uncover')")
         or target_zone is not None
         or automatic_longcat)
        and bool(regime.get("body"))
        and str(regime.get("direction", "")).lower() == "uncover"
        # Paso A (2026-10-26): el usuario elegía Kontext y la excepción se
        # respetaba. A partir de 2026-10-27 la decisión la toma el prompt, no
        # el desplegable: si pide quitar ropa sobre una zona de ropa medida, se
        # usa el motor con visión aunque el desplegable diga Kontext.
        # v5.1255 — lo mismo vale para la ZONA OBJETIVO: es la receta ya
        # medida para quitar una prenda medida (Kontext responde con otra
        # prenda), y el soporte es persona ∪ objetivo porque la silueta de
        # 352 px termina por encima de la suela del calzado.
    )
    # El soporte de composición es la persona medida; un objetivo local (los
    # zapatos) puede caer fuera de esa máscara de 352 px, así que se une a ella
    # para que la escritura no se recorte a sí misma. Las dos rutas lo comparten.
    compose_support = subject if target_zone is None else (subject | zone)
    if vision_zone:
        vision_model = get_longcat_model_config()["model_file"]
        if automatic:
            permit_automatic_model(vision_model)
        ok, note = client.load(flux_version=vision_model,
                               allow_vision_engine=True)
        if not ok:
            return kept(f"el motor con visión no está disponible: {note}")
    else:
        if automatic:
            permit_automatic_model(_km["model_file"])
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
    if target_zone is not None:
        # Un objetivo local no se describe con el sujeto entero: el recorte
        # solo enseña la zona y nombrar a «la mujer» hacía que el motor
        # inventara una persona dentro del recorte. El prompt describe el
        # resultado medido en esa zona.
        body_prompt = manager._compose_local_target_prompt(
            instruction, target_query, lower_body=lower_body_target)
    measured_negative = negative_prompt or regime["negative_prompt"]
    if measured_negative and not vision_zone and target_zone is None:
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
    if vision_zone:
        # El recorte de la persona le da al muestreador la resolución que el
        # marco completo le niega (ver `_vision_zone_pass`). La escritura
        # sigue siendo solo la zona medida: el exterior no se toca.
        settings = selection.load_settings()["edit"]
        # denoise=1.0 (full noise within mask): CLIPSeg residual 567px vs 24022px at
        # 0.95 — the 5% preserved latent at 0.95 keeps fabric structure. The "tono
        # de piel oscuro" documented for 1.0 was caused by ±40 harmonization (now ±30).
        _vision_denoise = 1.0
        # v5.1345d — 24 pasos (antes 12) para cambios de cuerpo/ropa: con 12
        # el modelo a veces conservaba la prenda (log 2026-11-13 «sin ropa»,
        # «desnuda», «sin sujetador»). denoise sigue a 1.0 y el cliente acota
        # a 28; el resto del recorrido (pasos configurables) no cambia.
        vision_recipe = {
            "steps": 24,
            "guidance": 3.5,
            "denoise": _vision_denoise,
            "no_kontext": True,
        }
        print(
            f"[ImgEditor] Sin selección: motor con visión (LongCat) denoise={_vision_denoise:.2f}, guidance=3.5",
            flush=True,
        )
        # v5.1345c — la verificación de ropa residual también aplica a la
        # exposición completa. Medido 2026-11-13 (log «sin ropa» y «desnuda»):
        # con zona «persona medida menos la cabeza» _check_clothing quedaba a
        # False y la ruta declaraba «Validación visual OK: cambio real» aunque
        # la ropa siguiera puesta. La referencia es la ropa MEDIDA en la imagen
        # original dentro de la zona pegada; no depende del tipo de prenda.
        _check_clothing = (
            not target_query
            and str(regime.get("direction", "")).lower() == "uncover"
            and zone_label in (
                "zona de ropa medida", "persona medida menos la cabeza",
            )
        )
        result, note = _vision_zone_pass(
            manager, client, image, zone, compose_support, body_prompt,
            measured_negative, seed, settings, progress_callback, cancel_check,
            recipe=vision_recipe, label="motor con visión (LongCat)",
            is_vision_zone=True, target_query=target_query,
            check_clothing_residual=_check_clothing,
        )
        if result is None:
            return kept(note)
        # v5.1311 — residual alto: un reintento con la zona crecida dentro
        # del soporte ya medido (sin listas de prendas). Si sigue fallando,
        # el marcador TARGET_RESIDUAL_FAIL impide «Validación visual OK».
        # v5.1345c — el reintento también cubre la exposición completa medida
        # (antes solo con target_query, así que «sin ropa» no reintentaba).
        if (
            (target_query or _check_clothing)
            and _note_reports_residual_failure(note)
            and not (cancel_check and cancel_check())
        ):
            expanded = _expand_zone_within_support(zone, compose_support)
            grew = int(expanded.sum()) > int(np.asarray(zone, dtype=bool).sum())
            if grew:
                print(
                    "[ImgEditor] Objetivo residual: reintento con zona ampliada "
                    f"dentro del soporte ({int(zone.sum())} → {int(expanded.sum())} px)",
                    flush=True,
                )
                retry_seed = (int(seed) + 1) if seed is not None else None
                result2, note2 = _vision_zone_pass(
                    manager, client, image, expanded, compose_support,
                    body_prompt, measured_negative, retry_seed, settings,
                    progress_callback, cancel_check,
                    recipe=vision_recipe, label="motor con visión (LongCat, residual)",
                    is_vision_zone=True, target_query=target_query,
                    verify_zone=initial_target_zone,
                    write_mask=initial_target_zone,
                )
                if result2 is not None:
                    result, note = result2, note2
                    effective = Image.fromarray(expanded.astype(np.uint8) * 255, "L")
        # Segundo reintento con zona aún más amplia dentro del soporte medido.
        # v5.1336: este reintento se comparaba con `zone` (la zona ORIGINAL),
        # así que se disparaba aunque produjera EXACTAMENTE la misma máscara
        # que el primer reintento: una tercera llamada idéntica a la segunda,
        # gastando una iteración de GPU sin ninguna posibilidad de progreso
        # (medido: 3 llamadas con las máscaras 100 → 600 → 600 px). El radio
        # efectivo lo fija `residual_retry_min_radius_pixels` cuando la zona
        # es pequeña (`max(minimum_radius, span*factor)`), así que dos
        # factores distintos dan la misma zona.
        # Ahora se compara con la zona del PRIMER reintento: si no hay
        # margen real dentro del soporte medido, no se llama (se ahorra la
        # iteración en vez de repetirla). Con zonas grandes el segundo
        # reintento sí aporta (factor `residual_retry_growth_fraction_2`).
        if (
            target_query
            and _note_reports_residual_failure(note)
            and not (cancel_check and cancel_check())
        ):
            _growth_2 = _residual_policy()[2]
            expanded2 = _expand_zone_within_support(
                zone, compose_support, grow_frac=_growth_2,
            )
            grew2 = int(expanded2.sum()) > int(np.asarray(expanded, dtype=bool).sum())
            if grew2:
                print(
                    "[ImgEditor] Objetivo residual: segundo reintento con zona ampliada "
                    f"dentro del soporte ({int(expanded.sum())} → {int(expanded2.sum())} px)",
                    flush=True,
                )
                retry_seed2 = (int(seed) + 2) if seed is not None else None
                result3, note3 = _vision_zone_pass(
                    manager, client, image, expanded2, compose_support,
                    body_prompt, measured_negative, retry_seed2, settings,
                    progress_callback, cancel_check,
                    recipe=vision_recipe, label="motor con visión (LongCat, residual 2)",
                    is_vision_zone=True, target_query=target_query,
                    verify_zone=initial_target_zone,
                    write_mask=initial_target_zone,
                )
                if result3 is not None:
                    result, note = result3, note3
                    effective = Image.fromarray(expanded2.astype(np.uint8) * 255, "L")
    else:
        # v5.1251 — SIN ancla: la receta medida para «quitar» es img2img puro
        # sobre el latente de la foto con el ruido acotado a la zona medida.
        # Crop-based approach (same as vision engine) for resolution on measured zone.
        _zone_denoise = float(regime["engine"].get("img2img_denoise", 0.90))
        if not math.isfinite(_zone_denoise) or _zone_denoise <= 0:
            _zone_denoise = 0.90
        _zone_denoise = max(0.80, min(0.90, _zone_denoise))
        print(
            f"[ImgEditor] Sin selección: Kontext img2img puro (crop-based), "
            f"denoise={_zone_denoise:.2f} en la {zone_label}",
            flush=True,
        )
        settings = selection.load_settings()["edit"]
        result, note = _kontext_zone_pass(
            manager, client, image, zone, compose_support, body_prompt,
            measured_negative, seed, settings, progress_callback, cancel_check,
            denoise=_zone_denoise, label="Kontext img2img puro (crop)",
            write_mask=target_zone, no_hole=target_zone is not None,
        )
        if result is None:
            return kept(note)
    if cancel_check and cancel_check():
        return None, "Cancelado", None
    if quality_mode or auto_upscale:
        # The caller rejects an unchanged/ghost candidate and continues on its
        # existing retry route. Finishing it first wastes a full TOP pass and
        # can disguise the failed edit with unrelated pixel changes. Keep the
        # raw candidate for that caller; successful finishes still receive its
        # final validation, including any changes introduced by the finisher.
        # Residual failure is not a visual pass either: skip TOP so the
        # alternative route sees the raw candidate.
        if _note_reports_residual_failure(note):
            finish_allowed, reason = False, TARGET_RESIDUAL_FAIL
        else:
            finish_allowed, reason = manager._validate_visual_edit(
                image, result, instruction,
            )
        if cancel_check and cancel_check():
            return None, "Cancelado", None
        if finish_allowed:
            from roop.img_editor.top_tier_finish import run_top_tier_quality
            # v5.1321 — esta ruta conserva el resto de la foto: ampliar el
            # lienzo no aporta y costaba el paso más caro del acabado.
            selection_steps = dict(quality_steps or {})
            selection_steps["upscale"] = False
            result, finish_note, quality_analysis = run_top_tier_quality(
                result, enhance_tier=enhance_tier, use_generative=False,
                preserve_faces=quality_preserve_faces, quality_steps=selection_steps,
                progress_callback=progress_callback, cancel_check=cancel_check,
            )
            manager._last_quality_analysis = quality_analysis
            if finish_note:
                note = f"{note} | {finish_note}"
        else:
            print(
                f"[ImgEditor] Acabado TOP omitido: candidato rechazado antes del acabado ({reason})",
                flush=True,
            )
    if cancel_check and cancel_check():
        return None, "Cancelado", None
    manager._last_instruction_compliance = {
        "compliant": None, "source": "described_subject",
        "note": f"{zone_label}; el resto de la foto se conserva",
    }
    return result, (
        f"Sin selección: «{query}» | prompt escrito en la {zone_label}; "
        f"el resto de la foto se conserva | {note}"
    ), effective
