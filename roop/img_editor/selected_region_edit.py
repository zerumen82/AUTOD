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

# v5.1229 — el motor CON VISIÓN para un cambio corporal medido.
# Medido en FLUX.1/Kontext: el prompt negativo es inerte (la guía va en
# FluxGuidance y el muestreo va a CFG=1) y el texto positivo no basta: la
# participación de piel en la zona de prenda medida bajó de 0,2675 a 0,0676
# (el modelo viste más). El motor con visión (Qwen-VL ve la foto y el texto
# es la única autoridad) es el único del proyecto medido que sigue una
# instrucción de estado final. La excepción es estrecha: solo la ruta de
# selección la pide, y su exterior sigue siendo píxel a píxel.
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
                    zone = zone & ~head
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
            not regime["body"] and head_selected
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
        # distintos). La ropa medida es la superficie que la instrucción
        # describe, así que se repinta la ropa y la persona sobrevive.
        generation_zone = None
        if regime["body"]:
            generation_zone = _measured_clothing_zone(
                manager, image, support, analysis)
            if generation_zone is not None:
                print(
                    f"[ImgEditor] Selección: cambio corporal medido "
                    f"(dirección={regime['direction']}, "
                    f"intensidad={regime['intensity']:.2f}) → zona de ropa medida "
                    f"({int(generation_zone.sum())} px): la prenda de la persona "
                    f"seleccionada es la superficie de la petición",
                    flush=True,
                )
            else:
                print(
                    f"[ImgEditor] Selección: cambio corporal medido "
                    f"(dirección={regime['direction']}, "
                    f"intensidad={regime['intensity']:.2f}) → zona medida completa",
                    flush=True,
                )
        from ui.tabs.comfy_launcher import is_comfyui_running, start
        if not is_comfyui_running():
            ok, note, _ = start(directly_run=True)
            if not ok:
                return omitted(f"ComfyUI no disponible: {note}")
        manager._release_local_gpu_before_comfy()
        from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
        client = get_flux_edit_comfy_client()
        # v5.1229 — un cambio corporal medido se ejecuta en el motor CON
        # VISIÓN. La ruta de selección conserva el exterior píxel a píxel, así
        # que el modelo solo puede repintar la zona medida: la ropa. El motor
        # con visión (Qwen-VL ve la foto y el texto manda) es el único medido
        # que sigue la instrucción de estado final; en Kontext la zona medida
        # salía con otra prenda encima (piel 0,0676 frente a 0,2675 original).
        # Solo la dirección «quitar» usa ese motor: para «poner» y «cambiar» la
        # receta medida es la anclada de Kontext.
        vision_body = bool(
            regime["body"] and generation_zone is not None
            and str(regime["direction"]).lower() == "uncover")
        if vision_body or pose_change:
            ok, note = client.load(
                flux_version=VISION_BODY_MODEL, allow_vision_engine=True)
            if not ok:
                # The vision engine is the one that follows the text. Without
                # it the route still edits the measured zone with the plain
                # recipe instead of leaving the photo untouched.
                print(
                    f"[ImgEditor] Selección: motor con visión no disponible "
                    f"({note}); se usa Kontext",
                    flush=True,
                )
                vision_body = False
                pose_change = False
                ok, note = client.load(flux_version=model["model_file"])
            engine_label = ("Turbo con visión"
                            if (vision_body or pose_change) else "Kontext")
        else:
            ok, note = client.load(flux_version=model["model_file"])
            engine_label = "Kontext"
        if not ok:
            return omitted(f"{engine_label} no disponible: {note}")
        if cancel_check and cancel_check():
            return None, "Cancelado", None
        if vision_body:
            # v5.1230 — el cambio corporal medido en dirección «quitar» se
            # ejecuta con el motor CON VISIÓN: el modelo ve la foto y el texto
            # es la única autoridad (receta de junio, la única medida que llega
            # al estado final pedido). Decisión del usuario (2026-10-21):
            # cumplimiento total, identidad sacrificada.
            #
            # Dos entregadas, según lo que haya medido en la foto:
            #  * SIN otras personas: se reinterpreta el MARCO (medido: la
            #    persona sale desnuda y sin zapatos). Pegar solo el soporte de
            #    esa foto nueva es un collage roto, así que la entrega es el
            #    marco entero.
            #  * CON otras personas: el marco NO se reinterpreta, porque la
            #    receta redibuja también sus caras (medido el 21/10 en la foto
            #    de dos personas: 85,9 y 66,3 niveles de diferencia, 99 % de
            #    los píxeles de las dos caras). Ahí la zona de ropa medida es
            #    el agujero y la foto de alrededor sobrevive: las otras
            #    personas quedan intactas y el marco también.
            other_people = bool(protected is not None and np.asarray(protected).any())
            body_prompt = manager._compose_generation_prompt(
                instruction, engine="longcat",
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
                "num_inference_steps": 12,
                "guidance_scale": 1.0,
                "denoise": 1.0,
                "seed": seed,
                "progress_callback": progress_callback,
                "prompt_authority": True,
                "no_kontext": True,
                "cancel_check": cancel_check,
            }
            if other_people:
                gen_options.update({
                    "image": reference,
                    "mask_image": Image.fromarray(
                        generation_zone.astype(np.uint8) * 255, "L").crop(box),
                    "keep_source_pixels": True,
                    "preserve_frame": True,
                })
                print(
                    "[ImgEditor] Selección: cambio corporal medido → motor con "
                    "visión con la zona de ropa medida como agujero (hay otras "
                    "personas en la foto: el marco no se reinterpreta)",
                    flush=True,
                )
            else:
                gen_options.update({
                    # La foto completa: el marco que se reinterpreta. Sin
                    # agujero y sin ancla de latentes, el texto manda.
                    "image": image,
                    "mask_image": None,
                    "keep_source_pixels": False,
                    "preserve_frame": False,
                })
                print(
                    "[ImgEditor] Selección: cambio corporal medido → motor con "
                    "visión, marco reinterpretado (el texto manda)",
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
            _focal = reference
            _long = max(reference.size)
            if _long > 0:
                _scale = max(1.0, float(settings["maximum_reference_side"]) / _long)
                if _scale > 1.0:
                    _focal = reference.resize(
                        (max(1, int(round(reference.width * _scale))),
                         max(1, int(round(reference.height * _scale)))),
                        Image.Resampling.LANCZOS)
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
            # v5.1238 — el cambio de pose/ángulo medido: la FOTO COMPLETA es
            # la entrada, sin agujero y sin ancla, con el texto medido como
            # única autoridad (el motor con visión ve la foto y obedece el
            # texto). Un cambio de postura no puede ser un recorte local: el
            # cuerpo se reposiciona entero.
            zone_prompt, measured_negative = _zone_prompt(
                manager, settings, instruction, regime, negative_prompt)
            gen_options = {
                "image": image,
                "prompt": zone_prompt,
                "negative_prompt": measured_negative,
                "num_inference_steps": 12,
                "guidance_scale": 1.0,
                "denoise": 1.0,
                "seed": seed,
                "progress_callback": progress_callback,
                "mask_image": None,
                "keep_source_pixels": False,
                "prompt_authority": True,
                "no_kontext": True,
                "preserve_frame": False,
                "cancel_check": cancel_check,
            }
            print(
                "[ImgEditor] Selección: cambio de postura/ángulo medido → "
                "motor con visión, marco reinterpretado (el texto manda; las "
                "caras de las demás personas se conservan)",
                flush=True,
            )
        else:
            gen_options = {
                "image": reference,
                # v5.1228 — la frase de estado va pegada a la instrucción, antes
                # de la cláusula de contexto: el modelo lee el estado final como
                # parte de lo que tiene que hacer en esa zona.
                "prompt": settings["prompt_template"].format(
                    instruction=instruction + regime["state_suffix"])
                + regime["enforce_clause"],
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
        if (vision_body and not other_people) or pose_change:
            # v5.1230/v5.1238 — la entrega del cambio medido que rehace el
            # MARCO (el corporal sin otras personas, y el de postura/ángulo)
            # es el marco reinterpretado: pegar solo el soporte de una foto
            # nueva es un collage roto (medido). Las caras medidas de las
            # demás personas se conservan de la foto original.
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
            effective = Image.fromarray(support.astype(np.uint8) * 255, "L")
            alpha = np.asarray(effective.filter(ImageFilter.GaussianBlur(radius))).copy()
            alpha[~support] = 0
            alpha = Image.fromarray(alpha, "L")
            # The fallback inside the crop is the original photo, never the
            # blinded reference: the grey patch must never reach the result.
            result = image.copy()
            result.paste(Image.composite(patch, original_crop, alpha.crop(box)), box)
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
            "note": ("Marco reinterpretado (identidad no preservada); "
                     "cumplimiento visual pendiente de inspección")
            if ((vision_body and not other_people) or pose_change) else
            "Región protegida; cumplimiento visual pendiente de inspección",
        }
        labels = ", ".join(region.label for region in analysis.regions if region.id in set(ids))
        if pose_change:
            message = (f"Selección: {labels} | {engine_label}: {note} | "
                       "marco reinterpretado (cambio de postura; las caras "
                       f"de las demás personas se conservan){unmeasured_note}")
        elif vision_body and not other_people:
            message = (f"Selección: {labels} | {engine_label}: {note} | "
                       "marco reinterpretado (identidad no preservada)")
        elif vision_body:
            message = (f"Selección: {labels} | {engine_label}: {note} | "
                       "zona de ropa medida; el marco y las otras personas "
                       "se conservan")
        else:
            message = (f"Selección: {labels} | {engine_label}: {note} | "
                       f"exterior protegido{blind_note}{unmeasured_note}")
        if finish_note:
            message += f" | {finish_note} (tamaño de entrada conservado para proteger el exterior)"
        return result, message, effective
    except Exception as error:
        return omitted(f"no se pudo editar la región ({error})")
