#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Modificadores opcionales para GENERAR — 5 selectores, cobertura amplia sin pisarse."""

import re
from typing import Dict, List, Tuple

ModifierOption = Tuple[str, str]

AMATEUR_IMAGE_TYPES = frozenset({"amateur_street", "smartphone"})

# SDXL/Pony works best when the decisive subject/action stays in the first CLIP chunk.
# ComfyUI can chunk longer prompts, but later chunks are weaker for composition/counts.
CLIP_TOKEN_BUDGET = 76
CLIP_SCENE_RESERVE = 40

PHOTOREAL_IMAGE_TYPES = frozenset({"photoreal", "amateur_street", "smartphone"})
USER_SCENE_WEIGHT = 1.6
LONG_SCENE_WORDS = 24

# Máximo: 1 estilo + 2 añadidos más (evita diluir la escena del usuario)
MAX_EXTRA_MODIFIERS = 2

CATEGORY_ORDER: List[str] = [
    "image_type",
    "lighting",
    "skin_finish",
    "framing",
    "color_grade",
]

MODIFIERS: Dict[str, List[ModifierOption]] = {
    "image_type": [
        ("auto", ""),
        ("photoreal", "photorealistic RAW photo, film grain, realistic skin"),
        ("amateur_street", "RAW street photo, gritty urban, candid"),
        ("smartphone", "smartphone RAW photo"),
        ("cinematic", "cinematic film still"),
        ("glamour", "glamour beauty photo"),
    ],
    "lighting": [
        ("auto", ""),
        ("natural", "natural daylight"),
        ("golden_hour", "golden hour light"),
        ("studio", "soft studio light"),
        ("dramatic", "dramatic shadows"),
        ("neon", "neon night light, cinematic realism"),
    ],
    "skin_finish": [
        ("auto", ""),
        ("natural", "natural skin"),
        ("detailed", "detailed skin pores"),
        ("smooth", "smooth beauty skin"),
        ("sweaty", "sweaty skin"),
    ],
    "framing": [
        ("auto", ""),
        ("portrait", "portrait shot"),
        ("full_body", "full body shot"),
        ("close_up", "close-up shot"),
        ("wide", "wide angle shot"),
    ],
    "color_grade": [
        ("auto", ""),
        ("warm", "warm tones"),
        ("cool", "cool tones"),
        ("vibrant", "vibrant colors"),
        ("film", "film grain look"),
    ],
}

UI_LABELS: Dict[str, Dict[str, str]] = {
    "image_type": {
        "auto": "Automático",
        "photoreal": "Fotorrealismo",
        "amateur_street": "Amateur / calle",
        "smartphone": "Foto de móvil",
        "cinematic": "Cinematográfico",
        "glamour": "Glamour / estudio",
    },
    "lighting": {
        "auto": "Automático",
        "natural": "Natural",
        "golden_hour": "Dorada",
        "studio": "Estudio",
        "dramatic": "Dramática",
        "neon": "Neón nocturna",
    },
    "skin_finish": {
        "auto": "Automático",
        "natural": "Natural",
        "detailed": "Hiperdetallada",
        "smooth": "Suave / beauty",
        "sweaty": "Sudor / brillo",
    },
    "framing": {
        "auto": "Automático",
        "portrait": "Retrato",
        "full_body": "Cuerpo completo",
        "close_up": "Primer plano",
        "wide": "Plano amplio",
    },
    "color_grade": {
        "auto": "Automático",
        "warm": "Tonos cálidos",
        "cool": "Tonos fríos",
        "vibrant": "Colores vivos",
        "film": "Película / grano",
    },
}

_INCOMPAT: Dict[str, Dict[str, frozenset]] = {
    "amateur_street": {
        "lighting": frozenset({"studio", "neon"}),
        "skin_finish": frozenset({"detailed", "smooth"}),
        "color_grade": frozenset({"vibrant"}),
    },
    "smartphone": {
        "lighting": frozenset({"studio", "neon"}),
        "skin_finish": frozenset({"detailed", "smooth"}),
        "color_grade": frozenset({"vibrant"}),
    },
    "photoreal": {
        "lighting": frozenset({"neon"}),
        "color_grade": frozenset({"vibrant"}),
    },
    "glamour": {
        "skin_finish": frozenset({"sweaty"}),
        "framing": frozenset({"wide"}),
    },
}


def rewriter_caption_trustworthy(base_en: str, rewritten_en: str) -> bool:
    """True si el caption del rewriter mantiene solapamiento semántico con la traducción base."""
    from roop.img_editor.gen_prompt_config import get_rewriter_config

    base = (base_en or "").strip()
    rewritten = (rewritten_en or "").strip()
    if not base or not rewritten:
        return False
    if len(rewritten) < max(12, len(base) * 0.35):
        return False

    min_ratio = float(get_rewriter_config().get("min_overlap_ratio", 0.28))
    base_words = {w for w in re.findall(r"[a-z]{3,}", base.lower()) if len(w) >= 3}
    rw_words = {w for w in re.findall(r"[a-z]{3,}", rewritten.lower()) if len(w) >= 3}
    if not base_words:
        return True
    overlap = len(base_words & rw_words) / len(base_words)
    return overlap >= min_ratio


def get_dropdown_choices(category: str) -> List[Tuple[str, str]]:
    labels = UI_LABELS.get(category, {})
    return [(labels.get(opt_id, opt_id), opt_id) for opt_id, _ in MODIFIERS[category]]


def get_compatible_dropdown_choices(
    image_type: str,
    category: str,
    current_value: str = "auto",
) -> Tuple[List[Tuple[str, str]], str]:
    """Opciones filtradas según el estilo elegido. Devuelve (choices, valor seguro)."""
    all_choices = get_dropdown_choices(category)
    it = image_type or "auto"

    if it == "auto":
        valid_ids = {oid for _, oid in all_choices}
        safe = current_value if current_value in valid_ids else "auto"
        return all_choices, safe

    filtered = [
        (label, oid) for label, oid in all_choices
        if oid == "auto" or not _is_blocked(it, category, oid)
    ]
    valid_ids = {oid for _, oid in filtered}
    safe = current_value if current_value in valid_ids else "auto"
    return filtered, safe


def get_all_categories() -> List[str]:
    return list(CATEGORY_ORDER)


def _fragment(category: str, option_id: str) -> str:
    for oid, fragment in MODIFIERS.get(category, []):
        if oid == option_id:
            return fragment
    return ""


def _is_blocked(image_type: str, category: str, option_id: str) -> bool:
    rules = _INCOMPAT.get(image_type, {})
    return option_id in rules.get(category, frozenset())


def _resolve_modifier_parts(modifiers: Dict[str, str]) -> Tuple[List[Tuple[str, str]], List[str]]:
    image_type = modifiers.get("image_type", "auto") or "auto"
    result: List[Tuple[str, str]] = []
    capped_labels: List[str] = []

    for cat in CATEGORY_ORDER:
        opt = modifiers.get(cat, "auto") or "auto"
        if opt == "auto" or _is_blocked(image_type, cat, opt):
            continue
        frag = _fragment(cat, opt)
        if frag:
            result.append((cat, frag))

    if len(result) > 1 + MAX_EXTRA_MODIFIERS:
        style_parts = [x for x in result if x[0] == "image_type"]
        other_parts = [x for x in result if x[0] != "image_type"]
        kept_others = other_parts[:MAX_EXTRA_MODIFIERS]
        dropped = other_parts[MAX_EXTRA_MODIFIERS:]
        for cat, _ in dropped:
            opt = modifiers.get(cat, "auto") or "auto"
            capped_labels.append(UI_LABELS.get(cat, {}).get(opt, opt))
        result = style_parts + kept_others

    return result, capped_labels


def _estimate_clip_tokens(text: str) -> int:
    """Aproximación rápida de tokens CLIP (sin tokenizer)."""
    if not text:
        return 0
    return len(re.findall(r"\w+", text)) + text.count(",") + text.count(":")


def get_effective_prefix(
    model_alias: str,
    image_type: str,
    model_configs: Dict,
    has_style_modifier: bool = False,
    is_sdxl: bool = True,
) -> str:
    """Prefijo del modelo desde config.
    Para modelos explicit, inyecta nsfw_tags del config al inicio (cada modelo especifica los suyos).
    Todo definido en model_configs.json — sin hardcodes por alias en código."""
    conf = model_configs.get(model_alias, model_configs.get("default", {}))
    is_explicit = bool(conf.get("explicit", False))
    nsfw_tags = (conf.get("nsfw_tags") or "").strip()

    if not is_sdxl:
        compact = (conf.get("prefix_compact") or "").strip()
        if compact:
            prefix = compact if compact.endswith(", ") else f"{compact}, "
        elif is_explicit and nsfw_tags:
            prefix = nsfw_tags + ", "
        else:
            prefix = ""
        if is_explicit and nsfw_tags and nsfw_tags.lower() not in prefix.lower():
            prefix = f"{nsfw_tags}, {prefix}" if prefix else f"{nsfw_tags}, "
        return prefix

    # --- SDXL ---
    if has_style_modifier and image_type in PHOTOREAL_IMAGE_TYPES:
        compact = conf.get("prefix_compact")
        if compact is not None:
            prefix = compact
        else:
            prefix = conf.get("prefix", "")
    else:
        prefix = conf.get("prefix", "")

    if not prefix.strip():
        if is_explicit:
            from roop.img_editor.gen_prompt_config import get_prompt_extras
            base = (get_prompt_extras().get("explicit_sdxl_prefix_fallback") or "").strip()
            if base and not base.endswith(", "):
                base = f"{base}, " if base.endswith(",") else f"{base}, "
            return (nsfw_tags + ", " + base) if nsfw_tags and base else (nsfw_tags + ", " if nsfw_tags else base)
        return prefix

    # Inyectar nsfw_tags al inicio si explicit y no están ya en el prefix
    if is_explicit and nsfw_tags:
        nsfw_lower = nsfw_tags.lower()
        prefix_lower = prefix.lower()
        # Solo inyectar si ningún tag de nsfw_tags está ya en prefix
        already_present = any(tag.strip().lower() in prefix_lower for tag in nsfw_tags.split(",") if tag.strip())
        if not already_present:
            prefix = nsfw_tags + ", " + prefix

    return prefix


def _split_style_and_extras(
    modifiers: Dict[str, str],
) -> Tuple[str, str, List[str], str]:
    """Separa estilo (image_type) de extras; aplica bloqueos en tiempo de generación."""
    image_type = modifiers.get("image_type", "auto") or "auto"
    parts, _ = _resolve_modifier_parts(modifiers)
    style_frag = ""
    extra_frags: List[str] = []
    for cat, frag in parts:
        if cat == "image_type":
            style_frag = frag
        else:
            extra_frags.append(frag)
    suffix_display = ", ".join(frag for _, frag in parts)
    return style_frag, ", ".join(extra_frags), extra_frags, image_type


def assemble_generation_prompt(
    model_alias: str,
    translated: str,
    modifiers: Dict[str, str],
    model_configs: Dict,
    is_sdxl: bool = True,
) -> Tuple[str, str, str]:
    """Monta prompt: prefijo mínimo → escena usuario (primero) → estilo → extras.
    Recorta por presupuesto CLIP (~77 tokens) sin tocar el texto del usuario."""
    style_frag, extras_joined, extra_list, image_type = _split_style_and_extras(modifiers)
    image_type = resolve_effective_image_type(image_type, model_alias, model_configs)
    has_style = image_type != "auto"
    suffix_display = ", ".join(x for x in [style_frag, extras_joined] if x)

    prefix = get_effective_prefix(
        model_alias, image_type, model_configs,
        has_style_modifier=has_style, is_sdxl=is_sdxl,
    )
    scene = (translated or "").strip().rstrip(",").strip()
    scene_words = len(scene.split()) if scene else 0
    weight = USER_SCENE_WEIGHT if scene_words >= LONG_SCENE_WORDS else 1.35
    if is_sdxl and scene:
        weighted_scene = f"({scene}:{weight})"
    else:
        weighted_scene = scene

    conf = model_configs.get(model_alias, model_configs.get("default", {}))
    photoreal_tail = (conf.get("photoreal_tail") or "").strip()
    if photoreal_tail and not photoreal_tail.endswith(","):
        photoreal_tail = f"{photoreal_tail}, "

    from roop.img_editor.gen_semantic import scene_needs_anatomy_integrity
    from roop.img_editor.gen_prompt_config import get_prompt_extras, get_gen_thresholds

    extras_cfg = get_prompt_extras()
    tail_parts: List[str] = []
    if (
        is_sdxl
        and _model_explicit(model_alias, model_configs)
        and scene_needs_anatomy_integrity(scene)
        and extras_cfg.get("anatomy_integrity_positive")
    ):
        tail_parts.append(extras_cfg["anatomy_integrity_positive"])
    if photoreal_tail and image_type in PHOTOREAL_IMAGE_TYPES:
        tail_parts.append(photoreal_tail.rstrip(", ").strip())
    if style_frag and style_frag.lower() not in prefix.lower():
        tail_parts.append(style_frag)
    if extras_joined:
        tail_parts.extend(extra_list)

    max_tail = int(get_gen_thresholds().get("flux_max_tail_parts", 5))
    if not is_sdxl and image_type in PHOTOREAL_IMAGE_TYPES and len(tail_parts) > max_tail:
        trimmed = [style_frag] if style_frag else []
        for part in tail_parts:
            if part not in trimmed:
                trimmed.append(part)
        tail_parts = trimmed[:max_tail]

    if scene_words >= LONG_SCENE_WORDS and len(tail_parts) > 3:
        keep = []
        if style_frag:
            keep.append(style_frag)
        for part in tail_parts:
            if part not in keep:
                keep.append(part)
        tail_parts = keep[:8]

    body_parts: List[str] = []
    if weighted_scene:
        body_parts.append(weighted_scene)
    body_parts.extend(tail_parts)

    body = ", ".join(p for p in body_parts if p)
    final = f"{prefix}{body}" if prefix else body

    est = _estimate_clip_tokens(final)
    if is_sdxl and est > CLIP_TOKEN_BUDGET:
        core_parts = [style_frag] if style_frag else []
        if photoreal_tail:
            for p in photoreal_tail.rstrip(", ").split(","):
                p = p.strip()
                if p and p not in core_parts:
                    core_parts.append(p)
                if len(core_parts) >= 6:
                    break
        core_tail = ", " + ", ".join(core_parts[:6]) if core_parts else ""
        final = f"{prefix}{weighted_scene}{core_tail}" if prefix else f"{weighted_scene}{core_tail}"
        est = _estimate_clip_tokens(final)

    if est > CLIP_TOKEN_BUDGET - 5:
        print(f"[GenFlux] [Warn] Prompt ~{est} tokens (CLIP ~{CLIP_TOKEN_BUDGET}); cola UI recortada")

    return final, suffix_display, image_type


def _model_explicit(model_alias: str, model_configs: Dict) -> bool:
    conf = (model_configs or {}).get(model_alias, {})
    return bool(conf.get("explicit", False))


def resolve_effective_image_type(
    image_type: str,
    model_alias: str,
    model_configs: Dict,
) -> str:
    """Solo lo que el usuario elige en UI — sin forzar estilo por modelo."""
    return (image_type or "auto").strip() or "auto"


def _dedupe_comma_list(text: str) -> str:
    seen = set()
    out: List[str] = []
    for part in (text or "").split(","):
        token = part.strip()
        if not token:
            continue
        key = token.lower()
        if key not in seen:
            seen.add(key)
            out.append(token)
    return ", ".join(out)


def _merge_negative_tokens(neg: str, extra: str) -> str:
    combined = f"{neg}, {extra}" if neg else extra
    return _dedupe_comma_list(combined)


def _prepend_negative_tokens(neg: str, priority: str) -> str:
    """Put structural blockers first so SDXL sees them in the first CLIP chunk."""
    return _dedupe_comma_list(f"{priority}, {neg}" if neg else priority)


def get_effective_negative(
    model_alias: str,
    base_negative: str,
    modifiers: Dict[str, str] = None,
    model_configs: Dict = None,
    prompt_en: str = "",
    is_sdxl: bool = True,
) -> str:
    """Refuerza negativo según config del modelo + reglas generales (anti-censor, anti-stylize para photoreal).
    Los términos específicos de cada modelo (scores, anti-illustrated fuertes, etc.) van en su negative_prompt del JSON."""

    neg = _dedupe_comma_list((base_negative or "").strip())
    mods = modifiers or {}
    image_type = resolve_effective_image_type(
        mods.get("image_type", "auto") or "auto",
        model_alias,
        model_configs or {},
    )
    conf = (model_configs or {}).get(model_alias, {})
    is_explicit = bool(conf.get("explicit", False))

    from roop.img_editor.gen_semantic import scene_needs_anatomy_integrity, is_multi_person_scene
    from roop.img_editor.gen_prompt_config import get_prompt_extras

    extras_cfg = get_prompt_extras()
    needs_anatomy = scene_needs_anatomy_integrity(prompt_en)
    multi_person = is_multi_person_scene(prompt_en)

    if is_sdxl and is_explicit and (multi_person or needs_anatomy):
        block = extras_cfg.get("multi_subject_negative", "")
        if block:
            neg = _prepend_negative_tokens(neg, block)

    if is_sdxl and is_explicit and needs_anatomy:
        block = extras_cfg.get("anti_dismembered_negative", "")
        if block:
            neg = _prepend_negative_tokens(neg, block)

    if is_explicit:
        block = extras_cfg.get("anti_censor_negative", "")
        if block:
            neg = _prepend_negative_tokens(neg, block)

    if image_type in PHOTOREAL_IMAGE_TYPES:
        block = extras_cfg.get("anti_stylize_negative", "")
        if block:
            neg = _merge_negative_tokens(neg, block)

    return neg


def compose_generation_parts(base: str, **modifiers: str) -> Tuple[str, str, str]:
    image_type = modifiers.get("image_type", "auto") or "auto"
    user_base = (base or "").strip()
    _, suffix_display, _, _ = _split_style_and_extras(modifiers)
    return user_base, suffix_display, image_type


def compose_enhanced_prompt(base: str, **modifiers: str) -> str:
    user_base, suffix, _ = compose_generation_parts(base, **modifiers)
    if not user_base:
        return user_base
    return f"{user_base}, {suffix}" if suffix else user_base


def _skipped_modifier_labels(modifiers: Dict[str, str]) -> Tuple[List[str], List[str]]:
    image_type = modifiers.get("image_type", "auto") or "auto"
    parts, capped = _resolve_modifier_parts(modifiers)
    applied = {cat for cat, _ in parts}
    blocked = []
    for cat in CATEGORY_ORDER:
        opt = modifiers.get(cat, "auto") or "auto"
        if opt != "auto" and cat not in applied and _is_blocked(image_type, cat, opt):
            blocked.append(UI_LABELS.get(cat, {}).get(opt, opt))
    return blocked, capped


def preview_modifiers(
    image_type: str = "auto",
    lighting: str = "auto",
    skin_finish: str = "auto",
    framing: str = "auto",
    color_grade: str = "auto",
) -> str:
    modifiers = {
        "image_type": image_type,
        "lighting": lighting,
        "skin_finish": skin_finish,
        "framing": framing,
        "color_grade": color_grade,
    }
    parts, capped = _resolve_modifier_parts(modifiers)
    active = [
        UI_LABELS[cat].get(modifiers[cat], modifiers[cat])
        for cat, _ in parts
    ]
    blocked, capped = _skipped_modifier_labels(modifiers)

    if not active and not blocked and not capped:
        return ""

    html = ""
    if active:
        tags = "".join(
            f"<span style='display:inline-block;background:#1e293b;color:#94a3b8;"
            f"padding:2px 8px;border-radius:10px;font-size:11px;margin:2px 4px 0 0;"
            f"border:1px solid #334155;'>{t}</span>"
            for t in active
        )
        html += f"<div style='margin-top:6px;font-size:11px;color:#64748b;'>Se añadirá: {tags}</div>"

    if blocked:
        stags = "".join(
            f"<span style='display:inline-block;background:#1e293b;color:#f59e0b;"
            f"padding:2px 8px;border-radius:10px;font-size:11px;margin:2px 4px 0 0;"
            f"border:1px solid #92400e;'>{t}</span>"
            for t in blocked
        )
        html += (
            f"<div style='margin-top:4px;font-size:11px;color:#92400e;'>"
            f"No combina con estilo: {stags}</div>"
        )

    if capped:
        ctags = "".join(
            f"<span style='display:inline-block;background:#1e293b;color:#64748b;"
            f"padding:2px 8px;border-radius:10px;font-size:11px;margin:2px 4px 0 0;"
            f"border:1px solid #475569;'>{t}</span>"
            for t in capped
        )
        html += (
            f"<div style='margin-top:4px;font-size:11px;color:#64748b;'>"
            f"Omitido (máx. {MAX_EXTRA_MODIFIERS} extras): {ctags}</div>"
        )
    return html
