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

# Señales de alucinación frecuentes del rewriter 0.5B en captions txt2img
_REWRITER_HALLUCINATION_MARKERS = frozenset({
    "chisling", "chowing down", "blue skin", "light blue skin",
    "walking on a dusty street", "with legs in",
})

_CONTRADICTORY_HAIR_PAIRS = (
    (("dark hair", "brown hair", "castaño"), ("blonde", "blond", "white hair")),
)

# Fallback si prefix_compact está vacío en modelos explicit (evita prompt solo con escena)
# NOTA: los tags NSFW los lee del config (nsfw_tags) — cada modelo especifica los suyos
EXPLICIT_SDXL_PREFIX = "photorealistic, RAW photo, realistic, film grain, detailed skin texture, sharp focus, natural lighting, realistic anatomy, "

# Prefijo fallback para FLUX explicit
EXPLICIT_FLUX_PREFIX = ""

SDXL_ANTI_STYLIZE_NEGATIVE = (
    "anime, manga, cartoon, comic, illustration, cel shaded, stylized, "
    "digital art, concept art, painting, drawing, sketch, 2d, toon, "
    "3d render, cgi, render, overrendered, oversaturated, "
    "plastic skin, doll, figurine, airbrushed, waxy skin, smooth skin, "
    "flawless skin, video game, flat shading, painterly, artistic"
)

SDXL_MULTI_SUBJECT_NEGATIVE = (
    "solo, alone, single person, 1girl solo, 1boy solo, portrait solo, only one person"
)

SDXL_ANTI_DISMEMBERED_NEGATIVE = (
    "floating penis, disembodied penis, detached penis, penis only, cock only, "
    "genitals only, severed penis, floating cock, disconnected genitals, "
    "penis without body, penis in air, isolated genitals, floating limbs, "
    "disembodied body parts, body parts floating, cropped penis, extra penis, "
    "duplicate penis, headless penis, torso missing, missing body"
)

# Señales generales en prompt EN (no hardcode por frase del usuario)
_ANATOMY_INTEGRITY_ANCHORS = (
    "penis", "cock", "oral", "blowjob", "sucking", "deepthroat", "fellatio",
    "threesome", "two men", "another man", "second man", "mmf", "cuckold",
    "stranger", "man with", "men", "male",
)

SDXL_ANATOMY_INTEGRITY_POSITIVE = (
    "full bodies visible, complete anatomy, penis attached to male body, "
    "all subjects fully visible, connected bodies, proper human proportions"
)

SDXL_ANTI_CENSOR_NEGATIVE = (
    "censored, sfw, clothed, covered, mosaic censoring, bar censor, "
    "blurry crotch, pixelated, cropped, underwear, bra, panties"
)

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
    """True si el caption del rewriter no contradice ni inventa sobre la traducción base."""
    base = (base_en or "").strip()
    rewritten = (rewritten_en or "").strip()
    if not base or not rewritten:
        return False
    if len(rewritten) < max(12, len(base) * 0.35):
        return False

    rw_lower = rewritten.lower()
    if any(marker in rw_lower for marker in _REWRITER_HALLUCINATION_MARKERS):
        return False

    for group_a, group_b in _CONTRADICTORY_HAIR_PAIRS:
        has_a = any(term in rw_lower for term in group_a)
        has_b = any(term in rw_lower for term in group_b)
        if has_a and has_b:
            return False

    base_words = {w for w in re.findall(r"[a-z]{3,}", base.lower()) if len(w) >= 3}
    rw_words = {w for w in re.findall(r"[a-z]{3,}", rw_lower) if len(w) >= 3}
    if not base_words:
        return True
    overlap = len(base_words & rw_words) / len(base_words)
    return overlap >= 0.28


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
        if is_explicit and nsfw_tags:
            return nsfw_tags + ", "
        return ""

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
            base = EXPLICIT_SDXL_PREFIX
            return (nsfw_tags + ", " + base) if nsfw_tags else base
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


def scene_needs_anatomy_integrity(prompt_en: str) -> bool:
    """Escenas con genitales y/o varias personas → exigir anatomía conectada."""
    pl = f" {(prompt_en or '').lower()} "
    hits = sum(1 for tag in _ANATOMY_INTEGRITY_ANCHORS if tag in pl)
    return hits >= 2


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

    tail_parts: List[str] = []
    if is_sdxl and _model_explicit(model_alias, model_configs) and scene_needs_anatomy_integrity(scene):
        tail_parts.append(SDXL_ANATOMY_INTEGRITY_POSITIVE)
    if photoreal_tail:
        tail_parts.append(photoreal_tail.rstrip(", ").strip())
    if style_frag and style_frag.lower() not in prefix.lower():
        tail_parts.append(style_frag)
    if extras_joined:
        tail_parts.extend(extra_list)

    if scene_words >= LONG_SCENE_WORDS and len(tail_parts) > 3:
        # Keep photoreal base + style + all key realism helpers. Prioritize everything photoreal/RAW/gritty for better realism on long rewriter outputs. Keep as many as possible.
        keep = []
        if photoreal_tail:
            pparts = [p.strip() for p in photoreal_tail.rstrip(", ").strip().split(",") if p.strip()]
            keep.extend(pparts)
        if style_frag and style_frag not in keep:
            keep.append(style_frag)
        for extra in tail_parts:
            lower = extra.lower()
            if any(k in lower for k in ["raw", "street", "photo", "film", "grain", "skin", "pore", "anatomy", "gritty", "candid", "natural", "sweaty"]):
                if extra not in keep:
                    keep.append(extra)
        tail_parts = keep[:8]

    body_parts: List[str] = []
    if weighted_scene:
        body_parts.append(weighted_scene)
    body_parts.extend(tail_parts)

    body = ", ".join(p for p in body_parts if p)
    final = f"{prefix}{body}" if prefix else body

    est = _estimate_clip_tokens(final)
    if is_sdxl and est > CLIP_TOKEN_BUDGET:
        # Keep prefix + scene + style + as many photoreal descriptors as fit (general for all photoreal models). Prioritize film grain, pores, anatomy, gritty, raw, candid etc. Never drop realism terms first.
        core_parts = []
        if style_frag:
            core_parts.append(style_frag)
        if photoreal_tail:
            parts = [p.strip() for p in photoreal_tail.rstrip(", ").strip().split(",") if p.strip()]
            for p in parts:
                lower = p.lower()
                if any(k in lower for k in ["raw", "street", "photo", "film", "grain", "skin", "pore", "texture", "anatomy", "proportion", "gritty", "candid", "lighting", "focus", "depth", "urban", "sweaty", "real"]):
                    if p not in core_parts:
                        core_parts.append(p)
            for p in parts:
                if p not in core_parts:
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
) -> str:
    """Refuerza negativo según config del modelo + reglas generales (anti-censor, anti-stylize para photoreal).
    Los términos específicos de cada modelo (scores, anti-illustrated fuertes, etc.) van en su negative_prompt del JSON."""

    neg = _dedupe_comma_list((base_negative or "").strip())
    mods = modifiers or {}
    image_type = mods.get("image_type", "auto") or "auto"
    conf = (model_configs or {}).get(model_alias, {})
    is_explicit = bool(conf.get("explicit", False))

    needs_anatomy = scene_needs_anatomy_integrity(prompt_en)
    if is_explicit and (conf.get("multi_subject", True) or needs_anatomy):
        neg = _prepend_negative_tokens(neg, SDXL_MULTI_SUBJECT_NEGATIVE)

    if is_explicit and needs_anatomy:
        neg = _prepend_negative_tokens(neg, SDXL_ANTI_DISMEMBERED_NEGATIVE)

    if is_explicit:
        neg = _prepend_negative_tokens(neg, SDXL_ANTI_CENSOR_NEGATIVE)

    # Anti-estilizado general para modelos photoreal / explicit SDXL (los términos específicos van en el negative_prompt del config del modelo)
    if image_type in PHOTOREAL_IMAGE_TYPES:
        neg = _merge_negative_tokens(neg, SDXL_ANTI_STYLIZE_NEGATIVE)

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
