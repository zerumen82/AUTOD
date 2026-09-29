# -*- coding: utf-8 -*-
"""Modo Calidad: una foto nítida en HD ya no sale idéntica; una pequeña crece."""
import socket
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _listening(port: int) -> bool:
    sock = socket.socket()
    sock.settimeout(0.3)
    try:
        return sock.connect_ex(("127.0.0.1", port)) == 0
    finally:
        sock.close()


def _sharp_hd() -> Image.Image:
    yy, xx = np.mgrid[0:1080, 0:1920]
    base = (
        90
        + 50 * np.sin(xx / 6.5)
        + 40 * np.sin(yy / 4.5)
        + 25 * np.sin((xx * 0.7 + yy) / 3.2)
        + 15 * ((xx * 3 + yy * 5) % 7)
    )
    rgb = np.stack([
        np.clip(base, 0, 255),
        np.clip(base * 0.85 + 20, 0, 255),
        np.clip(base * 0.7 + 35 + 10 * np.sin(xx / 40.0), 0, 255),
    ], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb)


def _soft_small() -> Image.Image:
    import cv2
    rng = np.random.default_rng(3)
    noise = rng.integers(40, 180, size=(160, 240, 3), dtype=np.uint8)
    img = Image.fromarray(noise).resize((240, 160), Image.BOX)
    bgr = cv2.GaussianBlur(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), (0, 0), 3.5)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _mean_diff(a: Image.Image, b: Image.Image) -> float:
    aa = np.array(a.convert("RGB"), dtype=np.int16)
    bb = np.array(b.convert("RGB"), dtype=np.int16)
    return float(np.mean(np.abs(aa - bb)))


def main() -> int:
    from roop.img_editor.top_tier_finish import (
        run_quality_enhance_after_edit,
        run_top_tier_quality,
    )

    fails = 0

    def check(name: str, cond: bool, extra: str = "") -> None:
        nonlocal fails
        mark = "OK  " if cond else "FAIL"
        print(f"{mark} {name}" + (f" — {extra}" if extra else ""))
        if not cond:
            fails += 1

    src = _sharp_hd()
    out, note, analysis = run_top_tier_quality(
        src, enhance_tier="hd", use_generative=False, preserve_faces=False,
    )
    scores = analysis.get("scores", {})
    diff = _mean_diff(out, src)
    check(
        "foto nítida en HD cambia",
        diff > 0.4 and "conservad" not in note,
        f"diff={diff:.3f} nit={scores.get('sharpness')} poster={scores.get('poster')} | {note}",
    )
    check("foto nítida en HD no crece", out.size == src.size, str(out.size))
    for token in ("desposterizar", "color", "luz"):
        check(f"nota de calidad contiene {token}", token in note)

    small = _soft_small()
    gpu_busy = _listening(8188) or _listening(7861)
    real_upscale = None
    if gpu_busy:
        from roop.img_editor.image_quality_pipeline import ImageQualityFinisher
        real_upscale = ImageQualityFinisher.upscale_to_tier

        def _lanczos_probe(self, image, tier="hd", **kwargs):
            target = {"hd": 1920, "4k": 3840, "8k": 7680}.get(tier, 1920)
            w, h = image.size
            scale = target / float(max(w, h))
            nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            return image.resize((nw, nh), Image.LANCZOS), f"lanczos sonda {nw}×{nh}"

        ImageQualityFinisher.upscale_to_tier = _lanczos_probe
        print("ComfyUI o la UI están activos: la foto pequeña se amplía con Lanczos de sonda")
    try:
        grown, note_small, _ = run_top_tier_quality(
            small, enhance_tier="hd", use_generative=False, preserve_faces=False,
        )
    finally:
        if real_upscale is not None:
            from roop.img_editor.image_quality_pipeline import ImageQualityFinisher
            ImageQualityFinisher.upscale_to_tier = real_upscale
    check(
        "foto pequeña crece hacia HD",
        max(grown.size) > max(small.size) >= 160,
        f"{small.size} -> {grown.size} | {note_small}",
    )
    check("la ampliación queda anotada", "×" in note_small or "ya ~" not in note_small.lower() or max(grown.size) > max(small.size))

    edited, note_edit, _ = run_quality_enhance_after_edit(
        src, src, enhance_tier="hd", use_generative=False, preserve_faces=False,
    )
    diff_edit = _mean_diff(edited, src)
    check(
        "calidad después de editar también cambia",
        diff_edit > 0.4 and edited.size == src.size and "conservad" not in note_edit,
        f"diff={diff_edit:.3f} | {note_edit}",
    )
    for token in ("desposterizar", "color", "luz"):
        check(f"nota post-edición contiene {token}", token in note_edit)

    none_steps = {
        "deposter": False, "upscale": False, "color": False, "realism": False,
    }
    off, note_off, _ = run_top_tier_quality(
        src, enhance_tier="hd", use_generative=False, preserve_faces=False,
        quality_steps=none_steps,
    )
    check(
        "sin pasos devuelve la misma foto",
        np.array_equal(np.array(off), np.array(src)) and "ninguna opción" in note_off,
        note_off,
    )
    off_edit, note_off_edit, _ = run_quality_enhance_after_edit(
        src, src, enhance_tier="hd", use_generative=False, preserve_faces=False,
        quality_steps=none_steps,
    )
    check(
        "sin pasos tras editar devuelve la misma foto",
        np.array_equal(np.array(off_edit), np.array(src)) and "ninguna opción" in note_off_edit,
        note_off_edit,
    )

    kept, note_kept, _ = run_top_tier_quality(
        small, enhance_tier="hd", use_generative=False, preserve_faces=False,
        quality_steps={"upscale": False},
    )
    check(
        "sin ampliar la foto pequeña no crece",
        kept.size == small.size and "paso ampliar omitido" in note_kept
        and "desposterizar" in note_kept,
        f"{small.size} -> {kept.size} | {note_kept}",
    )

    colored, note_color, _ = run_top_tier_quality(
        src, enhance_tier="hd", use_generative=False, preserve_faces=False,
        quality_steps={
            "deposter": False, "upscale": False, "color": True, "realism": False,
        },
    )
    check(
        "solo color y luz conserva el tamaño",
        colored.size == src.size
        and "color " in note_color
        and "luz " in note_color
        and "paso desposterizar omitido" in note_color
        and "paso realismo omitido" in note_color
        and "paso color omitido" not in note_color
        and _mean_diff(colored, src) > 0.05,
        f"diff={_mean_diff(colored, src):.3f} | {note_color}",
    )

    canvas = src.resize((768, 432), Image.Resampling.LANCZOS)
    back, note_back, _ = run_quality_enhance_after_edit(
        canvas, src, enhance_tier="hd", use_generative=False, preserve_faces=False,
        quality_steps={
            "deposter": False, "upscale": False, "color": True, "realism": False,
        },
    )
    check(
        "sin ampliar vuelve al tamaño de la foto original",
        back.size == src.size and f"tamaño original {src.size[0]}×{src.size[1]}" in note_back,
        f"{canvas.size} -> {back.size} | {note_back}",
    )

    from roop.img_editor.image_quality_pipeline import get_quality_finisher
    from roop.img_editor.top_tier_finish import _apply_quality_realism

    textured, texture_note = _apply_quality_realism(
        get_quality_finisher(),
        src,
        {"gentle": False, "sharpen": 1.08},
        0.64,
    )
    check(
        "poster alto aplica textura local",
        "realismo textura local" in texture_note and "omitida" not in texture_note,
        texture_note,
    )
    check(
        "la textura local cambia la foto",
        _mean_diff(textured, src) > 0.05 and textured.size == src.size,
        f"diff={_mean_diff(textured, src):.3f}",
    )
    only_realism, note_realism, _ = run_top_tier_quality(
        src, enhance_tier="hd", use_generative=False, preserve_faces=False,
        quality_steps={
            "deposter": False, "upscale": False, "color": False, "realism": True,
        },
    )
    check(
        "solo realismo no se omite",
        "paso realismo omitido" not in note_realism
        and ("nitidez" in note_realism or "realismo textura" in note_realism)
        and only_realism.size == src.size
        and _mean_diff(only_realism, src) > 0.05,
        note_realism,
    )

    print(f"TOTAL FAIL={fails}")
    return fails


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
