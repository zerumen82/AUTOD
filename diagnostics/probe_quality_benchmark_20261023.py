# -*- coding: utf-8 -*-
"""Banco de referencia del Modo Calidad (medición, no cambio de producción).

Tres fixtures deterministas (nítida HD, blanda pequeña, posterizada) pasan por
``run_top_tier_quality`` con todos los pasos y sin generación (ComfyUI no
interviene). Se mide la ENTRADA y la SALIDA con el mismo analizador del
finisher: nitidez, posterización, diferencia de color y tiempo. El JSON queda
en ``diagnostics/out/quality_benchmark_20261023/benchmark.json`` y sirve de
línea base para la puerta de aceptación y la política de upscale.

No mide identidad: los fixtures no tienen caras. La identidad se mide en un
render real con foto (ver spec/03) o pasando ``--face-photo``.

Uso:
    python diagnostics/probe_quality_benchmark_20261023.py
    python diagnostics/probe_quality_benchmark_20261023.py --face-photo ruta.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "diagnostics" / "out" / "quality_benchmark_20261023"
TIER = "hd"


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


def _posterized() -> Image.Image:
    """Degradado suave cuantizado a 6 niveles por canal (banding real)."""
    yy, xx = np.mgrid[0:1440, 0:1920]
    v = 40 + 170 * (0.55 * (xx / 1919.0) + 0.45 * (yy / 1439.0))
    rgb = np.stack([v, v * 0.9 + 12, v * 0.75 + 40], axis=-1)
    quant = np.round(np.clip(rgb, 0, 255) / 255.0 * 5.0) / 5.0 * 255.0
    return Image.fromarray(quant.astype(np.uint8))


def _scores(finisher, image: Image.Image) -> dict:
    a = finisher.analyze_image(image, tier=TIER)
    s = a.get("scores", {})
    return {
        "sharpness": round(float(s.get("sharpness", 0.0)), 4),
        "poster": round(float(s.get("poster", 0.0)), 4),
        "noise": round(float(s.get("noise", 0.0)), 4),
    }


def _delta_e(a: Image.Image, b: Image.Image) -> float:
    """Diferencia media de color en Lab (DeltaE76) a lado común de 640 px."""
    import cv2
    side = 640
    out = []
    for img in (a, b):
        long_side = max(img.size)
        scale = side / float(long_side) if long_side > side else 1.0
        nw = max(1, int(round(img.size[0] * scale)))
        nh = max(1, int(round(img.size[1] * scale)))
        arr = cv2.cvtColor(np.array(img.convert("RGB").resize((nw, nh), Image.LANCZOS)), cv2.COLOR_RGB2Lab)
        out.append(arr.astype(np.float32))
    h = min(out[0].shape[0], out[1].shape[0])
    w = min(out[0].shape[1], out[1].shape[1])
    return float(np.mean(np.linalg.norm(out[0][:h, :w] - out[1][:h, :w], axis=-1)))


def _identity(original: Image.Image, result: Image.Image) -> dict:
    """Coseno ArcFace mejor emparejado. Ninguna cara => se declara."""
    try:
        from roop.img_editor.face_preserver import FacePreserver
        fp = FacePreserver()
        ok, msg = fp.initialize()
        if not ok:
            return {"error": msg, "cosine": None}
        src = fp.detect_faces(original)
        dst = fp.detect_faces(result)
        if not src or not dst:
            return {"faces_in": len(src), "faces_out": len(dst), "cosine": None}
        best = 0.0
        for a in src:
            for b in dst:
                best = max(best, float(fp.compare_faces(a.get("embedding"), b.get("embedding"))))
        return {"faces_in": len(src), "faces_out": len(dst), "cosine": round(best, 4)}
    except Exception as exc:  # el banco no debe caerse por el detector
        return {"error": str(exc), "cosine": None}


def _contract_checks(finisher) -> int:
    """Contrato de la puerta de aceptación y de los topes de resolución.

    Barato y determinista: no amplía nada; comprueba que un paso que empeora se
    revierte y que un paso neutro se acepta, y que el tier no sube por encima
    de lo que permite config/quality_mode.json.
    """
    from PIL import ImageFilter

    from roop.img_editor.image_quality_pipeline import limit_quality_tier, quality_settings
    from roop.img_editor.top_tier_finish import _StepGate

    fails = 0

    def check(name: str, cond: bool, extra: str = "") -> None:
        nonlocal fails
        print(f"{'OK  ' if cond else 'FAIL'} {name}" + (f" — {extra}" if extra else ""))
        if not cond:
            fails += 1

    src = _sharp_hd()
    gate = _StepGate(finisher, TIER)
    before_m = gate.snapshot(src)
    blurred = src.filter(ImageFilter.GaussianBlur(2.0))
    kept, why = gate.check(
        "nitidez de prueba", src, blurred, before_m, gate.snapshot(blurred), checks=("sharp",),
    )
    check("la puerta revierte un paso que borra nitidez", kept is src and "revertido" in why, why)
    same, why_same = gate.check(
        "paso neutro", src, src.copy(), before_m, gate.snapshot(src.copy()), checks=("sharp", "clip"),
    )
    check("la puerta acepta un paso neutro", same is not src and not why_same, why_same)

    cfg = quality_settings().get("upscale", {})
    expected_8k = "8k" if bool(cfg.get("allow_8k", False)) else "4k"
    tier_640, _ = limit_quality_tier(Image.new("RGB", (640, 480)), "8k")
    check("8K sobre 640 px no pasa de HD", tier_640 == "hd", f"640px+8k → {tier_640}")
    tier_720, _ = limit_quality_tier(Image.new("RGB", (1280, 720)), "8k")
    check("1280 px no llega a 8K", tier_720 == expected_8k, f"1280px+8k → {tier_720}")
    tier_hd, _ = limit_quality_tier(Image.new("RGB", (1920, 1080)), "hd")
    check("HD no se degrada", tier_hd == "hd", tier_hd)
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--face-photo", default="", help="foto real con cara para medir identidad")
    args = ap.parse_args()

    from roop.img_editor.image_quality_pipeline import get_quality_finisher
    from roop.img_editor.top_tier_finish import run_top_tier_quality

    finisher = get_quality_finisher()
    cases = [
        ("nitida_hd", _sharp_hd()),
        ("blanda_pequena", _soft_small()),
        ("posterizada", _posterized()),
    ]

    rows = []
    fails = 0
    print("Contrato de la puerta y de los topes:")
    fails += _contract_checks(finisher)
    print(f"Banco Modo Calidad — tier={TIER.upper()} (sin generación, sin caras)")
    for name, img in cases:
        before = _scores(finisher, img)
        t0 = time.perf_counter()
        out, note, analysis = run_top_tier_quality(
            img, enhance_tier=TIER, use_generative=False, preserve_faces=False,
        )
        seconds = time.perf_counter() - t0
        if out is None:
            print(f"FAIL {name}: el acabado no devolvió imagen")
            fails += 1
            continue
        after = _scores(finisher, out)
        row = {
            "case": name,
            "input_size": list(img.size),
            "output_size": list(out.size),
            "input": before,
            "output": after,
            "delta_sharpness": round(after["sharpness"] - before["sharpness"], 4),
            "delta_poster": round(after["poster"] - before["poster"], 4),
            "delta_e": round(_delta_e(img, out), 3),
            "seconds": round(seconds, 2),
            "note": note,
            "analysis_present": bool(analysis),
        }
        rows.append(row)
        print(
            f"  {name:16s} {img.size}->{out.size}  nit {before['sharpness']:.3f}->{after['sharpness']:.3f}"
            f"  post {before['poster']:.3f}->{after['poster']:.3f}  dE {row['delta_e']:.2f}  {seconds:.1f}s"
        )
        # Invariantes que HOY deben cumplirse; la calidad se mide, no se asevera.
        if name == "nitida_hd" and out.size != img.size:
            print(f"FAIL {name}: cambió el tamaño {img.size}->{out.size}")
            fails += 1
        if name == "blanda_pequena" and max(out.size) <= max(img.size):
            print(f"FAIL {name}: no creció ({img.size}->{out.size})")
            fails += 1
        if name == "posterizada" and out.size != img.size:
            print(f"FAIL {name}: cambió el tamaño {img.size}->{out.size}")
            fails += 1

    if args.face_photo:
        path = Path(args.face_photo)
        if not path.exists():
            print(f"FAIL --face-photo no existe: {path}")
            fails += 1
        else:
            photo = Image.open(path).convert("RGB")
            before = _scores(finisher, photo)
            t0 = time.perf_counter()
            out, note, _ = run_top_tier_quality(
                photo, enhance_tier=TIER, use_generative=False, preserve_faces=True,
            )
            seconds = time.perf_counter() - t0
            ident = _identity(photo, out) if out is not None else {"cosine": None}
            row = {
                "case": "foto_real",
                "file": str(path),
                "input_size": list(photo.size),
                "output_size": list(out.size) if out is not None else None,
                "input": before,
                "output": _scores(finisher, out) if out is not None else None,
                "identity": ident,
                "seconds": round(seconds, 2),
                "note": note,
            }
            rows.append(row)
            print(
                f"  foto_real        {photo.size}->{row['output_size']}  identidad {ident.get('cosine')}"
                f"  {seconds:.1f}s"
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"tier": TIER, "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "cases": rows}
    (OUT_DIR / "benchmark.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JSON: {OUT_DIR / 'benchmark.json'}")
    print(f"TOTAL FAIL={fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
