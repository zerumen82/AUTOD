# -*- coding: utf-8 -*-
"""Calibra bandas y A/B de upscale del Modo Calidad con fotos reales.

Mide, sin GPU y sin ComfyUI (ESRGAN forzado a CPU):
  1. Bandas en fotos reales limpias y en las mismas fotos posterizadas
     (8/16/32 niveles) y cuánto bajan con desposterizar/naturalizar.
  2. Cambio natural de bandas con los pasos suaves del acabado, para fijar la
     tolerancia de la puerta sin revertir pasos legítimos.
  3. A/B real ESRGAN x2 frente a Lanczos en 3 zonas con textura: ratio de
     nitidez y correlación de estructura. Control negativo: ruido (textura
     inventada) para comprobar que la estructura lo separa.
  4. Máscara facial 468 puntos frente a la elipse: peso en las esquinas del
     recorte (fuera de la cara) y cobertura en el centro.

Salida: diagnostics/out/quality_calibration_20261110/{report.json,report.md}
No cambia la configuración: propone umbrales medidos y el informe los deja
escritos. Exit 1 sólo si una invariante del instrumento falla.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import roop.globals  # noqa: E402
import roop.img_editor.top_tier_finish as ttf  # noqa: E402
from roop.img_editor.image_quality_pipeline import (  # noqa: E402
    ImageQualityFinisher,
    quality_settings,
)

OUTPUT = ROOT / "diagnostics/out/quality_calibration_20261110"

CANDIDATES = [
    "diagnostics/out/appearance_reference_20261023/1793812998803975900/result.png",
    "diagnostics/out/appearance_reference_20261023/1792858682799844700/result.png",
    "diagnostics/out/appearance_reference_20261023/1792858216525529500/result.png",
    "diagnostics/out/appearance_reference_20261023/1792854592938590500/result.png",
    "diagnostics/out/appearance_reference_20261023/1792952153054322000/result.png",
    "img_editor/edit_1794006695.png",
]


def _load(path: Path, max_side: int = 1600) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / float(max(img.size))
        img = img.resize(
            (max(1, int(img.size[0] * scale)), max(1, int(img.size[1] * scale))),
            Image.LANCZOS,
        )
    return img


def _band(img: Image.Image) -> tuple[float, float]:
    start = time.perf_counter()
    value = ttf._gate_banding(img)
    return float(value), time.perf_counter() - start


def _quantize(img: Image.Image, levels: int) -> Image.Image:
    step = max(1, 256 // levels)
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    quant = (np.round(arr / step) * step).clip(0, 255).astype(np.uint8)
    return Image.fromarray(quant)


def _ab_metrics(patch: Image.Image, candidate: Image.Image) -> tuple[float, float]:
    ref = patch.resize(candidate.size, Image.LANCZOS)
    ai_l = np.asarray(candidate.convert("L"), dtype=np.float32)
    ref_l = np.asarray(ref.convert("L"), dtype=np.float32)
    var_ai = float(cv2.Laplacian(ai_l, cv2.CV_32F).var())
    var_ref = float(cv2.Laplacian(ref_l, cv2.CV_32F).var())
    ratio = var_ai / var_ref if var_ref > 1e-6 else 0.0
    ai_g = np.hypot(
        cv2.Sobel(ai_l, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(ai_l, cv2.CV_32F, 0, 1, ksize=3),
    ).ravel()
    ref_g = np.hypot(
        cv2.Sobel(ref_l, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(ref_l, cv2.CV_32F, 0, 1, ksize=3),
    ).ravel()
    if ai_g.std() < 1e-6 or ref_g.std() < 1e-6:
        return ratio, 0.0
    corr = float(np.corrcoef(ai_g, ref_g)[0, 1])
    return ratio, corr if np.isfinite(corr) else 0.0


def _mesh_probe(photo: Image.Image) -> dict:
    app = ttf._get_face_mesh()
    if app is None:
        return {"available": False}
    rgb = np.ascontiguousarray(np.array(photo.convert("RGB")))
    result = app.process(rgb)
    faces = getattr(result, "multi_face_landmarks", None) if result else None
    if not faces:
        return {"available": True, "face": False}
    mesh_points = faces[0].landmark
    width, height = photo.size
    points = np.array(
        [[p.x * width, p.y * height] for p in mesh_points], dtype=np.float32,
    )
    x1, y1 = np.floor(points.min(axis=0)).astype(int)
    x2, y2 = np.ceil(points.max(axis=0)).astype(int)
    margin = int(max(x2 - x1, y2 - y1) * 0.30)
    x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
    x2, y2 = min(width, x2 + margin), min(height, y2 + margin)
    crop = cv2.cvtColor(
        np.array(photo.crop((x1, y1, x2, y2)).convert("RGB")), cv2.COLOR_RGB2BGR,
    )
    mesh_mask = ttf._face_mesh_mask(crop, 1.0)
    ellipse_mask = ttf._face_restore_mask(crop, 1.0, use_mesh=False)

    def corners(mask):
        fh, fw = mask.shape
        ch, cw = max(1, fh // 10), max(1, fw // 10)
        return float(np.mean([
            mask[:ch, :cw].mean(), mask[:ch, -cw:].mean(),
            mask[-ch:, :cw].mean(), mask[-ch:, -cw:].mean(),
        ]))

    def center(mask):
        fh, fw = mask.shape
        return float(mask[fh // 4:3 * fh // 4, fw // 4:3 * fw // 4].mean())

    return {
        "available": True,
        "face": True,
        "crop": [int(x1), int(y1), int(x2), int(y2)],
        "mesh_used": mesh_mask is not None,
        "mesh_corners": corners(mesh_mask) if mesh_mask is not None else None,
        "ellipse_corners": corners(ellipse_mask),
        "mesh_center": center(mesh_mask) if mesh_mask is not None else None,
        "ellipse_center": center(ellipse_mask),
    }


def main() -> int:
    photos = [ROOT / p for p in CANDIDATES if (ROOT / p).is_file()]
    if not photos:
        print("Sin fotos reales de calibración", flush=True)
        return 1
    OUTPUT.mkdir(parents=True, exist_ok=True)
    settings = quality_settings()
    finisher = ImageQualityFinisher()
    print(f"[Calibracion] fotos={len(photos)} paso_max=1600", flush=True)

    band_rows = []
    for path in photos:
        photo = _load(path)
        clean, t_clean = _band(photo)
        q8 = _quantize(photo, 8)
        q16 = _quantize(photo, 16)
        q32 = _quantize(photo, 32)
        b8, _ = _band(q8)
        b16, _ = _band(q16)
        b32, _ = _band(q32)
        softened = finisher.soften_poster_bands(q8, 0.35)
        b8s, _ = _band(softened)
        depixel = finisher.depixelize(q8, 0.5)
        b8d, _ = _band(depixel)
        gentle = [
            finisher.depixelize(photo, 0.12),
            finisher.enhance_color(photo, 0.18),
            finisher.detail_boost(photo, 0.14),
            finisher.sharpen(photo, 1.04),
        ]
        gentle_bands = [_band(g)[0] for g in gentle]
        row = {
            "photo": str(path.relative_to(ROOT)).replace("\\", "/"),
            "size": list(photo.size),
            "clean": round(clean, 3),
            "q32": round(b32, 3),
            "q16": round(b16, 3),
            "q8": round(b8, 3),
            "q8_soften": round(b8s, 3),
            "q8_depixel": round(b8d, 3),
            "gentle_band_deltas": [round(b - clean, 3) for b in gentle_bands],
            "band_ms": round(t_clean * 1000.0, 1),
        }
        band_rows.append(row)
        print(
            f"[Bandas] {row['photo']} {row['size']} limpieza={clean:.2f} "
            f"q32={b32:.2f} q16={b16:.2f} q8={b8:.2f} "
            f"soften={b8s:.2f} depixel={b8d:.2f} "
            f"delta_pasos={row['gentle_band_deltas']}",
            flush=True,
        )

    saved_providers = list(getattr(roop.globals, "execution_providers", []) or [])
    ab_rows = []
    negative = []
    try:
        roop.globals.execution_providers = ["CPUExecutionProvider"]
        from roop.processors.Frame_Upscale import Frame_Upscale

        upscaler = Frame_Upscale()
        upscaler.Initialize({"devicename": "cpu", "subtype": "esrganx2"})
        rng = np.random.default_rng(17)
        for path in photos:
            photo = _load(path)
            boxes = finisher._textured_crop_boxes(photo, 256, 3)
            row = {
                "photo": str(path.relative_to(ROOT)).replace("\\", "/"),
                "zones": len(boxes), "ratios": [], "corrs": [],
            }
            for box in boxes:
                patch = photo.crop(box)
                bgr = cv2.cvtColor(np.array(patch.convert("RGB")), cv2.COLOR_RGB2BGR)
                start = time.perf_counter()
                ai_bgr = upscaler.Run(bgr)
                elapsed = time.perf_counter() - start
                ai = Image.fromarray(cv2.cvtColor(ai_bgr, cv2.COLOR_BGR2RGB))
                ratio, corr = _ab_metrics(patch, ai)
                row["ratios"].append(round(ratio, 3))
                row["corrs"].append(round(corr, 3))
                row.setdefault("seconds", []).append(round(elapsed, 2))
                noise = Image.fromarray(
                    rng.integers(0, 255, (ai.size[1], ai.size[0], 3), dtype=np.uint8)
                )
                n_ratio, n_corr = _ab_metrics(patch, noise)
                negative.append((n_ratio, n_corr))
            if row["ratios"]:
                ab_rows.append(row)
                print(
                    f"[A/B] {row['photo']} ratio={row['ratios']} "
                    f"estructura={row['corrs']} s={row.get('seconds')}",
                    flush=True,
                )
        upscaler.Release()
    except Exception as exc:
        print(f"[A/B] no medido: {exc}", flush=True)
    finally:
        roop.globals.execution_providers = saved_providers

    mesh_rows = []
    for path in photos[:3]:
        photo = _load(path, 1200)
        probe = _mesh_probe(photo)
        probe["photo"] = str(path.relative_to(ROOT)).replace("\\", "/")
        mesh_rows.append(probe)
        if probe.get("face"):
            print(
                f"[Malla] {probe['photo']} mesh={probe['mesh_used']} "
                f"esquinas malla={probe['mesh_corners']} elipse={probe['ellipse_corners']} "
                f"centro malla={probe['mesh_center']} elipse={probe['ellipse_center']}",
                flush=True,
            )
        else:
            print(f"[Malla] {probe['photo']} sin cara: {probe}", flush=True)

    ratios = [v for row in ab_rows for v in row["ratios"]]
    corrs = [v for row in ab_rows for v in row["corrs"]]
    neg_corrs = [c for _r, c in negative]
    gentle_rises = [
        max(row["gentle_band_deltas"]) for row in band_rows
    ]

    def _median(values):
        return round(float(np.median(values)), 3) if values else None

    suggestions = {
        "ai_gain_min_ratio_current": settings["upscale"].get("ai_gain_min_ratio"),
        "ai_gain_min_ratio_median_real": _median(ratios),
        "ai_gain_min_ratio_min_real": round(min(ratios), 3) if ratios else None,
        "ai_gain_min_corr_current": settings["upscale"].get("ai_gain_min_corr"),
        "ai_gain_min_corr_median_real": _median(corrs),
        "ai_gain_min_corr_min_real": round(min(corrs), 3) if corrs else None,
        "ai_gain_min_corr_negative_max": round(max(neg_corrs), 3) if neg_corrs else None,
        "band_rise_tolerance_current": settings["gate"].get("band_rise_tolerance"),
        "band_rise_max_gentle_step": round(max(gentle_rises), 3) if gentle_rises else None,
        "band_clean_max": round(max(row["clean"] for row in band_rows), 3),
        "band_q8_min": round(min(row["q8"] for row in band_rows), 3),
        "band_q8_after_fix_max": round(
            max(max(row["q8_soften"], row["q8_depixel"]) for row in band_rows), 3
        ),
        "band_q8_fix_min_gain": round(
            min(
                row["q8"] - min(row["q8_soften"], row["q8_depixel"])
                for row in band_rows
            ),
            3,
        ) if band_rows else None,
    }

    report = {
        "date": "2026-11-10",
        "photos": [row["photo"] for row in band_rows],
        "banding": band_rows,
        "upscale_ab": ab_rows,
        "upscale_negative_control": {
            "ratio_median": _median([r for r, _c in negative]),
            "corr_median": _median(neg_corrs),
            "corr_max": round(max(neg_corrs), 3) if neg_corrs else None,
        },
        "mesh_vs_ellipse": mesh_rows,
        "suggestions": suggestions,
        "settings_snapshot": settings,
    }
    (OUTPUT / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Calibración Modo Calidad — fotos reales (2026-11-10)",
        "",
        f"Fotos: {len(band_rows)}. ESRGAN x2 forzado a CPU; sin GPU ni ComfyUI.",
        "",
        "## Bandas",
        "",
        "| foto | tamaño | limpia | q32 | q16 | q8 | q8+naturalizar | q8+desposterizar | Δ pasos suaves | ms |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in band_rows:
        lines.append(
            f"| {row['photo']} | {row['size'][0]}×{row['size'][1]} | "
            f"{row['clean']:.2f} | {row['q32']:.2f} | {row['q16']:.2f} | "
            f"{row['q8']:.2f} | {row['q8_soften']:.2f} | {row['q8_depixel']:.2f} | "
            f"{row['gentle_band_deltas']} | {row['band_ms']:.0f} |"
        )
    lines += [
        "",
        f"- Máximo de foto limpia: **{suggestions['band_clean_max']:.3f}**.",
        f"- Mínimo del posterizado a 8 niveles: **{suggestions['band_q8_min']:.3f}**.",
        f"- Máximo tras naturalizar/desposterizar: **{suggestions['band_q8_after_fix_max']:.3f}** "
        f"(ganancia mínima por foto **{suggestions['band_q8_fix_min_gain']:.3f}**).",
        f"- Subida máxima con los pasos suaves del acabado: **{suggestions['band_rise_max_gentle_step']:.3f}** "
        f"(tolerancia actual {suggestions['band_rise_tolerance_current']}).",
        "",
        "## A/B de upscale (ESRGAN x2 real vs Lanczos)",
        "",
        "| foto | zonas | ratio | estructura | segundos |",
        "|---|---|---|---|---|",
    ]
    for row in ab_rows:
        lines.append(
            f"| {row['photo']} | {row['zones']} | {row['ratios']} | "
            f"{row['corrs']} | {row.get('seconds')} |"
        )
    lines += [
        "",
        f"- Ratio real: mediana **{suggestions['ai_gain_min_ratio_median_real']}**, "
        f"mínimo **{suggestions['ai_gain_min_ratio_min_real']}** "
        f"(umbral actual {suggestions['ai_gain_min_ratio_current']}).",
        f"- Estructura real: mediana **{suggestions['ai_gain_min_corr_median_real']}**, "
        f"mínimo **{suggestions['ai_gain_min_corr_min_real']}** "
        f"(umbral actual {suggestions['ai_gain_min_corr_current']}).",
        f"- Control negativo (ruido): estructura mediana "
        f"**{report['upscale_negative_control']['corr_median']}**, "
        f"máximo **{report['upscale_negative_control']['corr_max']}**.",
        "",
        "## Malla 468 frente a elipse",
        "",
        "| foto | malla | esquinas malla | esquinas elipse | centro malla | centro elipse |",
        "|---|---|---|---|---|---|",
    ]
    for row in mesh_rows:
        lines.append(
            f"| {row['photo']} | {row.get('mesh_used')} | {row.get('mesh_corners')} | "
            f"{row.get('ellipse_corners')} | {row.get('mesh_center')} | "
            f"{row.get('ellipse_center')} |"
        )
    lines += [
        "",
        "## Límites",
        "",
        "- La degradación por bandas es sintética sobre contenido real (cuantización a",
        "  8/16/32 niveles), no una foto de cámara con bandas nativas.",
        "- El control negativo de textura inventada es ruido, no una alucinación de",
        "  un modelo: mide separación de estructura, no todas las alucinaciones.",
        "- Sin PASS visual: la aceptación sigue siendo del usuario.",
    ]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[Calibracion] informe={OUTPUT / 'report.md'}", flush=True)
    print(f"[Calibracion] sugerencias={json.dumps(suggestions, ensure_ascii=False)}", flush=True)

    fails = 0
    if suggestions["band_q8_min"] <= suggestions["band_clean_max"] + 0.15:
        print("FAIL bandas: la métrica no separa posterizado de limpio", flush=True)
        fails += 1
    fix_gain = suggestions.get("band_q8_fix_min_gain")
    if fix_gain is None or fix_gain <= 0.10:
        print(
            "FAIL bandas: naturalizar/desposterizar no baja la medida en todas "
            f"las fotos (ganancia mínima {fix_gain})",
            flush=True,
        )
        fails += 1
    if corrs and neg_corrs and np.median(corrs) <= np.median(neg_corrs):
        print("FAIL A/B: la estructura no separa ESRGAN de ruido", flush=True)
        fails += 1
    mesh_used = [r for r in mesh_rows if r.get("mesh_used")]
    if mesh_used:
        worse = [
            r for r in mesh_used
            if r["mesh_corners"] is not None
            and r["mesh_corners"] >= r["ellipse_corners"]
        ]
        if worse:
            print(f"FAIL malla: sin mejora en esquinas ({len(worse)} foto(s))", flush=True)
            fails += 1
    print(f"TOTAL FAIL={fails}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
