#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnóstico de cobertura de la zona medida (CPU, sin render).

Sobre la misma foto del caso, con el MISMO medidor de ropa que usa producción
(`_get_clothing_segmenter().segment_clothing`, umbral 0,45), se mide dónde
están los restos de la prenda que la entrega con LongCat Turbo deja colgando
(tiras/cruces del hombro y el brazo):

  A) FUENTE: ropa medida de la foto original vs zona de generación.
     Ropa original FUERA de la zona = prendas que la zona ya no cubre de
     origen (ahí el modelo no puede pintar nada).
  B) RESULTADO: ropa detectada en la entrega vs zona.
     Resto DENTRO de la zona = el modelo volvió a pintar tela donde sí tenía
     permiso (varianza/receta); resto FUERA = tela intacta no cubierta.

Salida: `coverage_overlay.png` (resto fuera en rojo, dentro en azul, contorno
verde de la zona) y `metrics.json` en
`diagnostics/out/zone_coverage_20261028/<timestamp>/`.

Solo lectura: no toca config, producción, UI ni GPU.

Uso:
    python -u -m diagnostics.probe_editor_zone_coverage_20261028
    python -u -m diagnostics.probe_editor_zone_coverage_20261028 --result ruta.png
    python -u -m diagnostics.probe_editor_zone_coverage_20261028 --refine
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from roop.img_editor import selected_region_edit as sre  # noqa: E402
from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402

SOURCE = ROOT / "ui/tob/ComfyUI/input/fast_1793020935.png"
DEFAULT_RESULT = (ROOT / "diagnostics/out/qwen_body_ab_20261028/"
                  "20261028_164231/result_longcat_turbo.png")
THRESHOLD = 0.45  # mismo umbral que _measured_clothing_zone
# Consultas DIAGNÓSTICAS de tela fina (solo medición, nunca enrutan edición).
REFINE_PROMPTS = (
    "a striped fabric strap",
    "a thin strip of cloth",
    "a piece of torn fabric",
)


def clothing_mask(manager: ImgEditorManager, image: Image.Image,
                  threshold: float = THRESHOLD) -> np.ndarray | None:
    """Máscara de ropa con el segmentador de producción, sin dilatar."""
    try:
        segmenter = manager._get_clothing_segmenter()
    except Exception as exc:
        print(f"[Cobertura] sin segmentador de ropa: {exc}", flush=True)
        return None
    if not segmenter or not segmenter.is_available():
        print("[Cobertura] segmentador de ropa no disponible", flush=True)
        return None
    try:
        mask, _ = segmenter.segment_clothing(image, threshold=threshold)
    except Exception as exc:
        print(f"[Cobertura] segment_clothing falló: {exc}", flush=True)
        return None
    if mask is None:
        return None
    clothes = np.asarray(mask.convert("L")) > 127
    return clothes if clothes.any() else None


def components(mask: np.ndarray, limit: int = 8) -> list[dict]:
    """Componentes conexas de una máscara, por tamaño descendente."""
    if not mask.any():
        return []
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        out.append({"bbox": [int(x), int(y), int(x + w), int(y + h)],
                    "area": int(area)})
    out.sort(key=lambda c: -c["area"])
    return out[:limit]


def summarize(cand: np.ndarray, zone: np.ndarray, limit: int = 8) -> dict:
    """Resumen de una máscara candidata contra la zona de generación."""
    inside = cand & zone
    outside = cand & ~zone
    total = int(cand.sum())
    return {
        "px": total,
        "inside_zone_px": int(inside.sum()),
        "outside_zone_px": int(outside.sum()),
        "outside_ratio": round(float(outside.sum()) / total, 4) if total else 0.0,
        "inside_components": components(inside, limit),
        "outside_components": components(outside, limit),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=str(SOURCE))
    ap.add_argument("--result", default=str(DEFAULT_RESULT),
                    help="entrega a diagnosticar (por defecto el A/B LongCat)")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--refine", action="store_true",
                    help="barrido de detección de tela fina (consultas "
                         "diagnósticas) y de ropa con umbrales bajos")
    ap.add_argument("--refine-thresholds", nargs="+", type=float,
                    default=[0.5, 0.4, 0.3])
    args = ap.parse_args()

    source_path = Path(args.source)
    result_path = Path(args.result)
    for path in (source_path, result_path):
        if not path.exists():
            print(f"[Cobertura] FAIL: no existe {path}", flush=True)
            return 1

    photo = Image.open(source_path).convert("RGB")
    result = Image.open(result_path).convert("RGB")
    if result.size != photo.size:
        result = result.resize(photo.size, Image.Resampling.LANCZOS)

    manager = ImgEditorManager()

    # Zona de generación de producción (misma ruta que el A/B y la UI).
    person, _ = sre._described_subject_mask(manager, photo, ["The woman"])
    zone = sre._measured_clothing_zone(manager, photo, person, None)
    if zone is None:
        print("[Cobertura] FAIL: la zona medida quedó vacía", flush=True)
        return 1
    zone = np.asarray(zone) > 0
    print(f"[Cobertura] zona de generación={int(zone.sum())} px", flush=True)

    report: dict = {
        "source": str(source_path),
        "result": str(result_path),
        "threshold": args.threshold,
        "zone_px": int(zone.sum()),
    }

    # A) Ropa original vs zona (¿la zona cubre la prenda de origen?).
    src_clothes = clothing_mask(manager, photo)
    if src_clothes is None:
        print("[Cobertura] FAIL: no se pudo medir la ropa de la fuente", flush=True)
        return 1
    src_inside = int((src_clothes & zone).sum())
    src_outside = src_clothes & ~zone
    report["source_clothing"] = {
        "px": int(src_clothes.sum()),
        "inside_zone_px": src_inside,
        "outside_zone_px": int(src_outside.sum()),
        "outside_ratio": round(float(src_outside.sum()) / float(src_clothes.sum()), 4),
        "outside_components": components(src_outside),
    }
    print(f"[Cobertura] FUENTE: ropa={report['source_clothing']['px']} px, "
          f"fuera de la zona={report['source_clothing']['outside_zone_px']} px "
          f"({report['source_clothing']['outside_ratio']:.1%}) en "
          f"{len(report['source_clothing']['outside_components'])} componentes",
          flush=True)

    # B) Ropa en la entrega vs zona (¿qué restos quedan y dónde?).
    res_clothes = clothing_mask(manager, result)
    if res_clothes is None:
        print("[Cobertura] AVISO: no se pudo medir ropa en la entrega",
              flush=True)
        report["result_clothing"] = None
    else:
        res_inside = res_clothes & zone
        res_outside = res_clothes & ~zone
        report["result_clothing"] = {
            "px": int(res_clothes.sum()),
            "inside_zone_px": int(res_inside.sum()),
            "outside_zone_px": int(res_outside.sum()),
            "outside_ratio": (round(float(res_outside.sum()) / float(res_clothes.sum()), 4)
                              if res_clothes.any() else 0.0),
            "inside_components": components(res_inside),
            "outside_components": components(res_outside),
        }
        print(f"[Cobertura] ENTREGA: tela restante={report['result_clothing']['px']} px, "
              f"dentro de la zona={report['result_clothing']['inside_zone_px']} px, "
              f"fuera={report['result_clothing']['outside_zone_px']} px "
              f"({report['result_clothing']['outside_ratio']:.1%})",
              flush=True)

    # C) Opcional: barrido de tela fina (--refine) para localizar las tiras
    #    que el umbral 0,45 no alcanza. Solo medición: consultas diagnósticas,
    #    nunca participan del enrutado ni de la edición.
    refine_straps = None
    refine_low = None
    if args.refine:
        segmenter = manager._get_clothing_segmenter()
        if not segmenter or not segmenter.is_available():
            print("[Cobertura] AVISO: sin segmentador para --refine", flush=True)
            report["refine"] = None
        else:
            report["refine"] = {"prompts": list(REFINE_PROMPTS),
                                "tela_fina": {}, "clothing_low": {}}
            for th in args.refine_thresholds:
                mask, _arr = segmenter.segment_with_prompt(
                    result, list(REFINE_PROMPTS), threshold=th, dilation=0)
                if mask is None:
                    continue
                cand = (np.asarray(mask.convert("L")) > 127) & person
                summary = summarize(cand, zone)
                report["refine"]["tela_fina"][f"{th:g}"] = summary
                print(f"[Cobertura] TELA FINA th={th:g}: {summary['px']} px "
                      f"(dentro {summary['inside_zone_px']} / "
                      f"fuera {summary['outside_zone_px']}), "
                      f"componentes fuera={len(summary['outside_components'])}",
                      flush=True)
                if refine_straps is None:
                    refine_straps = cand
            for th in args.refine_thresholds:
                cand = clothing_mask(manager, result, threshold=th)
                if cand is None:
                    continue
                summary = summarize(cand, zone)
                report["refine"]["clothing_low"][f"{th:g}"] = summary
                print(f"[Cobertura] ROPA th={th:g}: {summary['px']} px "
                      f"(dentro {summary['inside_zone_px']} / "
                      f"fuera {summary['outside_zone_px']}), "
                      f"componentes fuera={len(summary['outside_components'])}",
                      flush=True)
                refine_low = cand  # el último (umbral más bajo) para el overlay

    # Overlay: resto fuera en ROJO, dentro en AZUL, contorno de zona en VERDE,
    # tela fina en MAGENTA, ropa de umbral bajo en CIAN.
    overlay = np.array(result)  # copia escribible (np.asarray da solo lectura)
    zone_edge = cv2.morphologyEx(
        zone.astype(np.uint8), cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).astype(bool)
    overlay[zone_edge] = (0, 255, 0)
    if res_clothes is not None:
        inside = (res_clothes & zone)
        outside = (res_clothes & ~zone)
        overlay[inside] = (255, 90, 0)   # azul en RGB
        overlay[outside] = (255, 0, 0)   # rojo en RGB
    if refine_low is not None:
        overlay[refine_low] = (0, 255, 255)  # cian: ropa a umbral bajo
    if refine_straps is not None:
        overlay[refine_straps] = (255, 0, 255)  # magenta: tela fina

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "diagnostics" / "out" / "zone_coverage_20261028" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(out_dir / "coverage_overlay.png")
    (out_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Cobertura] overlay: {out_dir / 'coverage_overlay.png'}", flush=True)
    print(f"[Cobertura] métricas: {out_dir / 'metrics.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
