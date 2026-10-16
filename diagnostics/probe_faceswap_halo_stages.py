"""H-FS-004: mide halo/nitidez en las etapas REALES ya renderizadas del lote.

Solo lectura de artefactos existentes (debug_swap del lote 20261013_014548,
archivo f3 = 206561289): NO ejecuta inferencia ni render (el contenido es
explícito; regla de MEMORY: sin renders nuevos sobre esas salidas).

Métricas (por etapa vs. original):
  1. píxeles cambiados (absdiff > 2) dentro/fuera de la máscara final —
     fuera debe ser 0 (fuga = halo duro).
  2. nitidez (varianza de Laplaciano) en bandas concéntricas:
     cara (mask>0.9), transición (0<mask<=0.9), anillo exterior
     (dilatada 8/16/32 px sin máscara) — original vs cada etapa.
     Si el anillo exterior de la salida es mucho más suave que el
     original → halo de blur; si mucho más nítido → halo de sharpen.
  3. tamaño de la máscara vs bbox de la cara (¿la máscara cubre más
     que la cara? = elipse sobredimensionada).

Uso:
    venv\\Scripts\\python.exe diagnostics\\probe_faceswap_halo_stages.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

DEBUG = os.path.join(ROOT, "debug_swap")
OUT_DIR = os.path.join(ROOT, "diagnostics", "out", "halo_stages_20261013")


def stages_for(suffix):
    return {
        "raw": f"01_raw_model_{suffix}.png",
        "after_enhancer": f"02_after_enhancer_{suffix}.png",
        "after_color": f"03_after_color_{suffix}.png",
        "final": f"04_final_result_{suffix}.png",
    }


def laplacian_var(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    suffixes = sys.argv[1:] or ["f3"]
    report = {"frames": {}}
    for suffix in suffixes:
        report["frames"][suffix] = measure_frame(suffix)
    for suffix, data in report["frames"].items():
        v = data.get("verdict", {})
        m = data.get("mask", {})
        f = data.get("stages", {}).get("final", {})
        s = f.get("sharpness", {}).get("face_full", {})
        print(
            f"[{suffix}] mask={m.get('area_frac')} "
            f"fuga_fuera={f.get('changed_outside_mask')} px "
            f"(media={f.get('mean_abs_outside')}, dist_max={f.get('outside_max_dist_px')}) "
            f"nitidez_cara orig={s.get('orig')} -> final={s.get('stage')} "
            f"halo16={data.get('halo_outer16_ratio_final_vs_orig')} verdict={v}"
        )
    path = os.path.join(OUT_DIR, "report.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"JSON: {path}")
    print("OK")
    return 0


def measure_frame(suffix):
    orig = cv2.imread(os.path.join(DEBUG, f"07_original_frame_{suffix}.png"))
    mask = cv2.imread(os.path.join(DEBUG, f"05_mask_final_{suffix}.png"), cv2.IMREAD_GRAYSCALE)
    if orig is None or mask is None:
        return {"error": f"faltan artefactos de {suffix}"}
    h, w = orig.shape[:2]
    orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

    report = {"size": [w, h], "stages": {}}

    m01 = mask > 2
    face_area = int(m01.sum())
    full_area = int((mask > 229).sum())
    ys, xs = np.where(m01)
    mask_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    report["mask"] = {
        "area_px": face_area,
        "full_area_px": full_area,
        "area_frac": round(face_area / (w * h), 4),
        "bbox": mask_bbox,
        "bbox_w": mask_bbox[2] - mask_bbox[0],
        "bbox_h": mask_bbox[3] - mask_bbox[1],
    }

    # Bandas: cara plena, transición (feather), anillos exteriores
    rings = {}
    for dil in (8, 16, 32):
        k = np.ones((2 * dil + 1, 2 * dil + 1), np.uint8)
        outer = cv2.dilate(mask, k) > 2
        rings[f"outer_{dil}px"] = outer & ~m01
    soft = (mask > 2) & (mask <= 229)

    bands = {"face_full": mask > 229, "feather": soft}
    bands.update(rings)

    crop_prev = None
    crop_prev_name = None
    for stage, fname in stages_for(suffix).items():
        img = cv2.imread(os.path.join(DEBUG, fname))
        if img is None:
            report["stages"][stage] = {"error": f"falta {fname}"}
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.shape != orig.shape:
            # Etapa de recorte (raw/enhancer/color): se mide a sí misma y la
            # cadena contigua recorte→recorte (mismo tamaño, alineados).
            entry = {
                "kind": "crop",
                "size": [img.shape[1], img.shape[0]],
                "sharpness_crop": round(laplacian_var(gray), 1),
            }
            if crop_prev is not None and crop_prev.shape == img.shape:
                d = cv2.absdiff(img, crop_prev).max(axis=2)
                entry[f"diff_vs_{crop_prev_name}"] = {
                    "changed_px": int(np.count_nonzero(d > 2)),
                    "mean": round(float(d.mean()), 3),
                }
            report["stages"][stage] = entry
            crop_prev, crop_prev_name = img, stage
            continue
        diff = cv2.absdiff(img, orig).max(axis=2)
        changed_outside = int(np.count_nonzero(diff[~m01]))
        changed_inside = int(np.count_nonzero(diff[m01]))
        # Magnitud y distancia de la fuga exterior a la máscara
        dist_to_mask = cv2.distanceTransform((m01).astype(np.uint8), cv2.DIST_L2, 3)
        outside_diff = diff[~m01]
        changed_vals = outside_diff[outside_diff > 2]
        if changed_vals.size:
            far = (dist_to_mask[~m01])[outside_diff > 2]
            mean_outside = round(float(changed_vals.mean()), 3)
            max_dist = int(far.max())
            px_far_8 = int(np.count_nonzero(far > 8))
        else:
            mean_outside, max_dist, px_far_8 = 0.0, 0, 0
        entry = {
            "kind": "full",
            "changed_inside": changed_inside,
            "changed_outside_mask": changed_outside,
            "outside_frac": round(changed_outside / max(1, int((~m01).sum())), 5),
            "mean_abs_outside": mean_outside,
            "outside_max_dist_px": max_dist,
            "outside_changed_farther_than_8px": px_far_8,
            "mean_abs_inside": round(float(cv2.absdiff(img, orig)[m01].mean()), 3),
            "sharpness": {},
        }
        for bname, bmask in bands.items():
            if not bmask.any():
                continue
            entry["sharpness"][bname] = {
                "orig": round(laplacian_var(orig_gray[bmask]), 1),
                "stage": round(laplacian_var(gray[bmask]), 1),
            }
        report["stages"][stage] = entry
        crop_prev, crop_prev_name = None, None

    # Resumen de halo: en el anillo exterior de 16 px, ratio final/orig
    final_entry = report["stages"].get("final", {})
    halo_ratio = None
    if final_entry.get("sharpness", {}).get("outer_16px"):
        s = final_entry["sharpness"]["outer_16px"]
        halo_ratio = round(s["stage"] / max(1.0, s["orig"]), 3)
    report["halo_outer16_ratio_final_vs_orig"] = halo_ratio
    report["verdict"] = {
        "leak_outside_mask": bool(final_entry.get("changed_outside_mask", 0) > 0),
        "leak_far": bool(final_entry.get("outside_changed_farther_than_8px", 0) > 100),
        "blur_halo": bool(halo_ratio is not None and halo_ratio < 0.6),
        "sharp_halo": bool(halo_ratio is not None and halo_ratio > 1.8),
        "soft_face": bool(
            final_entry.get("sharpness", {}).get("face_full")
            and final_entry["sharpness"]["face_full"]["stage"]
            < 0.6 * final_entry["sharpness"]["face_full"]["orig"]
        ),
    }
    return report


if __name__ == "__main__":
    raise SystemExit(main())
