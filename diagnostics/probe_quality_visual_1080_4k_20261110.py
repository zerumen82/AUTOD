# -*- coding: utf-8 -*-
"""Ejecuta el Modo Calidad a 1080p y 4K sobre una foto real y deja la evidencia.

No declara PASS: guarda original, resultados, recortes al 100 % y un JSON con
medidas (tamaño, nitidez, bandas, recorte, croma, identidad si hay caras). El
veredicto visual lo firma el usuario en el propio JSON o en la UI.

Uso:
  python diagnostics/probe_quality_visual_1080_4k_20261110.py --dry-run
  python diagnostics/probe_quality_visual_1080_4k_20261110.py --photo ruta/foto.jpg
  python diagnostics/probe_quality_visual_1080_4k_20261110.py --photo foto.jpg --tier 4k --no-faces
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _metrics(image: Image.Image, original: Image.Image) -> dict:
    import roop.img_editor.top_tier_finish as ttf

    comparable = original
    if comparable.size != image.size:
        comparable = comparable.resize(image.size, Image.LANCZOS)
    chroma_mean = chroma_p95 = None
    try:
        chroma_mean, chroma_p95 = ttf._gate_chroma_shift(comparable, image, 1024)
    except Exception:
        pass
    return {
        "size": list(image.size),
        "sharpness": round(ttf._gate_sharpness(image), 2),
        "sharpness_reference": round(ttf._gate_sharpness(comparable), 2),
        "banding": round(ttf._gate_banding(image), 3),
        "banding_reference": round(ttf._gate_banding(comparable), 3),
        "clipping": round(ttf._gate_clipping(image), 5),
        "clipping_reference": round(ttf._gate_clipping(comparable), 5),
        "chroma_mean_vs_reference": (
            round(chroma_mean, 2) if chroma_mean is not None else None
        ),
        "chroma_p95_vs_reference": (
            round(chroma_p95, 2) if chroma_p95 is not None else None
        ),
    }


def _crops(image: Image.Image, output: Path, stem: str) -> list[str]:
    """Tres recortes 512 al 100 %: centro, zona superior y tercio derecho."""
    width, height = image.size
    side = min(512, width, height)
    boxes = {
        "center": (
            max(0, width // 2 - side // 2),
            max(0, height // 2 - side // 2),
        ),
        "upper": (max(0, width // 2 - side // 2), max(0, height // 6)),
        "right": (max(0, width - side - width // 12), max(0, height // 2 - side // 2)),
    }
    saved = []
    for name, (x, y) in boxes.items():
        crop = image.crop((x, y, x + side, y + side))
        path = output / f"{stem}_{name}_100pct.png"
        crop.save(path)
        saved.append(str(path.relative_to(ROOT)).replace("\\", "/"))
    return saved


def _identity(original: Image.Image, result: Image.Image) -> dict:
    try:
        import roop.img_editor.top_tier_finish as ttf

        ref = ttf._face_embeddings(original)
        cand = ttf._face_embeddings(result)
        cosines = ttf._matched_face_cosines(ref, cand)
        if cosines is None:
            return {"faces": 0, "cosines": None}
        return {
            "faces": int(len(cosines)),
            "cosines": [round(float(v), 4) for v in np.atleast_1d(cosines)],
        }
    except Exception as exc:
        return {"faces": 0, "cosines": None, "note": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo", help="foto real de entrada")
    parser.add_argument("--tier", choices=["hd", "4k", "both"], default="both")
    parser.add_argument(
        "--out", default="diagnostics/out/quality_visual_20261110",
    )
    parser.add_argument("--no-faces", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from roop.img_editor.image_quality_pipeline import limit_quality_tier

    tiers = ["hd", "4k"] if args.tier == "both" else [args.tier]
    output = ROOT / args.out
    output.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        photo = None
        if args.photo:
            photo = Image.open(ROOT / args.photo if not Path(args.photo).is_absolute() else args.photo).convert("RGB")
        size = photo.size if photo is not None else (1600, 1200)
        plan = {}
        for tier in tiers:
            effective, note = limit_quality_tier(
                photo if photo is not None else Image.new("RGB", size), tier,
            )
            plan[tier] = {"effective": effective, "note": note}
        print(json.dumps({
            "dry_run": True,
            "input": list(size),
            "tiers": plan,
            "faces": not args.no_faces,
            "output": str(output),
            "nota": "No ejecuta el acabado ni declara PASS visual.",
        }, indent=2, ensure_ascii=False))
        return 0

    if not args.photo:
        print("Falta --photo (o usa --dry-run para ver el plan)", flush=True)
        return 2
    photo_path = Path(args.photo)
    if not photo_path.is_absolute():
        photo_path = ROOT / photo_path
    if not photo_path.is_file():
        print(f"No existe la foto: {photo_path}", flush=True)
        return 2

    original = Image.open(photo_path).convert("RGB")
    original.save(output / "input.png")
    from roop.img_editor.top_tier_finish import run_top_tier_quality

    report = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "photo": str(photo_path).replace("\\", "/"),
        "input_size": list(original.size),
        "faces": not args.no_faces,
        "runs": {},
        "visual_verdict": {
            "status": "PENDIENTE",
            "reviewed_by": None,
            "notes": (
                "No se declara PASS. Revisa los recortes al 100 %: bandas, halo "
                "facial, identidad, textura inventada y tamaño."
            ),
            "criteria": {
                "bandas_y_bloques": None,
                "halo_o_ovalos": None,
                "identidad_facial": None,
                "textura_artificial": None,
                "nitidez_y_contraste": None,
                "resolucion_final": None,
            },
        },
    }
    for tier in tiers:
        effective, note = limit_quality_tier(original, tier)
        start = time.perf_counter()
        result, finish_note, analysis = run_top_tier_quality(
            original,
            enhance_tier=tier,
            preserve_faces=not args.no_faces,
            progress_callback=lambda p: print(
                f"[{tier}] {p.get('phase')} {p.get('progress')}", flush=True,
            ),
        )
        elapsed = time.perf_counter() - start
        stem = f"{tier}_{effective}"
        saved = output / f"{stem}.png"
        result.save(saved)
        run = {
            "requested_tier": tier,
            "effective_tier": effective,
            "limit_note": note,
            "seconds": round(elapsed, 2),
            "finish_note": finish_note,
            "analysis": {
                "scores": analysis.get("scores", {}),
                "final_check": analysis.get("final_check"),
            },
            "metrics": _metrics(result, original),
            "identity": _identity(original, result),
            "result": str(saved.relative_to(ROOT)).replace("\\", "/"),
            "crops_100pct": _crops(result, output, stem),
        }
        report["runs"][tier] = run
        print(f"[{tier}] {result.size} {elapsed:.1f}s | {finish_note}", flush=True)

    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"[Validacion] evidencia en {output}", flush=True)
    print(
        "[Validacion] PENDIENTE de revisión visual: no se declara PASS. "
        "Rellena visual_verdict en report.json tras revisar los recortes.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
