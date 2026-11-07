"""Measure the production identity gate with local ArcFace CPU inference.

Two supplied neutral portraits form a controlled two-person fixture. No image
generation/restoration model runs: this checks the gate's decisions on intact,
reordered, duplicated and missing identities, not face restoration quality.
Only the analyser resolver is replaced with an explicitly CPU-only analyser.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roop.img_editor import top_tier_finish as finish  # noqa: E402
from roop.img_editor.element_selection import load_settings  # noqa: E402
from roop.img_editor.image_quality_pipeline import quality_settings  # noqa: E402


def _canvas(first: Image.Image, second: Image.Image) -> Image.Image:
    cell = (max(first.width, second.width), max(first.height, second.height))
    canvas = Image.new("RGB", (cell[0] * 2, cell[1]))
    canvas.paste(ImageOps.pad(first.convert("RGB"), cell), (0, 0))
    canvas.paste(ImageOps.pad(second.convert("RGB"), cell), (cell[0], 0))
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-photo", type=Path, required=True)
    parser.add_argument("--second-photo", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "diagnostics/out/quality_identity_real_cpu_20261025")
    args = parser.parse_args()
    pack = args.model_root / "models/buffalo_l"
    required = ("det_10g.onnx", "w600k_r50.onnx")
    for name in required:
        path = pack / name
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"Local model missing; no download allowed: {path}")

    from insightface.app import FaceAnalysis

    detector = load_settings()["detector"]
    app = FaceAnalysis(name="buffalo_l", root=str(args.model_root),
                       allowed_modules=["detection", "recognition"],
                       providers=["CPUExecutionProvider"])
    side = int(detector["input_size"])
    app.prepare(ctx_id=-1, det_size=(side, side), det_thresh=float(detector["threshold"]))
    providers = {name: model.session.get_providers() for name, model in app.models.items()}
    assert providers and all(value == ["CPUExecutionProvider"] for value in providers.values()), providers
    settings = quality_settings()
    assert settings["identity"]["verify_after_face_restore"] is True
    first = Image.open(args.first_photo).convert("RGB")
    second = Image.open(args.second_photo).convert("RGB")
    original = _canvas(first, second)
    assert max(original.size) <= int(settings["identity"]["max_side"]), original.size
    blank = Image.new("RGB", second.size)
    cases = (
        ("intact", original.copy(), True),
        ("reordered", _canvas(second, first), True),
        ("duplicated_first", _canvas(first, ImageOps.pad(first, second.size)), False),
        ("missing_second", _canvas(first, blank), False),
    )
    out = args.output_root / str(time.time_ns())
    out.mkdir(parents=True, exist_ok=False)
    original.save(out / "source.png")
    report = {
        "scope": "Real local CPU face detection + ArcFace + production identity gate; no diffusion or face restoration",
        "providers": providers, "detector": detector, "settings": settings["identity"],
        "sources": [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for path in (args.first_photo, args.second_photo)],
        "cases": [],
    }
    with patch("roop.face_util.get_face_analyser", return_value=app):
        reference = finish._face_embeddings(original)
        assert reference is not None and len(reference) == 2, "Fixture must contain two detected faces"
        for label, candidate, expected_accept in cases:
            assert candidate.size == original.size, "Identity fixtures must preserve canvas geometry"
            started = time.monotonic()
            path = out / f"{label}.png"
            candidate.save(path)
            evidence = finish._face_embeddings(candidate)
            matched = finish._matched_face_cosines(reference, evidence)
            old_global_maximum = float(np.max(reference @ evidence.T)) if evidence is not None else None
            result, reason = finish._faces_keep_identity(original, original, candidate, settings)
            accepted = result is candidate and reason == ""
            report["cases"].append({
                "label": label, "file": path.name, "faces": 0 if evidence is None else len(evidence),
                "old_global_maximum": old_global_maximum,
                "individual_cosines": None if matched is None else matched.tolist(),
                "accepted": accepted, "expected_accept": expected_accept,
                "pass": accepted == expected_accept, "reason": reason,
                "seconds": time.monotonic() - started,
            })
    report["pass"] = sum(case["pass"] for case in report["cases"])
    report["fail"] = len(cases) - report["pass"]
    (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "pass": report["pass"], "fail": report["fail"],
                      "cases": report["cases"]}, ensure_ascii=True))
    return bool(report["fail"])


if __name__ == "__main__":
    raise SystemExit(main())
