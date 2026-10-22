"""Render one manually reviewed real SAM2 alternative; never a selection policy.

The CLI choice is post hoc and diagnostic. It must not be interpreted as an
automatic admission rule or as full-object validation. No annotated pixels are
used to build masks, prompts or conditioning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roop.img_editor.element_selection import image_fingerprint  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("candidate", help="Exact filename of the manually reviewed alternative")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    sam = json.loads(args.candidates.read_text(encoding="utf-8"))
    record, = [item for item in sam["candidates"] if item["mask"] == args.candidate]
    if not record["region_id"]:
        raise ValueError("This comparison requires a real source inventory region, not an invented grid label")
    analysis_path = ROOT / sam["analysis"]
    original = json.loads(analysis_path.read_text(encoding="utf-8"))
    source = ROOT / sam["source"]
    photo = Image.open(source).convert("RGB")
    assert sam["fingerprint"] == original["fingerprint"] == image_fingerprint(photo)
    candidate_path = args.candidates.parent / record["mask"]
    candidate = Image.open(candidate_path).convert("L")
    assert candidate.size == photo.size
    mask = np.asarray(candidate) > 0
    assert mask.any()
    target = ROOT / "diagnostics/out/editor_sam2_manual_analysis_20261019" / str(time.time_ns())
    target.mkdir(parents=True, exist_ok=False)
    for region in original["regions"]:
        if region["id"] == record["region_id"]:
            candidate.save(target / f"{region['id']}.png")
            ys, xs = np.where(mask)
            region["bbox"] = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
            region["coverage"] = float(mask.mean())
            region["components"] = cv2.connectedComponents(mask.astype(np.uint8))[0] - 1
        else:
            with Image.open(analysis_path.parent / f"{region['id']}.png") as unchanged:
                unchanged.save(target / f"{region['id']}.png")
    original["message"] = "Manual post-hoc SAM2 candidate for diagnostic render only"
    original["sam2_provenance"] = {
        "scope": "Manual post-hoc selection; NOT an automatic admission algorithm",
        "source_analysis": str(analysis_path.resolve()),
        "sam2_candidates": str(args.candidates.resolve()),
        "sam2_report_sha256": hashlib.sha256(args.candidates.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        "candidate": record,
        "annotation_pixels_used_in_inference": False,
    }
    analysis_file = target / "results.json"
    analysis_file.write_text(json.dumps(original, indent=2), encoding="utf-8")
    print("MANUAL DIAGNOSTIC ANALYSIS", analysis_file, flush=True)
    from diagnostics.probe_editor_selection_real_20261019 import main as real_probe
    sys.argv = [sys.argv[0], str(source), "--reuse-analysis", str(analysis_file),
                "--select", record["region_id"], "--prompt", args.prompt, "--seed", str(args.seed)]
    real_probe()


if __name__ == "__main__":
    main()
