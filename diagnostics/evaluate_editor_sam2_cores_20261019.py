"""Evaluate saved SAM2 masks against frozen sparse source cores; no inference.

All candidates remain in their recorded order. No candidate is selected,
conditioned, admitted, or fed back to production using these annotations.
Unknown pixels cannot establish full-object precision, recall or IoU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def rgb_hash(image):
    digest = hashlib.sha256(f"RGB:{image.width}:{image.height}:".encode("ascii"))
    digest.update(image.convert("RGB").tobytes())
    return digest.hexdigest()


def overlay(source, binary):
    fill = np.zeros((source.height, source.width, 4), dtype=np.uint8)
    fill[binary] = (25, 255, 100, 82)
    rendered = Image.alpha_composite(source.convert("RGBA"), Image.fromarray(fill)).convert("RGB")
    draw = ImageDraw.Draw(rendered)
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        points = [tuple(point) for point in contour.reshape(-1, 2).tolist()]
        if len(points) > 1:
            draw.line(points + points[:1], fill=(255, 10, 180), width=1)
    return rendered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--truth", type=Path,
                        default=ROOT / "diagnostics/out/editor_selection_gt_20261019/coffee_v1")
    parser.add_argument("--disputed-part", action="append", default=[],
                        help="Keep literal metrics but flag a frozen annotation whose source meaning is disputed")
    args = parser.parse_args()
    records = json.loads((args.run / "results.json").read_text(encoding="utf-8"))
    annotation_file = args.truth / "annotations.json"
    # The frozen writer hashes its UTF-8/LF JSON payload before Windows text
    # output converts newlines to CRLF. Validate that original canonical hash;
    # keep the frozen file and its checksum untouched.
    payload = annotation_file.read_text(encoding="utf-8").encode("utf-8")
    annotation_hash = hashlib.sha256(payload).hexdigest()
    assert annotation_hash == (args.truth / "annotations.sha256").read_text().strip()
    annotation = json.loads(payload)
    source = Image.open(args.run / "source.png").convert("RGB")
    assert rgb_hash(source) == annotation["source_rgb_fingerprint"] == records["fingerprint"]
    assert not annotation["dense_ground_truth"], "This evaluator only supports sparse source cores"
    cores = {part["name"]: np.asarray(Image.open(args.truth / f"core_{part['name']}.png")) > 0
             for part in annotation["parts"]}
    rows, groups = [], {}
    for record in records["candidates"]:
        binary = np.asarray(Image.open(args.run / record["mask"])) > 0
        assert binary.shape == (source.height, source.width), record["mask"]
        count, _labels = cv2.connectedComponents(binary.astype(np.uint8))
        metrics = {name: {"core_pixels": int(core.sum()),
                          "covered_pixels": int((binary & core).sum()),
                          "core_coverage": float(binary[core].mean())}
                   for name, core in cores.items()}
        rows.append({**record, "measured_mask_pixels": int(binary.sum()),
                     "measured_components": count - 1, "sparse_core_metrics": metrics})
        rendered = overlay(source, binary)
        rendered.save(args.run / f"tester_{Path(record['mask']).stem}_overlay.png")
        rendered.thumbnail((300, 200), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (300, 234), "white")
        tile.paste(rendered, (0, 34))
        text = ImageDraw.Draw(tile)
        text.text((3, 2), f"{Path(record['mask']).stem} {record['method']}", fill="black")
        text.text((3, 17), f"IoU predicted={record['model_iou']:.3f}; {record.get('label') or 'unlabelled'}", fill="black")
        groups.setdefault(str(record.get("region_id") or "grid"), []).append(tile)
    for name, tiles in groups.items():
        montage = Image.new("RGB", (1200, ((len(tiles) + 3) // 4) * 234), "white")
        for index, tile in enumerate(tiles):
            montage.paste(tile, ((index % 4) * 300, (index // 4) * 234))
        montage.save(args.run / f"tester_{name}_candidates.png")
    report = {
        "scope": "Saved-mask geometry only; sparse core coverage is not full-object recall/precision/IoU",
        "frozen_annotation_sha256": annotation_hash,
        "annotation_hash_scheme": "UTF-8 text with LF newlines, as frozen by the annotation writer",
        "annotation_file_bytes_sha256": hashlib.sha256(annotation_file.read_bytes()).hexdigest(),
        "source_rgb_fingerprint": rgb_hash(source),
        "inference_executed": False,
        "selection_policy": "None. All recorded candidates are retained; no admission decision uses these cores.",
        "disputed_core_names": args.disputed_part,
        "disputed_core_policy": "Literal counts retained for audit; do not use these cores to declare candidate completeness/failure.",
        "parts": annotation["parts"], "candidates": rows,
    }
    target = args.run / "tester_sparse_core_metrics.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Evaluated {len(rows)} saved candidates against frozen sparse cores: {target}")
    for row in rows:
        if row["mask"] in ("candidate_0012.png", "candidate_0014.png", "candidate_0039.png", "candidate_0066.png"):
            print(row["mask"], {name: round(100 * value["core_coverage"], 2)
                                 for name, value in row["sparse_core_metrics"].items()})


if __name__ == "__main__":
    main()
