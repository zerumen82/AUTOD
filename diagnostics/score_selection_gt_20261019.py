"""Score measured selector masks against the sparse manual GT positive cores.

Diagnostic only. The cores are conservative interior samples of the SOURCE
photo; they establish per-part core recall and cross-object core contamination,
never whole-object precision or IoU. No mask here is fed back into production,
conditioning or any render. Local files only, CPU, no downloads.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GT = ROOT / "diagnostics/out/editor_selection_gt_20261019/coffee_v1/annotations.json"
DEFAULT_CANDIDATES = ROOT / "diagnostics/out/editor_selection_multiscale_20261019"

# GT object -> production region label used by the real inventory (coffee photo).
OBJECT_TO_LABEL = {"cup": "coffee cup", "spoon": "silver spoon",
                   "table": "wooden table", "saucer": "saucer"}


def polygon_mask(polygon, size):
    image = Image.new("L", size, 0)
    ImageDraw.Draw(image).polygon([tuple(point) for point in polygon], fill=255)
    return np.asarray(image) > 0


def load_cores(gt):
    size = tuple(gt["source_size"])
    cores = []
    for part in gt["parts"]:
        mask = polygon_mask(part["polygon"], size)
        declared = int(part["positive_core_pixels"])
        if int(mask.sum()) != declared:
            raise ValueError(
                f"Core {part['name']}: polygon gives {int(mask.sum())} px, "
                f"annotations declare {declared}")
        cores.append({"name": part["name"], "object": part["object"],
                      "mask": mask, "pixels": declared})
    return cores


def score_mask(mask, cores, own_object):
    """Core recall of the mask's own object + hits on every foreign core."""
    own, other = {}, {}
    for core in cores:
        hits = float((mask & core["mask"]).sum()) / core["pixels"]
        record = own if core["object"] == own_object else other
        record[core["name"]] = round(hits, 4)
    return own, other


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES,
                        help="Multiscale evidence root containing */results.json")
    parser.add_argument("--production", type=Path, default=ROOT / "diagnostics/out/editor_selection_20261019/1792422658379575700",
                        help="Production run directory with object-*.png masks")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    gt = json.loads(args.gt.read_text(encoding="utf-8"))
    if gt.get("dense_ground_truth") is not False:
        raise ValueError("Only sparse-core annotations are accepted")
    cores = load_cores(gt)

    report = {"gt": str(args.gt), "annotation_type": gt["annotation_type"],
              "scope": ("core recall and cross-object core contamination only; "
                        "sparse cores cannot establish IoU/precision/full recall"),
              "scored_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "production": {}, "multiscale": {}}

    # --- production masks (object-*.png of a real run on the same photo) -----
    if args.production and args.production.is_dir():
        # match by region results.json labels for correctness
        results = json.loads((args.production / "results.json").read_text(encoding="utf-8"))
        by_label = {region["label"]: region["id"] for region in results["regions"]}
        for object_name, label in OBJECT_TO_LABEL.items():
            region_id = by_label.get(label)
            if region_id is None:
                report["production"][object_name] = {"missing_label": label}
                continue
            mask = np.asarray(Image.open(args.production / f"{region_id}.png").convert("L")) > 127
            own, other = score_mask(mask, cores, object_name)
            report["production"][object_name] = {
                "label": label, "mask_px": int(mask.sum()),
                "core_recall": own, "foreign_core_hits": other}

    # --- multiscale candidates ----------------------------------------------
    multiscale_root = args.candidates
    runs = sorted(p for p in multiscale_root.glob("*/results.json"))
    if not runs:
        raise FileNotFoundError(f"No multiscale results under {multiscale_root}")
    for run_path in runs:
        data = json.loads(run_path.read_text(encoding="utf-8"))
        coffee = data.get("photos", {}).get("coffee")
        if not coffee:
            continue
        entry = {"run": str(run_path.parent), "labels": {}}
        for label, candidates in coffee["labels"].items():
            object_name = next((o for o, region_label in OBJECT_TO_LABEL.items()
                                if region_label == label), None)
            if object_name is None:
                continue
            scored = []
            for record in candidates:
                mask_path = run_path.parent / "coffee" / f"{record['heat'].replace('_heat.npy', '_mask.png')}"
                if not mask_path.is_file():
                    continue
                mask = np.asarray(Image.open(mask_path).convert("L")) > 127
                own, other = score_mask(mask, cores, object_name)
                mean_recall = round(float(np.mean(list(own.values()))), 4)
                max_foreign = round(max(other.values()), 4) if other else 0.0
                scored.append({"query": record["query"], "method": record["method"],
                               "context_fraction": record.get("context_fraction"),
                               "mask_px": int(mask.sum()),
                               "core_recall": own, "mean_core_recall": mean_recall,
                               "foreign_core_hits": other,
                               "max_foreign_core_hit": max_foreign})
            best = max(scored, key=lambda r: (r["mean_core_recall"] - r["max_foreign_core_hit"]),
                       default=None)
            entry["labels"][label] = {"object": object_name, "candidates": scored,
                                      "best": best}
        report["multiscale"][str(run_path.parent)] = entry

    # --- concise console verdict -------------------------------------------
    print("PRODUCTION core recall (own object) and foreign contamination:")
    for object_name, record in report["production"].items():
        if "missing_label" in record:
            print(f"  {object_name:8} MISSING label {record['missing_label']!r}")
            continue
        recalls = [v for k, v in record["core_recall"].items()]
        foreign = [f"{k}={v}" for k, v in record["foreign_core_hits"].items() if v >= 0.01]
        print(f"  {object_name:8} recall={record['core_recall']} "
              f"mean={round(float(np.mean(recalls)), 3)} foreign={{{', '.join(foreign)}}}")

    for run, entry in report["multiscale"].items():
        print(f"MULTISCALE {Path(run).name}")
        for label, record in entry["labels"].items():
            best = record["best"]
            if not best:
                print(f"  {label:14} no candidates scored")
                continue
            baseline = next((c for c in record["candidates"]
                             if c["method"] == "global" and c["context_fraction"] is None
                             and c["query"] == best["query"]), None)
            delta = (round(best["mean_core_recall"] - baseline["mean_core_recall"], 4)
                     if baseline else None)
            print(f"  {label:14} best={best['query']!r} {best['method']} "
                  f"ctx={best['context_fraction']} recall_mean={best['mean_core_recall']} "
                  f"foreign={best['max_foreign_core_hit']} delta_vs_own_global={delta}")

    destination = args.output or (ROOT / "diagnostics/out/editor_selection_gt_20261019" / "score_report.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"REPORT {destination}")


if __name__ == "__main__":
    main()
