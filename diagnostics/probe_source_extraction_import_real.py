"""Verify UI source extraction with local models and supplied real photographs."""

import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import numpy as np

import roop.face_util as face_util
from ui.tabs.faceswap.logic import extract_face_images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photos", type=Path, nargs="+")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--counts", type=int, nargs="+")
    args = parser.parse_args()
    if args.counts is not None and len(args.counts) != len(args.photos):
        parser.error("--counts must provide one expected count per photograph")
    for photo in args.photos:
        if not photo.is_file():
            parser.error(f"Missing photograph: {photo}")
    args.report.mkdir(parents=True, exist_ok=True)
    rows = []
    with (args.report / "extract.log").open("w", encoding="utf-8") as log:
        with redirect_stdout(log):
            analyser = face_util.get_face_analyser_cpu()
            assert analyser is not None, "Local CPU analyser failed to initialize"
            for model in analyser.models.values():
                assert model.session.get_providers() == ["CPUExecutionProvider"]
            face_util.FACE_ANALYSER = analyser
            for photo_index, photo in enumerate(args.photos):
                started = time.monotonic()
                raw = face_util.extract_face_images(str(photo), is_source_face=True)
                enriched = extract_face_images(str(photo), is_source_face=True)
                assert len(raw) == len(enriched), "UI discarded detected faces"
                if args.counts is not None:
                    assert len(enriched) == args.counts[photo_index], (photo, len(enriched), args.counts[photo_index])
                for face_index, (face, crop) in enumerate(enriched):
                    assert crop.size > 0
                    assert np.isfinite(face.embedding).all()
                    assert np.linalg.norm(face.embedding) > 0
                    assert face.area > 0
                    assert cv2.imwrite(
                        str(args.report / f"photo_{photo_index}_face_{face_index}.png"),
                        cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                    )
                row = {
                    "photo": str(photo.resolve()),
                    "raw_faces": len(raw),
                    "ui_faces": len(enriched),
                    "seconds": time.monotonic() - started,
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
    (args.report / "results.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8",
    )
    print(json.dumps(rows, indent=2), flush=True)
    assert any(row["ui_faces"] > 0 for row in rows), "No real face extracted"
    assert "Error extrayendo caras" not in (args.report / "extract.log").read_text(
        encoding="utf-8",
    ), "UI swallowed an extraction exception"


if __name__ == "__main__":
    main()
