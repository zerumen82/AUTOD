"""Compara el render parcial con fotogramas reales y guarda una muestra visual."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from diagnostics.probe_faceswap_tracking_20261119 import PARTIAL_LOG, video_from_log  # noqa: E402

OUT = ROOT / "diagnostics/out/faceswap_tracking_20261119"


def read_frames(path, start=0, count=33):
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(count):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Faltan frames en {path}")
        frames.append(frame)
    capture.release()
    return frames


def changed_pixels(a, b):
    diff = np.max(np.abs(a.astype(np.int16) - b.astype(np.int16)), axis=2)
    return int(np.count_nonzero(diff > 12))


def main():
    original = read_frames(video_from_log(PARTIAL_LOG), 2109)
    before = read_frames(OUT / "partial_2109_2142.mp4")
    after = read_frames(OUT / "partial_2109_2142_fixed.mp4")
    rows = []
    for index, (base, old, new) in enumerate(zip(original, before, after)):
        roi = np.s_[:260, :220]
        rows.append({
            "frame": 2109 + index,
            "old_changed": changed_pixels(base[roi], old[roi]),
            "new_changed": changed_pixels(base[roi], new[roi]),
        })
    tiles = []
    for index in (3, 4, 5, 6, 7):
        cells = []
        for label, frames in (("Original", original), ("Anterior", before), ("Corregido", after)):
            cell = frames[index][:260, :220].copy()
            cv2.putText(cell, f"{2109 + index} {label}", (6, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 2)
            cells.append(cell)
        tiles.append(np.concatenate(cells, axis=1))
    cv2.imwrite(str(OUT / "samples_partial_flicker_fix.jpg"), np.concatenate(tiles, axis=0))
    report = {"frames": rows, "every_frame_changed": all(row["new_changed"] > 100 for row in rows)}
    (OUT / "partial_flicker_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["every_frame_changed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
