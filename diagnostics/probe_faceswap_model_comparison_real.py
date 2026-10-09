"""Plot comparable CPU FaceSwap artifacts without rerendering any model.

The crop uses the measured target bbox from the identity diagnostic. Different
source, target, alignment, enhancer or device settings are rejected explicitly.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("identity_json")
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--margin", type=float, default=.08)
    args = parser.parse_args()
    assert args.width > 0 and args.margin >= 0
    identity = json.loads(Path(args.identity_json).read_text(encoding="utf-8"))
    entries = [(Path(path), json.loads((Path(path) / "meta.json").read_text(encoding="utf-8")))
               for path in args.runs]
    reference = entries[0][1]
    assert identity["source"] == reference["source"] and identity["target"] == reference["target"]
    for _path, metadata in entries:
        for key in ("source", "target", "landmarks", "enhancer", "enhancer_blend", "device"):
            assert metadata[key] == reference[key], f"Unmatched {key}: cannot attribute differences to model"
        assert metadata["device"] == "cpu"
        assert np.array_equal(np.asarray(metadata["kps"]), np.asarray(reference["kps"])), "Different geometry"
        assert metadata["measurements"][1]["onnx_calls"] > 0, "Model never ran"
    bbox = np.asarray(identity["target_bbox"], float)
    border = float(np.max(bbox[2:] - bbox[:2])) * args.margin
    original = Image.open(reference["target"]).convert("RGB")
    x0, y0 = np.maximum(np.floor(bbox[:2] - border).astype(int), 0)
    x1, y1 = np.minimum(np.ceil(bbox[2:] + border).astype(int), original.size)
    assert x1 > x0 and y1 > y0
    bounds = (int(x0), int(y0), int(x1), int(y1))
    height = round((y1 - y0) * args.width / (x1 - x0))
    rendered = [("original", Path(reference["target"])),
                ("baseline128", entries[0][0] / "baseline128.png")]
    rendered += [(Path(metadata["model"]).stem, path / "hyperswap256.png") for path, metadata in entries]
    panels = []
    for label, path in rendered:
        photo = Image.open(path).convert("RGB")
        assert photo.size == original.size
        panel = Image.new("RGB", (args.width, height + 30), "white")
        panel.paste(photo.crop(bounds).resize((args.width, height), Image.Resampling.LANCZOS), (0, 30))
        ImageDraw.Draw(panel).text((5, 5), label, fill="black")
        panels.append(np.asarray(panel))
    output = Path(__file__).resolve().parents[1] / "diagnostics/out" / f"faceswap_model_comparison_{time.time_ns()}"
    output.mkdir()
    Image.fromarray(np.concatenate(panels, axis=1)).save(output / "comparison.png")
    source = Image.open(identity["source"]).convert("RGB")
    source.crop(tuple(int(value) for value in identity["source_bbox"])).save(output / "source_face.png")
    (output / "meta.json").write_text(json.dumps({"identity_json": args.identity_json,
        "runs": args.runs, "labels": [label for label, _path in rendered], "crop": bounds,
        "source": identity["source"], "target": identity["target"],
        "width": args.width, "margin": args.margin}, indent=2), encoding="utf-8")
    print(output, flush=True)


if __name__ == "__main__":
    main()
