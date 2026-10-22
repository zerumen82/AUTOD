"""CPU geometry contracts for reversible reference preparation; no inference.

Checks actual failing crop sizes (cup/head/spoon), thin images and boundary
padding. A reversible geometric mapping does not certify model adherence.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roop.img_editor.reference_canvas import prepare_reference_canvas  # noqa: E402


def photo(size, mode="RGB"):
    yy, xx = np.indices((size[1], size[0]))
    rgb = np.stack((xx % 251, yy % 241, (xx + yy) % 239), axis=-1).astype(np.uint8)
    return Image.fromarray(rgb).convert(mode)


def isotropic_budget_and_content_box():
    sizes = [(150, 174), (431, 301), (430, 377), (513, 511), (1201, 301),
             (3001, 47), (47, 3001), (1, 1), (1, 129), (129, 1), (600, 400)]
    for size in sizes:
        source = photo(size)
        encoded, mapping = prepare_reference_canvas(source, max_side=768, multiple=64)
        assert mapping.source_size == size and encoded.mode == "RGB"
        assert mapping.canvas_size == encoded.size
        assert max(encoded.size) == 768 and all(0 < axis <= 768 and axis % 64 == 0 for axis in encoded.size), (
            size, encoded.size,
        )
        x1, y1, x2, y2 = mapping.content_box
        assert 0 <= x1 < x2 <= encoded.width and 0 <= y1 < y2 <= encoded.height
        assert (x2 - x1, y2 - y1) == mapping.content_size
        # Allow integer rounding, never independent axis stretching to blocks.
        scale = 768 / max(size)
        errors = tuple(abs(content - original * scale)
                       for content, original in zip(mapping.content_size, size))
        assert max(errors) <= 1, (size, mapping.content_size, errors)
        expected = source.resize(mapping.content_size, Image.Resampling.LANCZOS)
        assert np.array_equal(np.asarray(encoded.crop(mapping.content_box)), np.asarray(expected)), (
            "content was cropped, shifted or stretched to the padded canvas", size,
        )
        assert mapping.restore(encoded).size == size


def edge_padding_preserves_content():
    source = photo((150, 174))
    encoded, mapping = prepare_reference_canvas(source, max_side=768, multiple=64)
    x1, y1, x2, y2 = mapping.content_box
    pixels = np.asarray(encoded)
    assert x1 or y1 or x2 < encoded.width or y2 < encoded.height, "fixture produced no padding"
    if x1:
        assert np.all(pixels[y1:y2, :x1] == pixels[y1:y2, x1:x1 + 1])
    if x2 < encoded.width:
        assert np.all(pixels[y1:y2, x2:] == pixels[y1:y2, x2 - 1:x2])
    if y1:
        assert np.all(pixels[:y1] == pixels[y1:y1 + 1])
    if y2 < encoded.height:
        assert np.all(pixels[y2:] == pixels[y2 - 1:y2])
    baseline = mapping.restore(encoded)
    # Simulate a model changing every padded pixel: none may leak into restore.
    damaged = np.asarray(encoded).copy()
    exterior = np.ones(damaged.shape[:2], dtype=bool)
    exterior[y1:y2, x1:x2] = False
    damaged[exterior] = (255, 0, 255)
    assert np.array_equal(np.asarray(mapping.restore(Image.fromarray(damaged))), np.asarray(baseline))


def unit_scale_round_trip_is_exact():
    for size in ((768, 701), (701, 768), (768, 768)):
        source = photo(size)
        encoded, mapping = prepare_reference_canvas(source, max_side=768, multiple=64)
        assert mapping.content_size == size
        assert np.array_equal(np.asarray(mapping.restore(encoded)), np.asarray(source)), (
            "padding-only round trip changed source pixels", size,
        )


def restore_preserves_landmark_geometry():
    for size in ((150, 174), (431, 301), (430, 377), (1201, 301)):
        source = Image.new("RGB", size)
        box = (size[0] // 5, size[1] // 5, size[0] * 4 // 5, size[1] * 4 // 5)
        ImageDraw.Draw(source).rectangle((box[0], box[1], box[2] - 1, box[3] - 1), fill="white")
        encoded, mapping = prepare_reference_canvas(source, max_side=768, multiple=64)
        restored = np.asarray(mapping.restore(encoded))[:, :, 0] >= 128
        ys, xs = np.where(restored)
        observed = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        assert max(abs(first - second) for first, second in zip(box, observed)) <= 1, (
            "content box or inverse mapping moved the landmarks", size, box, observed,
        )


def no_source_mutation():
    for mode in ("RGB", "RGBA", "L"):
        source = photo((150, 174), mode)
        original = source.tobytes()
        encoded, mapping = prepare_reference_canvas(source, max_side=768, multiple=64)
        mapping.restore(encoded)
        assert source.mode == mode and source.size == (150, 174) and source.tobytes() == original


def rejects_invalid_budget():
    source = photo((150, 174))
    invalid = [(0, 64), (-768, 64), (32, 64), (500, 64), (768, 0), (768, -64),
               (768.0, 64), (768, 64.0), ("768", 64), (768, None), (768, True)]
    for maximum, multiple in invalid:
        try:
            prepare_reference_canvas(source, max_side=maximum, multiple=multiple)
        except (TypeError, ValueError):
            continue
        raise AssertionError(f"invalid budget accepted: max_side={maximum!r}, multiple={multiple!r}")
    for size in ((0, 5), (5, 0)):
        try:
            prepare_reference_canvas(Image.new("RGB", size), max_side=768, multiple=64)
        except ValueError:
            continue
        raise AssertionError(f"empty source accepted: {size}")


def rejects_inconsistent_generated_size():
    _encoded, mapping = prepare_reference_canvas(photo((150, 174)), max_side=768, multiple=64)
    for size in ((mapping.canvas_size[0] - 1, mapping.canvas_size[1]), mapping.source_size):
        try:
            mapping.restore(Image.new("RGB", size))
        except ValueError:
            continue
        raise AssertionError(f"unexpected generated geometry silently accepted: {size}")


def main():
    cases = [isotropic_budget_and_content_box, edge_padding_preserves_content,
             unit_scale_round_trip_is_exact, restore_preserves_landmark_geometry,
             no_source_mutation, rejects_invalid_budget, rejects_inconsistent_generated_size]
    failures = []
    for check in cases:
        try:
            check()
            print(f"PASS {check.__name__}", flush=True)
        except Exception as error:
            failures.append(check.__name__)
            print(f"FAIL {check.__name__}: {error}", flush=True)
            traceback.print_exc()
    print(f"Reference canvas geometry: {len(cases) - len(failures)}/{len(cases)} PASS; failures={failures}", flush=True)
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
