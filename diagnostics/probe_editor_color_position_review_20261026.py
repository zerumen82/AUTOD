"""CPU audit of generic RGB handling, paste geometry and quality color checks.

No inference, network, models or production modifications. The synthetic
chart and circle expose acceptance limits; they do not validate an AI edit.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roop.img_editor.image_quality_pipeline import ImageQualityFinisher  # noqa: E402
from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402
from roop.img_editor.reference_canvas import prepare_reference_canvas  # noqa: E402
from roop.img_editor.top_tier_finish import _StepGate  # noqa: E402


def main():
    output = ROOT / "diagnostics/out/color_position_review_20261026" / str(time.time_ns())
    output.mkdir(parents=True, exist_ok=False)
    finisher = ImageQualityFinisher()
    pixels = np.empty((180, 300, 3), dtype=np.uint8)
    for column, color in enumerate(((180, 80, 50), (70, 160, 90), (60, 90, 170))):
        pixels[:, column * 100:(column + 1) * 100] = color
    chart = Image.fromarray(pixels)
    chart.save(output / "chart_original.png")
    round_trip = finisher._bgr_to_pil(finisher._pil_to_bgr(chart))
    rgb_exact = bool(np.array_equal(pixels, np.asarray(round_trip)))
    assert rgb_exact, "RGB/BGR round trip altered channels"

    # A deliberately wrong candidate tests the gate, not production output.
    wrong_color = Image.fromarray(pixels[:, :, ::-1].copy())
    wrong_color.save(output / "chart_deliberate_channel_swap.png")
    gate = _StepGate(finisher, "hd")
    before = gate.snapshot(chart)
    after = gate.snapshot(wrong_color)
    accepted, gate_note = gate.check(
        "audit color candidate", chart, wrong_color, before, after,
        checks=("clip", "sharp", "chroma"),
    )

    manager = object.__new__(ImgEditorManager)
    source = Image.new("RGB", (480, 320), (130, 130, 130))
    box = (60, 40, 380, 280)
    mask = Image.new("L", source.size)
    ImageDraw.Draw(mask).rectangle((60, 40, 379, 279), fill=255)
    identity = manager._integrate_local_generation(
        source, source.crop(box), mask, box, keep_measured_mask=True,
    )
    assert np.array_equal(np.asarray(source), np.asarray(identity)), "Identity paste changed pixels"

    # This input contains a circle; the wrong aspect ratio is intentional.
    patch = Image.new("RGB", (160, 40), (130, 130, 130))
    ImageDraw.Draw(patch).ellipse((68, 8, 91, 31), fill=(220, 60, 60))
    patch.save(output / "circle_wrong_aspect_input.png")
    composed = manager._integrate_local_generation(
        source, patch, mask, box, keep_measured_mask=True,
    )
    composed.save(output / "circle_composed.png")
    result = np.asarray(composed)
    changed = np.any(result != np.asarray(source), axis=2)
    outside_pixels = int(np.count_nonzero(changed & (np.asarray(mask) == 0)))
    assert outside_pixels == 0, "Paste leaked outside the measured rectangle"
    circle = (result[:, :, 0] > 180) & (result[:, :, 2] < 100)
    ys, xs = np.nonzero(circle)
    assert xs.size, "Circle disappeared"
    observed_box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

    canvas, mapping = prepare_reference_canvas(chart, max_side=768, multiple=64)
    restored = mapping.restore(canvas)
    assert restored.size == chart.size, "Reference mapping changed image dimensions"
    metrics = {
        "scope": "generic CPU audit; synthetic inputs, no AI inference or visual edit validation",
        "rgb_bgr_round_trip_exact": rgb_exact,
        "reference_restored_size": list(restored.size),
        "quality_gate_enabled": gate.enabled,
        "deliberate_channel_swap_accepted": accepted is wrong_color,
        "quality_gate_before": before,
        "quality_gate_after": after,
        "quality_gate_note": gate_note,
        "identity_paste_exact": True,
        "paste_changed_pixels_outside_mask": outside_pixels,
        "mismatched_aspect_input": list(patch.size),
        "paste_box_size": [box[2] - box[0], box[3] - box[1]],
        "circle_observed_box": observed_box,
        "circle_output_height_to_width": (observed_box[3] - observed_box[1]) / (observed_box[2] - observed_box[0]),
        "limits": [
            "Injected channel swap demonstrates a gate limit, not a production channel inversion.",
            "Wrong input aspect demonstrates stretching acceptance, not model-induced pose drift.",
            "Quality-disabled edits do not run this optional quality color step.",
        ],
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"EVIDENCE {output}", flush=True)


if __name__ == "__main__":
    main()
