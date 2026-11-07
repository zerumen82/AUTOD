"""Non-explicit CPU color regression checks and real classical-finish evidence.

Failure injection below is test-only. --photo runs the actual color/light
finish without replacing any processing, loading models or contacting ComfyUI.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roop.img_editor.image_quality_pipeline import (  # noqa: E402
    ImageQualityFinisher, quality_settings,
)
from roop.img_editor.top_tier_finish import (  # noqa: E402
    _StepGate, run_quality_enhance_after_edit, run_top_tier_quality,
)

COLOR_ONLY = {"deposter": False, "upscale": False, "color": True, "realism": False}
CHECKS = ("clip", "sharp", "chroma")


def chart():
    pixels = np.empty((120, 240, 3), dtype=np.uint8)
    for column, color in enumerate(((160, 90, 70), (80, 145, 100), (70, 100, 155))):
        pixels[:, column * 80:(column + 1) * 80] = color
    return Image.fromarray(pixels)


def lab_change(image, *, lightness=0.0, a=0.0, b=0.0, localized=False):
    lab = cv2.cvtColor(np.asarray(image, dtype=np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    affected = np.s_[:, :max(1, image.width // 10)] if localized else np.s_[:, :]
    lab[affected] += np.array([lightness, a, b], dtype=np.float32)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8))


def chroma_metrics(before, after):
    labs = [cv2.cvtColor(np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0,
                         cv2.COLOR_RGB2LAB) for im in (before, after)]
    difference = np.linalg.norm(labs[1][..., 1:] - labs[0][..., 1:], axis=-1)
    return {"mean_ab": float(difference.mean()), "p95_ab": float(np.percentile(difference, 95))}


def decide(before, candidate, config=None):
    settings = deepcopy(quality_settings()) if config is None else config
    with patch("roop.img_editor.image_quality_pipeline.quality_settings", return_value=settings):
        gate = _StepGate(ImageQualityFinisher(), "hd")
    return gate.check("color/luz", before, candidate, gate.snapshot(before),
                      gate.snapshot(candidate), checks=CHECKS)


def contracts():
    source = chart()

    def rejects_channel_swap():
        candidate = Image.fromarray(np.asarray(source)[..., ::-1].copy())
        kept, note = decide(source, candidate)
        assert kept is source and "croma" in note, note

    def rejects_moderate_cast():
        candidate = lab_change(source, a=4, b=-5)
        kept, note = decide(source, candidate)
        assert kept is source and "croma" in note, (note, chroma_metrics(source, candidate))

    def rejects_localized_cast():
        candidate = lab_change(source, a=8, b=-10, localized=True)
        kept, note = decide(source, candidate)
        assert kept is source and "croma" in note, (note, chroma_metrics(source, candidate))

    def accepts_identity_and_lightness():
        for candidate in (source.copy(), lab_change(source, lightness=4)):
            kept, note = decide(source, candidate)
            assert kept is candidate and not note, (note, chroma_metrics(source, candidate))

    def respects_configuration_and_disable():
        candidate = lab_change(source, a=4, b=-5)
        config = deepcopy(quality_settings())
        config["gate"].update({"chroma_mean_tolerance": 100.0, "chroma_p95_tolerance": 100.0})
        kept, note = decide(source, candidate, config)
        assert kept is candidate and not note, note
        config["acceptance_gate"] = False
        config["gate"].update({"chroma_mean_tolerance": -1, "chroma_max_side": 0})
        kept, note = decide(source, candidate, config)
        assert kept is candidate and not note, note

    def rejects_incompatible_geometry():
        kept, note = decide(source, source.resize((100, 100)))
        assert kept is source and "croma" in note, note

    def rejects_invalid_chroma_policy():
        for values in ({"chroma_mean_tolerance": -1}, {"chroma_p95_tolerance": float("nan")},
                       {"chroma_mean_tolerance": True}, {"chroma_max_side": 0},
                       {"chroma_max_side": 1.5}):
            config = deepcopy(quality_settings())
            config["gate"].update(values)
            kept, note = decide(source, source.copy(), config)
            assert kept is source and "croma" in note, (values, note)

    def measurement_failure_preserves_source():
        with patch("roop.img_editor.top_tier_finish._gate_chroma_shift",
                   side_effect=RuntimeError("injected measurement failure")):
            kept, note = decide(source, source.copy())
        assert kept is source and "croma no verificable" in note, note
        with patch("roop.img_editor.top_tier_finish._gate_chroma_shift",
                   return_value=(float("nan"), 0.0)):
            kept, note = decide(source, source.copy())
        assert kept is source and "croma no verificable" in note, note

    def both_finish_routes_reject_injected_cast():
        # Simulate a damaged optional color step, without altering production.
        with patch.object(ImageQualityFinisher, "enhance_color",
                          side_effect=lambda image, **kw: lab_change(image, a=4, b=-5)):
            first, note, _ = run_top_tier_quality(
                source, use_generative=False, preserve_faces=False, quality_steps=COLOR_ONLY,
            )
            second, edit_note, _ = run_quality_enhance_after_edit(
                source, source, use_generative=False, preserve_faces=False, quality_steps=COLOR_ONLY,
            )
        for kept, explanation in ((first, note), (second, edit_note)):
            assert np.array_equal(np.asarray(kept), np.asarray(source)), explanation
            assert "croma" in explanation, explanation

    def requested_recolor_survives_finish():
        edited = Image.fromarray(np.asarray(source)[..., ::-1].copy())
        finished, note, _ = run_quality_enhance_after_edit(
            edited, source, use_generative=False, preserve_faces=False, quality_steps=COLOR_ONLY,
        )
        drift = chroma_metrics(edited, finished)["mean_ab"]
        original_distance = chroma_metrics(source, finished)["mean_ab"]
        assert drift < 2.0 and original_distance > 20, (drift, original_distance, note)

    def color_off_is_exact():
        steps_off = dict.fromkeys(COLOR_ONLY, False)
        finished, _, _ = run_top_tier_quality(
            source, use_generative=False, preserve_faces=False, quality_steps=steps_off,
        )
        assert np.array_equal(np.asarray(finished), np.asarray(source))

    cases = [rejects_channel_swap, rejects_moderate_cast, rejects_localized_cast,
             accepts_identity_and_lightness, respects_configuration_and_disable,
             rejects_incompatible_geometry, rejects_invalid_chroma_policy,
             measurement_failure_preserves_source, both_finish_routes_reject_injected_cast,
             requested_recolor_survives_finish,
             color_off_is_exact]
    results = []
    for case in cases:
        try:
            case()
            results.append({"case": case.__name__, "passed": True})
            print(f"PASS {case.__name__}", flush=True)
        except Exception as error:
            results.append({"case": case.__name__, "passed": False,
                            "error": f"{type(error).__name__}: {error}"})
            print(f"FAIL {case.__name__}: {error}", flush=True)
    return results


def photo_evidence(photo, output):
    source = Image.open(photo).convert("RGB")
    source.save(output / "source.png")
    rows = []
    # Both calls use real finishing. No monkeypatching and no model inference.
    for route in ("quality", "after_edit"):
        started = time.perf_counter()
        if route == "quality":
            finished, note, _ = run_top_tier_quality(
                source, use_generative=False, preserve_faces=False, quality_steps=COLOR_ONLY,
            )
        else:
            finished, note, _ = run_quality_enhance_after_edit(
                source, source, use_generative=False, preserve_faces=False, quality_steps=COLOR_ONLY,
            )
        finished.save(output / f"{route}.png")
        assert finished.size == source.size, "Classical color finish changed geometry"
        rows.append({"route": route, "seconds": time.perf_counter() - started,
                     "chroma": chroma_metrics(source, finished), "note": note})
    # Labelled fault injection: not a spontaneous production or model defect.
    candidate = lab_change(source, a=4, b=-5)
    candidate.save(output / "injected_cast.png")
    protected, note = decide(source, candidate)
    protected.save(output / "guarded_cast.png")
    assert np.array_equal(np.asarray(protected), np.asarray(source)), note
    rows.append({"route": "controlled_fault_injection_not_model_output",
                 "candidate_chroma": chroma_metrics(source, candidate),
                 "protected_chroma": chroma_metrics(source, protected), "note": note})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photo", type=Path)
    parser.add_argument("--label", default="run")
    args = parser.parse_args()
    output = ROOT / "diagnostics/out/quality_color_guard_20261026" / f"{args.label}_{time.time_ns()}"
    output.mkdir(parents=True, exist_ok=False)
    cases = contracts()
    rows = photo_evidence(args.photo, output) if args.photo else []
    report = {"scope": "CPU classical finishing; no AI inference; non-explicit inputs",
              "cases": cases, "photos": rows,
              "failures": sum(not case["passed"] for case in cases)}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"TOTAL {len(cases) - report['failures']}/{len(cases)} PASS", flush=True)
    print(f"EVIDENCE {output}", flush=True)
    return bool(report["failures"])


if __name__ == "__main__":
    raise SystemExit(main())
