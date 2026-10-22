"""CPU contract: a head edit never touches a nearby unselected person.

Two heads overlap in the measured geometry, so the reference is blinded for the
unselected head and the crop is tightened. The blind patch must not erase the
selected head, and the grey patch must never reach the final photo.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.probe_editor_selection_contract_20261019 import (  # noqa: E402
    execute, fixture,
)
from roop.img_editor import element_selection as es  # noqa: E402


def overlapping_scene():
    """Two heads whose measured ellipses overlap by thousands of pixels."""
    photo, analysis = fixture()
    other = Image.new("L", photo.size)
    ImageDraw.Draw(other).ellipse((360, 60, 470, 180), fill=255)
    selected = Image.new("L", photo.size)
    ImageDraw.Draw(selected).ellipse((395, 65, 505, 185), fill=255)
    analysis.regions[0].mask = other
    analysis.regions[0].bbox = (360, 60, 470, 180)
    analysis.regions[1].mask = selected
    analysis.regions[1].bbox = (395, 65, 505, 185)
    return photo, analysis


def overlap_is_real():
    photo, analysis = overlapping_scene()
    other = np.asarray(analysis.regions[0].mask) >= 128
    chosen = np.asarray(analysis.regions[1].mask) >= 128
    shared = int((other & chosen).sum())
    assert shared > 1000, f"fixture requires overlapping heads, got {shared}"
    return photo, analysis, other, chosen


def blind_preserves_selected_head():
    photo, analysis, other, chosen = overlap_is_real()
    settings = es.load_settings()
    blinded = es.blind_unselected_in_reference(
        analysis, ["head-right"], photo,
        dilate_radius=int(settings["edit"]["blind_dilate_radius"]),
        keep_mask=es.union_mask(analysis, ["head-right"]),
    )
    arr = np.asarray(blinded.convert("RGB"))
    gray = tuple(int(v) for v in settings["edit"]["blind_fill"])
    # The selected head keeps its original pixels.
    assert np.array_equal(arr[chosen], np.asarray(photo)[chosen]), "blinding erased the selected head"
    # The unselected head is greyed out.
    unselected_only = other & ~chosen
    assert unselected_only.any(), "fixture requires a non shared unselected area"
    assert np.all(arr[unselected_only] == gray), "unselected head was not blinded"


def blind_reports_no_change_when_nothing_to_blind():
    photo, analysis = fixture()
    blinded = es.blind_unselected_in_reference(analysis, ["head-left", "head-right"], photo, 6)
    assert np.array_equal(np.asarray(blinded), np.asarray(photo)), "selected heads were blinded"


def crop_is_tightened_for_head():
    run = execute(["head-right"])
    settings = es.load_settings()["edit"]
    assert settings["head_crop_context_fraction"] < settings["crop_context_fraction"], (
        "fixture requires a tighter head context than the generic one"
    )
    reference = run.client.generations[0].get("image", run.client.generations[0].get("source_image"))
    selected = np.asarray(run.analysis.regions[1].mask) >= 128
    ys, xs = np.where(selected)
    box_width = int(xs.max() - xs.min()) + 1
    expected = int(round((box_width + 2 * box_width * settings["head_crop_context_fraction"])))
    assert reference.width <= expected + 2, (
        f"head crop kept the wide generic context: {reference.width} > {expected}"
    )


def grey_patch_never_reaches_the_photo():
    run = execute(["head-right"], scene_factory=overlapping_scene)
    settings = es.load_settings()["edit"]
    gray = np.array([int(v) for v in settings["blind_fill"]], dtype=np.uint8)
    out = np.asarray(run.output)
    source = np.asarray(run.source)
    changed = np.any(out != source, axis=2)
    assert changed.any(), "the selected region did not change"
    # Only the selected support may change; a grey pixel outside it is a leak.
    support = np.asarray(es.union_mask(run.analysis, ["head-right"])) > 0
    leaked = changed & ~support
    assert not leaked.any(), "pixels outside the selection changed"
    # A grey pixel outside the support only counts as a leak when the source
    # pixel was not grey already.
    grey_leak = np.all(out == gray, axis=2) & ~support & ~np.all(source == gray, axis=2)
    assert not grey_leak.any(), "grey blind patch leaked into the photo"


def main():
    cases = [
        ("blind_preserves_selected_head", blind_preserves_selected_head),
        ("blind_reports_no_change_when_nothing_to_blind", blind_reports_no_change_when_nothing_to_blind),
        ("crop_is_tightened_for_head", crop_is_tightened_for_head),
        ("grey_patch_never_reaches_the_photo", grey_patch_never_reaches_the_photo),
    ]
    failures = []
    for name, check in cases:
        try:
            check()
            print(f"PASS {name}", flush=True)
        except Exception as error:
            failures.append(name)
            print(f"FAIL {name}: {error}", flush=True)
            traceback.print_exc()
    print(f"Nearby head protection: {len(cases) - len(failures)}/{len(cases)} PASS; failures={failures}", flush=True)
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())