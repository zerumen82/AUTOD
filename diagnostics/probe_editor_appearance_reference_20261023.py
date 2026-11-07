"""Head appearance dispatch/resolution regression; CPU doubles, not visual proof."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.probe_editor_selection_contract_20261019 import execute, fixture  # noqa: E402
from roop.img_editor import element_selection as es  # noqa: E402
from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402


def selected_person_uses_head_reference():
    run = execute(ids=["object"], prompt="The selected person must be bald")
    assert len(run.client.generations) == 1, run.message
    request = run.client.generations[0]
    assert request.get("reference_canvas") is None, "person still uses generic regional recipe"
    reference = request["image"]
    assert max(reference.size) == es.load_settings()["edit"]["maximum_reference_side"], reference.size
    assert reference.width < reference.height, "head context was replaced by full landscape photo"
    assert "apariencia" in run.message.lower() or "exterior protegido" in run.message.lower(), run.message
    before, after = np.asarray(run.source), np.asarray(run.output)
    selected_head = np.asarray(run.analysis.regions[1].mask) > 0
    assert np.any(after[selected_head] != before[selected_head]), "head was not written"
    assert np.array_equal(after[~selected_head], before[~selected_head]), "appearance changed outside measured head"


def direct_head_keeps_focal_recipe():
    run = execute(prompt="The selected person must have long hair")
    assert len(run.client.generations) == 1, run.message
    request = run.client.generations[0]
    assert request.get("reference_canvas") is None
    assert max(request["image"].size) == es.load_settings()["edit"]["maximum_reference_side"]


def soft_head_does_not_expand_write_support():
    from PIL import ImageFilter

    def soften(analysis):
        analysis.regions[1].mask = analysis.regions[1].mask.filter(ImageFilter.GaussianBlur(6))

    run = execute(ids=["object"], prompt="The selected person must be bald",
                  analysis_mutator=soften)
    assert len(run.client.generations) == 1, run.message
    core = np.asarray(run.analysis.regions[1].mask) >= 128
    before, after = np.asarray(run.source), np.asarray(run.output)
    assert np.any(after[core] != before[core])
    assert np.array_equal(after[~core], before[~core]), "soft halo became editable support"


def ambiguous_or_missing_head_keeps_object_recipe():
    from PIL import Image, ImageDraw

    def include_both(analysis):
        region = analysis.regions[2]
        region.mask = Image.new("L", analysis.size)
        ImageDraw.Draw(region.mask).rectangle((40, 35, 520, 350), fill=255)
        region.bbox = (40, 35, 521, 351)

    for ids, mutator in ((["object"], include_both), (["small-object"], None)):
        run = execute(ids=ids, prompt="The selected element must have long hair",
                      analysis_mutator=mutator)
        assert len(run.client.generations) == 1, run.message
        assert run.client.generations[0].get("reference_canvas") is not None, (
            "ambiguous/missing head was guessed")


def global_focal_has_configured_resolution():
    from diagnostics.probe_editor_kontext_people_protection_20261018 import _execute
    run = _execute()
    assert run.client.generations, run.message
    for request in run.client.generations:
        assert max(request["image"].size) == es.load_settings()["edit"]["maximum_reference_side"], request["image"].size
    before, after = np.asarray(run.source), np.asarray(run.output)
    outside = np.asarray(run.mask) == 0
    assert np.array_equal(before[outside], after[outside]), "global focal exterior changed"


def scale_does_not_shrink_large_references():
    from PIL import Image
    original = Image.new("RGB", (1200, 800))
    assert ImgEditorManager._scale_kontext_focal_reference(original) is original


def proposals_show_complete_head_before_selection():
    from PIL import Image, ImageDraw

    photo, analysis = fixture()
    # Like the real inventory, a person proposal can miss hair at the sides
    # while still covering over half of the detected face/head.
    analysis.regions[2].mask = Image.new("L", photo.size)
    ImageDraw.Draw(analysis.regions[2].mask).rectangle((390, 90, 439, 339), fill=255)
    old_head = np.asarray(analysis.regions[1].mask) >= 128
    old_person = np.asarray(analysis.regions[2].mask) >= 128
    original_small = np.asarray(analysis.regions[3].mask).copy()
    inclusion = es.load_settings()["edit"]["head_inclusion_coverage"]
    es._complete_head_proposals(analysis.regions, inclusion)
    head = np.asarray(analysis.regions[1].mask) >= 128
    person = np.asarray(analysis.regions[2].mask) >= 128
    assert (head & ~old_head).any(), "face proposal still cuts through its border"
    assert np.all(person[head]), "person proposal excludes its own complete head"
    assert (person & ~old_person).any(), "incomplete person proposal was not completed"
    assert np.array_equal(np.asarray(analysis.regions[3].mask), original_small), "unrelated object grew"
    for region in analysis.regions:
        arr = np.asarray(region.mask) >= 128
        ys, xs = np.where(arr)
        assert region.coverage == float(arr.mean())
        assert region.bbox == (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    assert es.validate_selection(photo, analysis, ["object"])[0]
    run = execute(ids=["object"], prompt="The selected person must be bald",
                  scene_factory=lambda: (photo, analysis))
    before, after = np.asarray(run.source), np.asarray(run.output)
    assert np.any(after[head & ~old_head] != before[head & ~old_head]), "complete border was not edited"
    assert np.array_equal(after[~head], before[~head]), "edit escaped the displayed head"


def completed_proposal_does_not_claim_a_neighbor():
    from PIL import Image

    _photo, analysis = fixture()
    head = np.asarray(analysis.regions[0].mask) >= 128
    completed = np.asarray(ImgEditorManager._complete_localized_region_mask(
        analysis.regions[0].mask)) >= 128
    person = np.asarray(analysis.regions[2].mask).copy()
    person[completed & ~head] = 255
    ys, xs = np.where(head)
    count = int(len(ys) * 0.4)
    person[ys[:count], xs[:count]] = 255
    analysis.regions[2].mask = Image.fromarray(person)
    inclusion = es.load_settings()["edit"]["head_inclusion_coverage"]
    assert [es._head_inside_selection(r.mask, analysis.regions[2].mask, inclusion)
            for r in analysis.regions[:2]] == [False, True]
    es._complete_head_proposals(analysis.regions, inclusion)
    assert [es._head_inside_selection(r.mask, analysis.regions[2].mask, inclusion)
            for r in analysis.regions[:2]] == [False, True], "neighbor became editable"
    protected = es.protection_mask(analysis, ["object"])
    assert protected is not None
    assert np.all(np.asarray(protected)[completed] > 0), "neighbor lost protection"


def boundary_blend_removes_background_shift_and_keeps_edit():
    from PIL import Image, ImageDraw
    from roop.img_editor.selected_region_edit import _blend_head_boundary

    yy, xx = np.indices((128, 128))
    data = np.stack((100 + xx // 8, 105 + yy // 8, 110 + (xx + yy) // 16), axis=-1).astype(np.uint8)
    source = Image.fromarray(data)
    patch_pixels = np.clip(data.astype(np.int16) + 30, 0, 255).astype(np.uint8)
    # A strong requested local recolor, separate from the unwanted background shift.
    patch_pixels[45:80, 50:80] = (220, 40, 55)
    generated = Image.fromarray(patch_pixels)
    mask = Image.new("L", source.size)
    ImageDraw.Draw(mask).ellipse((20, 10, 108, 118), fill=255)
    result = np.asarray(_blend_head_boundary(source, generated, mask, (0, 0, 128, 128)))
    support = np.asarray(mask) > 0
    assert np.array_equal(result[~support], data[~support]), "boundary blend escaped mask"
    background = support & (xx < 42) & (yy > 40) & (yy < 85)
    before = np.abs(patch_pixels.astype(float) - data).mean(axis=2)
    after = np.abs(result.astype(float) - data).mean(axis=2)
    assert after[background].mean() < before[background].mean() / 2, "bright oval remains"
    red = result[50:75, 55:75].astype(float).mean(axis=(0, 1))
    assert red[0] > red[1] * 2, "requested local color was lost"


def boundary_noop_does_not_cancel_the_edit():
    from unittest.mock import patch

    with patch("roop.img_editor.selected_region_edit._blend_head_boundary",
               side_effect=lambda image, *_args: image.copy()) as blend:
        run = execute(prompt="The selected person must be bald")
    assert blend.called
    assert np.any(np.asarray(run.source) != np.asarray(run.output)), "boundary adjustment cancelled edit"


if __name__ == "__main__":
    failures = []
    checks = (selected_person_uses_head_reference, direct_head_keeps_focal_recipe,
              soft_head_does_not_expand_write_support, ambiguous_or_missing_head_keeps_object_recipe,
              global_focal_has_configured_resolution, scale_does_not_shrink_large_references)
    checks += (proposals_show_complete_head_before_selection,)
    checks += (completed_proposal_does_not_claim_a_neighbor,)
    checks += (boundary_blend_removes_background_shift_and_keeps_edit,
               boundary_noop_does_not_cancel_the_edit)
    for check in checks:
        try:
            check()
            print(f"PASS {check.__name__}", flush=True)
        except Exception as error:
            failures.append(check.__name__)
            print(f"FAIL {check.__name__}: {error}", flush=True)
    print(f"Appearance references: {len(checks) - len(failures)}/{len(checks)} PASS", flush=True)
    raise SystemExit(bool(failures))
