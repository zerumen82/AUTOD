"""CPU contracts for H-EDIT-SELECT; these do not certify visual adherence.

Real validation, crop, dispatch and composition run against adversarial inference
doubles. No model is loaded, downloaded, or installed by this diagnostic.
"""
from __future__ import annotations

import argparse
import copy
import sys
import traceback
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roop.img_editor import element_selection as es  # noqa: E402
from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402


def fixture():
    yy, xx = np.indices((420, 640))
    data = np.stack((70 + xx % 71, 80 + yy % 91, 95 + (xx + yy) % 101), axis=-1)
    photo = Image.fromarray(data.astype(np.uint8), "RGB")
    regions = []
    for ident, kind, box in (
        ("head-left", "head", (80, 70, 150, 165)),
        ("head-right", "head", (380, 75, 450, 170)),
        ("object", "object", (330, 40, 500, 340)),
        ("small-object", "object", (265, 220, 300, 260)),
    ):
        mask = Image.new("L", photo.size)
        ImageDraw.Draw(mask).rectangle((box[0], box[1], box[2] - 1, box[3] - 1), fill=255)
        regions.append(es.SelectionRegion(
            id=ident, label=ident, mask=mask, bbox=box, kind=kind,
            coverage=float((np.asarray(mask) > 0).mean()),
        ))
    return photo, es.SelectionAnalysis(es.image_fingerprint(photo), photo.size, regions)


def fingerprint_exact():
    photo = Image.new("RGB", (1024, 1024), (120, 130, 140))
    changed = photo.copy()
    changed.putpixel((513, 519), (121, 130, 140))
    assert es.image_fingerprint(photo) != es.image_fingerprint(changed), "one RGB pixel escaped fingerprint"
    assert es.image_fingerprint(photo) != es.image_fingerprint(photo.resize((512, 512))), "size escaped fingerprint"
    assert es.image_fingerprint(photo) == es.image_fingerprint(photo.convert("RGBA")), "canonical RGB mismatch"


def validation_rejects_bad_state():
    photo, analysis = fixture()
    assert es.validate_selection(photo, analysis, ["head-right"])[0]
    invalid = []
    stale = copy.deepcopy(analysis)
    stale.fingerprint = "stale"
    invalid.append((photo, stale, ["head-right"], "stale photo"))
    wrong_size = copy.deepcopy(analysis)
    wrong_size.size = (639, 420)
    invalid.append((photo, wrong_size, ["head-right"], "analysis dimensions"))
    invalid.append((photo, analysis, ["unknown"], "unknown ID"))
    invalid.append((photo, analysis, [], "empty IDs"))
    invalid.append((photo, None, ["head-right"], "missing analysis"))
    for mode in ("empty", "size", "missing"):
        bad = copy.deepcopy(analysis)
        bad.regions[1].mask = (
            Image.new("L", photo.size) if mode == "empty" else
            Image.new("L", (63, 42), 255) if mode == "size" else None
        )
        invalid.append((photo, bad, ["head-right"], f"{mode} mask"))
    duplicate = copy.deepcopy(analysis)
    duplicate.regions.append(copy.deepcopy(duplicate.regions[1]))
    invalid.append((photo, duplicate, ["head-right"], "ambiguous duplicate ID"))
    failures = []
    for image, state, ids, label in invalid:
        try:
            accepted, _reason = es.validate_selection(image, state, ids)
            if accepted:
                failures.append(label)
        except Exception as error:
            failures.append(f"{label} raised {type(error).__name__}")
    assert not failures, failures


def masks_union_and_protection():
    photo, analysis = fixture()
    union = es.union_mask(analysis, ["head-right", "small-object"])
    expected = (np.asarray(analysis.regions[1].mask) > 0) | (np.asarray(analysis.regions[3].mask) > 0)
    assert union.size == photo.size and np.array_equal(np.asarray(union) > 0, expected)
    protect = es.protection_mask(analysis, ["head-right"])
    assert protect is not None
    assert np.array_equal(np.asarray(protect) > 0, np.asarray(analysis.regions[0].mask) > 0)
    protect_object = es.protection_mask(analysis, ["object"])
    assert protect_object is not None
    assert not np.any(np.asarray(protect_object)[np.asarray(analysis.regions[1].mask) > 0]), (
        "head contained in explicitly selected object remained protected"
    )
    for helper, ids in ((es.union_mask, ["head-right"]), (es.protection_mask, ["object"])):
        bad = copy.deepcopy(analysis)
        # The unselected head must also retain its real geometry.
        bad.regions[1 if helper is es.union_mask else 0].mask = Image.new("L", (64, 42), 255)
        try:
            result = helper(bad, ids)
        except (ValueError, TypeError):
            continue
        assert result is None, f"{helper.__name__} silently resized malformed mask"


def point_selection_ambiguity():
    _photo, analysis = fixture()
    assert es.choose_at_point(analysis, 410, 100).id == "head-right", "contained head must be selectable"
    assert es.choose_at_point(analysis, 280, 240).id == "small-object"
    assert es.choose_at_point(analysis, -1, 10) is None
    assert es.choose_at_point(analysis, 639, 419) is None
    duplicate = copy.deepcopy(analysis.regions[3])
    duplicate.id = "equally-supported-object"
    analysis.regions.append(duplicate)
    assert es.choose_at_point(analysis, 280, 240) is None, "ambiguous objects silently resolved by order"


class RecordingInference:
    def __init__(self, fault=""):
        self.loads, self.generations = [], []
        self.fault = fault

    def load(self, *args, **kwargs):
        self.loads.append((args, kwargs))
        return self.fault != "load", "CPU inference double"

    def generate(self, **kwargs):
        self.generations.append(kwargs)
        if self.fault == "generate":
            raise RuntimeError("injected inference failure")
        photo = kwargs.get("image", kwargs.get("source_image"))
        return SimpleNamespace(image=Image.new("RGB", photo.size, (12, 240, 35))), "CPU generation"


def execute(ids=None, analysis_mutator=None, engine="kontext", fault="", finish=False,
            prompt="Make the selected element silver", quality_mode=False, missing_analysis=False,
            scene_factory=fixture):
    photo, analysis = scene_factory()
    if analysis_mutator:
        analysis_mutator(analysis)
    original_pixels = np.asarray(photo).copy()
    manager, client = ImgEditorManager(), RecordingInference(fault)
    guards, finishes = [], []

    def destructive_finish(image, **_kwargs):
        finishes.append(image.size)
        if fault == "finish":
            raise RuntimeError("injected finish failure")
        # Real quality mode may upscale; composition must return to the exact
        # source geometry before restoring the selection's exterior.
        larger = (image.width * 2, image.height * 2)
        return Image.new("RGB", larger, (250, 20, 30)), "CPU destructive finish", {}

    quality = SimpleNamespace(finish=destructive_finish, release_gpu_caches=lambda: None)
    with ExitStack() as stack:
        for target, result in (
            ("ui.tabs.comfy_launcher.is_comfyui_running", True),
            ("roop.img_editor.img_editor_manager.get_vram_gb", 8.0),
            ("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client", client),
            ("roop.img_editor.prompt_translator.translate_prompt", prompt),
            ("roop.img_editor.image_quality_pipeline.get_quality_finisher", quality),
            ("roop.img_editor.top_tier_finish.should_edit_top_finish", False),
        ):
            stack.enter_context(patch(target, return_value=result))
        # v5.1227 — the selected region is edited with the MEASURED regime of
        # the global route: the concept analyzer measures the direction and
        # the intensity of the instruction (no vocabulary of the user). That
        # measurement is allowed here. The routing, the rewriting and the
        # segmentation of the global route are NOT: an explicit selection
        # never falls back to textual inference.
        for name in ("_get_clipseg_masker", "_apply_prompt_rewriter",
                     "_get_clothing_segmenter", "_route_edit_engine", "_route_body_engine",
                     "_explicit_viewpoint_request", "_build_camera_pose_request"):
            if hasattr(manager, name):
                guard = stack.enter_context(patch.object(manager, name,
                    side_effect=AssertionError(f"explicit selection reached textual inference: {name}")))
                guards.append((name, guard))
        for target in ("roop.face_util.get_face_analyser", "roop.face_util.get_face_analyser_cpu",
                       "roop.face_util.detect_faces_robust"):
            guards.append((target, stack.enter_context(patch(target,
                side_effect=AssertionError("explicit selection reached face preflight")))))
        if finish:
            stack.enter_context(patch("roop.img_editor.top_tier_finish.run_top_tier_quality",
                                      side_effect=destructive_finish))
        else:
            guards.append(("run_top_tier_quality", stack.enter_context(patch(
                "roop.img_editor.top_tier_finish.run_top_tier_quality",
                side_effect=AssertionError("explicit selection reached global pure quality"),
            ))))
        stack.enter_context(patch.object(manager, "_release_local_gpu_before_comfy"))
        output, message, mask = manager.generate_intelligent(
            image=photo, prompt=prompt, engine=engine,
            selection_analysis=None if missing_analysis else analysis,
            selected_region_ids=["head-right"] if ids is None else ids,
            use_semantic=True, use_rewriter=True, enhance_faces=False, auto_upscale=finish,
            quality_mode=quality_mode,
        )
    for name, guard in guards:
        assert not guard.called, f"explicit selection invoked {name}"
    assert np.array_equal(np.asarray(photo), original_pixels), "input photo was mutated"
    return SimpleNamespace(source=photo, analysis=analysis, client=client, output=output,
                           message=message, mask=mask, finishes=finishes, prompt=prompt)


def _assert_clean_crop(run, reference):
    """The reference is an exact crop of the input, apart from the blind patch.

    Unselected heads inside the crop are greyed out so the editor cannot
    target them. Every other pixel must still be the untouched input.
    """
    source, ref = np.asarray(run.source), np.asarray(reference)
    height, width = ref.shape[0], ref.shape[1]
    candidates = [(y, x) for y, x in np.argwhere(np.all(source == ref[0, 0], axis=2))
                  if y + height <= source.shape[0] and x + width <= source.shape[1]]
    assert candidates, "reference was not an input crop"
    blind = es.load_settings()["edit"]["blind_fill"]
    blind_gray = int(blind[0])
    for y, x in candidates:
        window = source[y:y + height, x:x + width]
        if np.array_equal(window, ref):
            return
        changed = np.any(window != ref, axis=2)
        if not changed.any():
            continue
        # Only the grey blind patch may differ from the input crop.
        assert np.all(ref[changed] == blind_gray), "reference altered pixels outside the blind patch"
        return
    raise AssertionError("reference was not clean input crop")


def assert_omitted(run):
    assert run.output is not None, run.message
    assert np.array_equal(np.asarray(run.output), np.asarray(run.source)), run.message
    assert "omitida" in run.message.lower(), run.message


def assert_local(run, ids, finish=False):
    assert run.output is not None and run.output.size == run.source.size, run.message
    assert len(run.client.loads) == len(run.client.generations) == 1, run.message
    args, options = run.client.loads[0]
    version = str(options.get("flux_version", args[0] if args else ""))
    assert "kontext" in version.lower(), f"unexpected model: {version}"
    request = run.client.generations[0]
    assert run.prompt in request["prompt"], "complete instruction was lost"
    assert request.get("mask_image") is None, "Kontext received inpaint mask"
    settings = es.load_settings()["edit"]
    assert request.get("reference_canvas") == {
        "max_side": settings["maximum_reference_side"],
        "multiple": settings["reference_canvas_multiple"],
    }, "selected region lost its configured sampling resolution"
    reference = request.get("image", request.get("source_image"))
    if all(region.kind == "head" for region in run.analysis.regions if region.id in ids):
        assert reference.width * reference.height < run.source.width * run.source.height, "head reference lost localization"
    else:
        assert reference.width * reference.height <= run.source.width * run.source.height, "reference exceeds source context"
    _assert_clean_crop(run, reference)
    source, out = np.asarray(run.source), np.asarray(run.output)
    selected = np.asarray(es.union_mask(run.analysis, ids)) > 0
    assert np.any(out[selected] != source[selected]), "selected region did not change"
    assert not np.any(out[~selected] != source[~selected]), "pixels outside explicit selection changed"
    protected = es.protection_mask(run.analysis, ids)
    if protected is not None:
        assert not np.any(out[np.asarray(protected) > 0] != source[np.asarray(protected) > 0]), "other head changed"
    if finish:
        assert run.finishes, "destructive finish not exercised"


def explicit_head():
    assert_local(execute(), ["head-right"])


def explicit_multiple():
    ids = ["head-right", "small-object"]
    assert_local(execute(ids), ids)


def explicit_object_includes_head():
    run = execute(["object"])
    assert_local(run, ["object"])
    head = np.asarray(run.analysis.regions[1].mask) > 0
    assert np.any(np.asarray(run.output)[head] != np.asarray(run.source)[head]), "selected object's head frozen"


def overlapping_heads_protected():
    def move_other_head(analysis):
        mask = Image.new("L", analysis.size)
        ImageDraw.Draw(mask).rectangle((370, 75, 439, 169), fill=255)
        analysis.regions[0].mask = mask
        analysis.regions[0].bbox = (370, 75, 440, 170)

    photo, analysis = fixture()
    move_other_head(analysis)
    protected = es.protection_mask(analysis, ["head-right"])
    assert protected is not None, "unselected head lost protection due to overlapping selected head"
    other = np.asarray(analysis.regions[0].mask) > 0
    assert np.all(np.asarray(protected)[other] > 0), "other head only partially protected"
    run = execute(analysis_mutator=move_other_head)
    assert_local(run, ["head-right"])
    assert np.array_equal(np.asarray(run.output)[other], np.asarray(photo)[other]), "overlapping other head changed"


def explicit_after_finish():
    assert_local(execute(finish=True), ["head-right"], finish=True)


def early_route_precedence():
    for prompt in ("Improve quality and sharpness", "Show from an overhead camera angle",
                   "Make the person taller and change the posture"):
        assert_local(execute(prompt=prompt), ["head-right"])


def invalid_never_global():
    for ids, mutator in (
        (["unknown"], None), ([], None),
        (["head-right"], lambda state: setattr(state, "fingerprint", "stale")),
        (["head-right"], lambda state: setattr(state.regions[1], "mask", Image.new("L", state.size))),
    ):
        run = execute(ids, mutator)
        assert_omitted(run)
        assert not run.client.loads and not run.client.generations, "invalid selection reached inference"
    run = execute(missing_analysis=True)
    assert_omitted(run)
    assert not run.client.generations, "orphan selection reached inference"


def explicit_empty_prompt_never_global_quality():
    run = execute(prompt="", quality_mode=True)
    assert_omitted(run)
    assert not run.client.generations, "empty instruction caused generation"


def failures_never_global():
    for fault in ("load", "generate"):
        assert_omitted(execute(fault=fault))
    assert_omitted(execute(fault="finish", finish=True))
    run = execute(engine="flux_klein")
    assert_omitted(run)
    assert not run.client.generations, "incompatible engine silently used Kontext/global"


def narrow_references_have_usable_latents():
    def scene(size, box):
        photo = Image.new("RGB", size, (80, 100, 120))
        mask = Image.new("L", size)
        ImageDraw.Draw(mask).rectangle(box, fill=255)
        region = es.SelectionRegion("object", "narrow measured element", mask, box, "object")
        return photo, es.SelectionAnalysis(es.image_fingerprint(photo), size, [region])

    run = execute(["object"], scene_factory=lambda: scene((500, 2200), (4, 100, 5, 2099)))
    assert_local(run, ["object"])
    reference = run.client.generations[0]["image"].copy()
    settings = es.load_settings()["edit"]
    maximum = int(settings["maximum_reference_side"])
    minimum = int(settings["minimum_reference_side"])
    reference.thumbnail((maximum, maximum))
    assert min(reference.size) >= minimum, f"thin reference collapses under client thumbnail: {reference.size}"
    for size, box in (((40, 40), (10, 10, 15, 15)),
                      ((2000, 80), (0, 35, 1999, 36))):
        impossible = execute(["object"], scene_factory=lambda: scene(size, box))
        assert_omitted(impossible)
        assert not impossible.client.loads and not impossible.client.generations, (
            "impossible reference format reached inference"
        )


def object_context_keeps_identical_write_support():
    def mark_as_object(analysis):
        for region in analysis.regions:
            if region.id == "head-right":
                region.kind = "object"

    head_run = execute()
    object_run = execute(analysis_mutator=mark_as_object)
    assert_local(head_run, ["head-right"])
    assert_local(object_run, ["head-right"])
    head_reference = head_run.client.generations[0]["image"]
    object_reference = object_run.client.generations[0]["image"]
    assert object_reference.width * object_reference.height > head_reference.width * head_reference.height, (
        "object reference kept the head's insufficient context"
    )
    assert np.array_equal(np.asarray(head_run.mask), np.asarray(object_run.mask)), (
        "more context expanded the editable support"
    )
    assert np.array_equal(np.asarray(head_run.output), np.asarray(object_run.output)), (
        "same adversarial generation escaped the measured write support"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-only", action="store_true")
    args = parser.parse_args()
    cases = [fingerprint_exact, validation_rejects_bad_state, masks_union_and_protection,
             point_selection_ambiguity]
    if not args.selection_only:
        cases += [explicit_head, explicit_multiple, explicit_object_includes_head,
                  overlapping_heads_protected, explicit_after_finish,
                  early_route_precedence, invalid_never_global,
                  explicit_empty_prompt_never_global_quality, failures_never_global,
                  narrow_references_have_usable_latents, object_context_keeps_identical_write_support]
    failures = []
    for check in cases:
        try:
            check()
            print(f"PASS {check.__name__}", flush=True)
        except Exception as error:
            failures.append(check.__name__)
            print(f"FAIL {check.__name__}: {error}", flush=True)
            traceback.print_exc()
    print(f"H-EDIT-SELECT CPU: {len(cases) - len(failures)}/{len(cases)} PASS; failures={failures}", flush=True)
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
