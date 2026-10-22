"""CPU contract: measured Kontext head edits preserve every other person.

All images are synthetic, clothed, gender-neutral figures. Inference,
translation and model analysis are doubled. Fault cases additionally inject
an exception in PIL composite; successful composition and all production head
selection/crop helpers run unchanged. This proves dispatch/pixel protection,
never GPU adherence or visual edit quality.
"""
import os
import subprocess
import sys
import traceback
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.getcwd())

from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402


PROMPT = "The person wearing the amber scarf must have silver hair"
REGION = "person wearing the amber scarf"


class RecordingClient:
    """Adversarial diffusion double replaces every pixel of its reference."""

    def __init__(self):
        self.loads = []
        self.generations = []

    def load(self, *args, **kwargs):
        self.loads.append({**kwargs, **({"flux_version": args[0]} if args else {})})
        return True, "CPU inference double"

    def generate(self, **kwargs):
        self.generations.append(kwargs)
        source = kwargs.get("image", kwargs.get("source_image"))
        return SimpleNamespace(image=Image.new("RGB", source.size, (84, 163, 111))), "CPU generated"


class MeasuredMasker:
    """Deterministic stand-in for CLIPSeg inference, with recorded queries."""

    def __init__(self, scene, faces, ambiguous=False):
        self.scene = scene
        self.faces = faces
        self.ambiguous = ambiguous
        self.queries = []

    def generate_mask(self, image, query, **_kwargs):
        self.queries.append(query)
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        selected = self.faces if self.ambiguous or query in ("head", "hair") else self.faces[1:]
        for face in selected:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            fw, fh = x2 - x1, y2 - y1
            box = (x1 - fw // 3, max(0, y1 - fh), x2 + fw // 3, y2 + fh // 4)
            if query == "hair":
                box = (*box[:3], y1 + fh // 5)
            draw.rectangle(box, fill=255)
        return mask

    def unload(self):
        pass


def _fixture(dominant=False, near_weak_other=False):
    width, height = 640, 420
    yy, xx = np.indices((height, width))
    photo = np.stack((110 + xx % 31, 130 + yy % 23, 145 + (xx + yy) % 17), axis=-1).astype(np.uint8)
    scene = Image.fromarray(photo, "RGB")
    boxes = [(105, 145, 175, 225), (365, 140, 435, 220)]
    if dominant:
        boxes = [(15, 180, 45, 220), (185, 100, 615, 405)]
    elif near_weak_other:
        boxes = [(300, 145, 370, 225), (365, 140, 435, 220)]
    faces = [SimpleNamespace(bbox=np.array(box, dtype=np.float32), det_score=0.95,
                             embedding=np.eye(2, dtype=np.float32)[index])
             for index, box in enumerate(boxes)]
    if near_weak_other:
        faces[0].det_score = 0.31
    draw = ImageDraw.Draw(scene)
    for index, (x1, y1, x2, y2) in enumerate(boxes):
        fw, fh = x2 - x1, y2 - y1
        draw.rectangle((x1 - fw // 2, y2, min(width, x2 + fw // 2), height),
                       fill=(42, 75 + index * 15, 112))
        draw.ellipse((x1 - fw // 4, max(0, y1 - fh // 2), x2 + fw // 4, y2),
                     fill=(54 + index * 15, 41, 35))
        draw.ellipse((x1, y1, x2, y2), fill=(194, 151, 122))
        draw.rectangle((x1, y2, x2, min(height, y2 + fh // 3)),
                       fill=(30, 90, 160) if index == 0 else (192, 132, 38))
    return scene, faces


def _semantic_analysis():
    return SimpleNamespace(
        get_magnitude=lambda *_a: 0.58,
        detect_target=lambda *_a: "hair",
        # Appearance is measured separately; this scene has no clothing edit.
        get_axis_scores=lambda *_a: {"attribute": 0.0, "pose": 0.0, "structural": 0.0},
        score_concepts=lambda *_a: {"attribute_appearance": 0.4},
        get_body_transform_intensity=lambda *_a: 0.0,
        get_body_direction=lambda *_a: "change",
        is_structural_dominant=lambda *_a: False,
        get_structural_bias=lambda *_a: "neutral",
        is_quality_dominant=lambda *_a: False,
    )


def _execute(engine="kontext", ambiguous=False, dominant=False, finish=False,
             weak_target=False, composite_fault=None, near_weak_other=False,
             prompt=PROMPT, region=REGION, semantic_analyzer=None, rewritten_prompt=None):
    scene, faces = _fixture(dominant, near_weak_other)
    if weak_target:
        faces[1].det_score = 0.31
    manager = ImgEditorManager()
    masker = MeasuredMasker(scene, faces, ambiguous)
    client = RecordingClient()
    finish_calls = []
    composite_calls = []
    real_composite = Image.composite

    def failing_composite(image1, image2, mask):
        composite_calls.append((image1.size, image2.size, mask.size))
        if len(composite_calls) == composite_fault:
            raise RuntimeError("CPU injected composite failure")
        return real_composite(image1, image2, mask)

    def destructive_finish(image, **_kwargs):
        finish_calls.append(image.size)
        return Image.new("RGB", image.size, (71, 142, 103)), "CPU destructive finish", {}

    quality = SimpleNamespace(
        analyze_image=lambda *_a, **_kw: {"scores": {"poster": 0.0, "sharpness": 1.0, "restore": 0.0}},
        finish=destructive_finish,
    )
    with ExitStack() as stack:
        if composite_fault is not None:
            stack.enter_context(patch("PIL.Image.composite", side_effect=failing_composite))
        for name, value in (
            ("ui.tabs.comfy_launcher.is_comfyui_running", True),
            ("roop.face_util.get_face_analyser", SimpleNamespace(get=lambda *_a, **_kw: faces)),
            ("roop.face_util.get_face_analyser_cpu", SimpleNamespace(get=lambda *_a, **_kw: faces)),
            ("roop.face_util.detect_faces_robust", faces),
            ("roop.img_editor.img_editor_manager.get_vram_gb", 8.0),
            ("roop.img_editor.prompt_translator.translate_prompt", prompt),
            ("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client", client),
            ("roop.img_editor.image_quality_pipeline.get_quality_finisher", quality),
            ("roop.img_editor.instruction_verifier.get_instruction_verifier",
             SimpleNamespace(verify=lambda *_a, **_kw: {"compliant": True, "source": "CPU double"})),
            ("roop.img_editor.top_tier_finish.should_edit_top_finish", False),
            ("roop.img_editor.controlnet_utils.get_controlnet_utils", SimpleNamespace(
                detect_pose=lambda image: np.zeros((image.height, image.width, 3), dtype=np.uint8),
            )),
        ):
            stack.enter_context(patch(name, return_value=value))
        stack.enter_context(patch.object(manager, "_get_semantic_analyzer", return_value=(
            semantic_analyzer if semantic_analyzer is not None else _semantic_analysis()
        )))
        stack.enter_context(patch.object(manager, "_get_clipseg_masker", return_value=masker))
        stack.enter_context(patch.object(manager, "_get_clothing_segmenter", return_value=None))
        stack.enter_context(patch.object(manager, "_release_local_gpu_before_comfy", return_value=None))
        stack.enter_context(patch.object(manager, "_apply_prompt_rewriter", return_value=(
            prompt if rewritten_prompt is None else rewritten_prompt,
            0.58, "hair", False, True, False, [region], region,
        )))
        output, message, mask = manager.generate_intelligent(
            image=scene, prompt=prompt, engine=engine, use_semantic=True,
            use_rewriter=True, enhance_faces=False, auto_upscale=finish,
        )
    return SimpleNamespace(source=scene, faces=faces, manager=manager, masker=masker,
                           client=client, output=output, message=message, mask=mask,
                           finish_calls=finish_calls, composite_calls=composite_calls,
                           prompt=prompt, region=region)


def _assert_protected(run, finish=False):
    assert run.output is not None, run.message
    assert run.client.generations, f"no generation: {run.message}"
    assert run.client.loads and all("kontext" in str(call.get("flux_version", "")).lower()
                                    for call in run.client.loads), run.client.loads
    assert run.region in run.masker.queries, run.masker.queries
    for call in run.client.generations:
        reference = call.get("image", call.get("source_image"))
        assert reference.width * reference.height < run.source.width * run.source.height, (
            f"Kontext received full image {reference.size}, expected a measured local crop"
        )
        assert call.get("mask_image") is None, "Kontext must never receive an inpaint mask"
    source, output = np.asarray(run.source), np.asarray(run.output)
    assert output.shape == source.shape, (output.shape, source.shape)
    assert run.mask is not None, "localized output lost its measured composition mask"
    exterior = np.asarray(run.mask.convert("L")) == 0
    assert exterior.any(), "invalid fixture: no exterior to protect"
    assert np.array_equal(output[exterior], source[exterior]), "exterior of measured region changed"
    assert np.array_equal(output[:35], source[:35]), "far exterior changed"
    assert np.array_equal(output[-30:], source[-30:]), "lower exterior changed"
    other_head = run.manager._head_mask_from_faces(run.source, [run.faces[0]])
    protected = np.asarray(other_head) > 0
    assert np.array_equal(output[protected], source[protected]), "other head changed after feather/finish"
    target_box = tuple(int(v) for v in run.faces[1].bbox)
    x1, y1, x2, y2 = target_box
    assert np.any(output[y1:y2, x1:x2] != source[y1:y2, x1:x2]), "requested subject stayed unchanged"
    if finish:
        assert run.finish_calls, "destructive finish was not exercised"


def _localized():
    _assert_protected(_execute())


def _after_finish():
    _assert_protected(_execute(finish=True), finish=True)


def _ambiguous():
    run = _execute(ambiguous=True)
    assert not run.client.generations, "ambiguous head selection still generated"
    assert run.output is not None and np.array_equal(np.asarray(run.output), np.asarray(run.source)), run.message
    assert "omitida" in run.message.lower(), run.message


def _impossible_crop():
    run = _execute(dominant=True)
    head = run.manager._head_mask_from_faces(run.source, [run.faces[1]])
    assert run.manager._crop_focal_context(run.source, head, report=False) is None, (
        "invalid test fixture: head crop was viable"
    )
    assert not run.client.generations, "infeasible focal crop fell back to global generation"
    assert run.output is not None and np.array_equal(np.asarray(run.output), np.asarray(run.source)), run.message
    assert "omitida" in run.message.lower(), run.message


def _weak_target():
    run = _execute(weak_target=True)
    credible = run.manager._credible_faces(run.faces)
    assert len(credible) == 1 and credible[0] is run.faces[0], "invalid weak-target fixture"
    assert not run.client.generations, "discarded target caused editing of the remaining person"
    assert run.output is not None and np.array_equal(np.asarray(run.output), np.asarray(run.source)), run.message
    assert "omitida" in run.message.lower(), run.message


def _near_weak_other():
    run = _execute(near_weak_other=True, finish=True)
    credible = run.manager._credible_faces(run.faces)
    assert len(credible) == 1 and credible[0] is run.faces[1], "invalid weak-other fixture"
    other_head = np.asarray(run.manager._head_mask_from_faces(run.source, [run.faces[0]])) > 0
    target_head = np.asarray(run.manager._head_mask_from_faces(run.source, [run.faces[1]])) > 0
    assert (other_head & target_head).any(), "invalid nearby fixture: head supports do not overlap"
    _assert_protected(run, finish=True)


def _composite_failure():
    run = _execute(finish=True, composite_fault=1)
    assert run.client.generations, "fault case never exercised generation"
    assert len(run.composite_calls) == 1, run.composite_calls
    assert run.composite_calls[0][0] != run.source.size, "fault missed the local crop composite"
    assert not run.finish_calls, "unsafe crop proceeded into finish"
    assert run.output is not None and np.array_equal(np.asarray(run.output), np.asarray(run.source)), (
        "crop composition failure exposed the raw generated crop"
    )
    assert "omitida" in run.message.lower(), run.message


def _restoration_failure():
    run = _execute(finish=True, composite_fault=2)
    assert run.client.generations and run.finish_calls, "restoration case missed generation/finish"
    assert len(run.composite_calls) == 2, run.composite_calls
    assert run.composite_calls[0][0] != run.source.size, "first composite was not the crop"
    assert run.composite_calls[1][0] == run.source.size, "fault missed final exterior restoration"
    assert run.output is not None and np.array_equal(np.asarray(run.output), np.asarray(run.source)), (
        "exterior restoration failure exposed the globally modified finish"
    )
    assert "omitida" in run.message.lower(), run.message


def _rewrite_grounding(region, subject, expected_descriptor,
                       prompt="The person wearing the purple tie must have blue hair"):
    manager = ImgEditorManager()
    response = {
        "prompt": prompt,
        "mask_target": "hair", "is_global": False,
        "affected_regions": [region], "subject_query": subject,
    }
    # Exercise the real structured plan adapter, replacing only LLM inference.
    rewriter = SimpleNamespace(rewrite=lambda *_a, **_kw: response)
    with patch("roop.img_editor.prompt_rewriter.get_prompt_rewriter", return_value=rewriter):
        plan = manager._apply_prompt_rewriter(
            response["prompt"], magnitude=0.58, body_transform=False,
        )
    assert plan[4] is True, plan
    expected = manager._visual_subject_description(prompt).lower()
    assert expected and expected_descriptor in expected, (expected, expected_descriptor)
    assert plan[6] == [expected], plan


def _rewriter_preserves_descriptor():
    _rewrite_grounding("person wearing purple tie", "person", "purple tie")


def _rewriter_enriches_anatomical():
    _rewrite_grounding("hair", "person wearing purple tie", "purple tie")


def _rewriter_recovers_generic_both_fields():
    _rewrite_grounding("person", "person", "purple tie")
    _rewrite_grounding("man", "man", "on the left", prompt="The man on the left is bald")


def _described_subject_scope():
    from roop.img_editor.nlp.semantic_analyzer import LightLocalIntentAnalyzer

    prompt = "The person wearing the purple tie must have blue hair"
    region = "person wearing the purple tie"
    analyzer = LightLocalIntentAnalyzer()
    scope = ImgEditorManager._requested_change_scope(prompt)
    assert analyzer.is_structural_dominant(prompt), "fixture lost the original full-prompt structural signal"
    assert not analyzer.is_structural_dominant(scope), "requested predicate unexpectedly structural"
    with (
        patch.object(analyzer, "is_structural_dominant", wraps=analyzer.is_structural_dominant) as structural,
        patch.object(analyzer, "get_structural_bias", wraps=analyzer.get_structural_bias) as bias,
    ):
        run = _execute(prompt=prompt, region=region, semantic_analyzer=analyzer)
    assert structural.call_args_list and structural.call_args_list[0].args == (scope,), structural.call_args_list
    assert bias.call_args_list and bias.call_args_list[0].args == (scope,), bias.call_args_list
    _assert_protected(run)
    assert all(call.get("controlnet_image") is None and call.get("controlnet_pose_image") is None
               for call in run.client.generations), "descriptor caused structural ControlNet injection"


def _preservation_clause_survives_rewriter():
    from roop.img_editor.nlp.semantic_analyzer import LightLocalIntentAnalyzer

    prompt = ("The person wearing the purple tie must have blue hair. "
              "Leave the other person unchanged.")
    corrupted = ("The person wearing the purple tie must have blue hair. "
                 "Leave the other person changed.")
    run = _execute(
        prompt=prompt, region="person wearing the purple tie",
        semantic_analyzer=LightLocalIntentAnalyzer(), rewritten_prompt=corrupted,
    )
    _assert_protected(run)
    for call in run.client.generations:
        sent = call["prompt"]
        assert prompt.rstrip(".") in sent, sent
        assert "Leave the other person changed" not in sent, sent
        assert call.get("controlnet_pose_image") is None, "preservation clause caused structural ControlNet"


def _kontext_avoids_legacy_word_recipes():
    legacy_names = ("_is_bald_request", "_bald_recipe_tail",
                    "_unique_subject_hair_mask", "_bald_hair_coverage")
    with ExitStack() as stack:
        guards = [(name, stack.enter_context(patch.object(
            ImgEditorManager, name,
            side_effect=AssertionError(f"Kontext called legacy word-specific helper {name}"),
        ))) for name in legacy_names]
        run = _execute()
    # Check calls explicitly as well: production error handling must not hide
    # an attempted legacy recipe behind a caught inference/analysis exception.
    for name, guard in guards:
        assert guard.call_count == 0, f"Kontext invoked {name}: {guard.call_args_list}"
    _assert_protected(run)


def main():
    if "--baseline-head" in sys.argv:
        # Evaluate the committed source only inside this subprocess. Never
        # overwrite the working manager while the editor is implementing it.
        import roop.img_editor.img_editor_manager as manager_module
        source = subprocess.check_output([
            "git", "show", "HEAD:roop/img_editor/img_editor_manager.py",
        ]).decode("utf-8")
        exec(compile(source, "<committed HEAD img_editor_manager>", "exec"), manager_module.__dict__)
        globals()["ImgEditorManager"] = manager_module.ImgEditorManager
        print("BASELINE: committed HEAD manager evaluated in memory; working tree unchanged", flush=True)
    cases = [("localized", _localized), ("after_finish", _after_finish),
             ("ambiguous", _ambiguous), ("impossible_crop", _impossible_crop),
             ("weak_target", _weak_target), ("near_weak_other", _near_weak_other),
             ("composite_failure", _composite_failure),
             ("restoration_failure", _restoration_failure),
             ("rewriter_preserves_descriptor", _rewriter_preserves_descriptor),
             ("rewriter_enriches_anatomical", _rewriter_enriches_anatomical),
             ("rewriter_recovers_generic_both_fields", _rewriter_recovers_generic_both_fields),
             ("described_subject_scope", _described_subject_scope),
             ("preservation_clause_survives_rewriter", _preservation_clause_survives_rewriter),
             ("kontext_avoids_legacy_word_recipes", _kontext_avoids_legacy_word_recipes)]
    cases.extend((f"alias_{alias}", lambda alias=alias: _assert_protected(_execute(engine=alias)))
                 for alias in ("longcat", "imagine"))
    failures = []
    for name, check in cases:
        try:
            check()
            print(f"PASS {name}", flush=True)
        except Exception as error:
            failures.append(name)
            print(f"FAIL {name}: {error}", flush=True)
            traceback.print_exc()
    print(f"CPU protection contract: {len(cases) - len(failures)}/{len(cases)} PASS; failures={failures}", flush=True)
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
