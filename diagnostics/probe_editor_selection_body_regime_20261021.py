"""CPU contract: the selected region is edited with the MEASURED regime.

The explicit-selection route used to run a FIXED recipe (Kontext EDIT with
the 'offset' anchor) whatever the instruction said. With clean tokens any
anchor copies the photo (Q4+offset=copia, Q4+uxo=solo color), so a request
to remove the clothing was answered with a copy of the photo.

This probe fixes the measured behaviour of the regional route:

  * the engine recipe, the negative and the end-state sentence come from the
    measurement of the instruction, not from a fixed recipe;
  * a measured body change ("remove the clothing") runs on the engine WITH
    VISION (Qwen-VL sees the photo and the text is the only authority): in
    FLUX.1/Kontext the negative is inert (CFG=1) and the positive text alone
    leaves the garment on (measured skin share inside the garment zone:
    original 0.2675, Kontext 0.0676, vision engine 0.4346). By the user's
    decision (2026-10-21) that change REINTERPRETS THE FRAME: total
    compliance, the identity is not preserved;
  * every other instruction keeps the strict measured support: the exterior
    of the selection stays pixel-exact;
  * the head of the selected subject is NOT greyed out in the reference the
    editor receives (the measured inclusion rule is shared with the
    protection mask).

No model is loaded, downloaded, or installed by this diagnostic.
"""
from __future__ import annotations

import argparse
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


def scene(others=True):
    """A person whose measured head sticks out of the measured body.

    With `others` the fixture also carries another person, so the route has
    to decide between reinterpreting the frame and preserving it.
    """
    yy, xx = np.indices((420, 640))
    # Each pixel encodes its own position: the crop the editor receives can
    # be located exactly in the source, so the reference can be checked.
    data = np.stack((xx % 256, yy % 256, (xx + yy) % 256), axis=-1)
    photo = Image.fromarray(data.astype(np.uint8), "RGB")
    body = Image.new("L", photo.size)
    ImageDraw.Draw(body).rectangle((300, 70, 455, 400), fill=255)
    head = Image.new("L", photo.size)
    ImageDraw.Draw(head).ellipse((340, 40, 410, 130), fill=255)
    regions = [
        es.SelectionRegion("person", "mujer", body, (300, 70, 455, 400), "object"),
        es.SelectionRegion("head-person", "Cabeza/rostro 1", head, (340, 40, 410, 130), "head"),
    ]
    if others:
        other = Image.new("L", photo.size)
        ImageDraw.Draw(other).ellipse((60, 60, 130, 140), fill=255)
        regions.append(
            es.SelectionRegion("head-other", "Cabeza/rostro 2", other, (60, 60, 130, 140), "head"))
    for region in regions:
        region.coverage = float((np.asarray(region.mask) > 0).mean())
    return photo, es.SelectionAnalysis(es.image_fingerprint(photo), photo.size, regions)


class MeasuredSegmenter:
    """The garment surface of the fixture, measured and not guessed."""

    def is_available(self):
        return True

    def segment_clothing(self, image, threshold=0.45):
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).rectangle((300, 70, 455, 400), fill=255)
        return mask, {}


TRANSLATION = {
    "la mujer esta desnuda, sin zapatos": "The woman is naked, without shoes...",
    "Make the selected element silver": "Make the selected element silver",
    "debe ser calvo": "He must be bald...",
    "el hombre de la derecha debe ser cxalvo": "The man on the right must be cxalvo...",
    "CAMBIA EL ANGULO, IMAGEN DE LATERAL": "changes angle, side image",
}


class RecordingInference:
    def __init__(self, no_op_first=False):
        self.loads, self.generations = [], []
        self.no_op_first = no_op_first

    def load(self, *args, **kwargs):
        self.loads.append((args, kwargs))
        return True, "CPU inference double"

    def generate(self, **kwargs):
        self.generations.append(kwargs)
        photo = kwargs.get("image", kwargs.get("source_image"))
        if self.no_op_first and len(self.generations) == 1:
            # A no-op answer: the anchored recipe returned the photo itself.
            return SimpleNamespace(image=photo.copy()), "CPU no-op"
        return SimpleNamespace(image=Image.new("RGB", photo.size, (12, 240, 35))), "CPU generation"


def execute(prompt, ids=("person",), mutator=None, others=True, client=None):
    photo, analysis = scene(others=others)
    if mutator:
        mutator(analysis)
    manager = ImgEditorManager()
    client = client or RecordingInference()
    with ExitStack() as stack:
        for target, result in (
            ("ui.tabs.comfy_launcher.is_comfyui_running", True),
            ("roop.img_editor.img_editor_manager.get_vram_gb", 8.0),
            ("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client", client),
            ("roop.img_editor.prompt_translator.translate_prompt", TRANSLATION.get(prompt, prompt)),
            ("roop.img_editor.top_tier_finish.should_edit_top_finish", False),
        ):
            stack.enter_context(patch(target, return_value=result))
        # The garment surface is a MEASURED surface (the segmenter), not
        # textual inference: the route is allowed to ask for it.
        stack.enter_context(patch.object(manager, "_get_clothing_segmenter",
                                         return_value=MeasuredSegmenter()))
        for name in ("_get_clipseg_masker", "_apply_prompt_rewriter",
                     "_route_edit_engine", "_route_body_engine",
                     "_explicit_viewpoint_request", "_build_camera_pose_request"):
            if hasattr(manager, name):
                stack.enter_context(patch.object(manager, name,
                    side_effect=AssertionError(f"explicit selection reached textual inference: {name}")))
        stack.enter_context(patch.object(manager, "_release_local_gpu_before_comfy"))
        output, message, mask = manager.generate_intelligent(
            image=photo, prompt=prompt, engine="kontext",
            selection_analysis=analysis, selected_region_ids=list(ids),
            use_semantic=True, use_rewriter=True, enhance_faces=False, auto_upscale=False,
        )
    return SimpleNamespace(source=photo, analysis=analysis, client=client, output=output,
                           message=message, mask=mask)


def measured_body_change_uses_the_vision_engine():
    """A measured body change runs on the engine that follows the text."""
    run = execute("la mujer esta desnuda, sin zapatos")
    assert run.output is not None, run.message
    assert len(run.client.generations) == 1, run.message
    # The engine is the measured one: the vision model, loaded explicitly.
    loaded = [kwargs.get("flux_version") or (args[0] if args else "")
              for args, kwargs in run.client.loads]
    assert any("LongCat" in str(version) for version in loaded), (
        f"the measured body change did not load the vision engine: {loaded}")
    assert any(kwargs.get("allow_vision_engine")
               for _args, kwargs in run.client.loads), (
        "the vision engine was not requested explicitly")
    request = run.client.generations[0]
    prompt = str(request["prompt"])
    # The text is the only authority: no latent anchor.
    assert request.get("prompt_authority"), (
        "the measured body change did not give the text the authority")
    assert request.get("no_kontext"), (
        "the measured body change kept the latent anchor")
    assert "reference_canvas" not in request, (
        "the vision engine received the Kontext canvas")
    # The negative denies the garments that cover. With CFG=1 it has no
    # effect of its own, so it travels inside the text.
    assert "fully clothed" in prompt.lower(), (
        f"the measured negative did not travel in the text: {prompt}")
    # The end state and the enforcement are written from the measurement.
    assert "is fully nude" in prompt, f"the measured end state is missing: {prompt}"
    assert "must be clearly visible" in prompt, f"the enforcement is missing: {prompt}"
    assert "naked" in prompt, "the instruction itself was lost"


def body_change_with_other_people_preserves_the_frame():
    """With other people measured, the frame is not reinterpreted."""
    run = execute("la mujer esta desnuda, sin zapatos")
    request = run.client.generations[0]
    # The reinterpretation redraws every face (measured: 85.9 and 66.3 levels
    # on the two people of the two-person photo), so with other people the
    # measured garment zone is the hole and the photo around it survives.
    assert request.get("mask_image") is not None, (
        "the measured body change reinterpreted the frame with other people")
    assert request.get("keep_source_pixels"), (
        "the measured body change did not keep the photo around the zone")
    zone = np.asarray(request["mask_image"].convert("L")) > 0
    assert zone.any(), "the measured zone of the body change is empty"
    source, output = np.asarray(run.source), np.asarray(run.output)
    selected = np.asarray(es.union_mask(run.analysis, ["person"])) > 0
    assert np.any(output[selected] != source[selected]), (
        "the selected region did not change")
    assert not np.any(output[~selected] != source[~selected]), (
        "pixels outside the explicit selection changed")
    # The other person's measured face stays untouched.
    other = np.asarray(run.analysis.regions[2].mask) > 0
    assert not np.any(output[other] != source[other]), (
        "the face of an unselected person was not protected")


def body_change_without_other_people_reinterprets_the_frame():
    """Without other people, the frame is reinterpreted (the user's choice)."""
    run = execute("la mujer esta desnuda, sin zapatos", others=False)
    assert run.output is not None, run.message
    request = run.client.generations[0]
    assert not request.get("mask_image"), (
        "the measured body change kept the hole: the model completes the outfit")
    assert not request.get("keep_source_pixels"), (
        "the measured body change kept the photo pixels: the model copies it")
    source, output = np.asarray(run.source), np.asarray(run.output)
    selected = np.asarray(es.union_mask(run.analysis, ["person"])) > 0
    assert np.any(output[~selected] != source[~selected]), (
        "the measured body change did not reinterpret the frame")


def appearance_change_on_a_head_uses_the_focal_recipe():
    """A measured appearance change on a selected head: the focal recipe.

    Measured on the user's real cases:
      * the anchored Kontext recipe with the regional CANVAS does not edit
        anything (the anchor copies the photo);
      * the engine with vision and the head as a hole replaces the head with
        an invented one (the user, 01:40: «una edición pésima»);
      * reinterpreting the frame obeys but ignores the marked zone (the
        user: «lo de la selección de formas detectadas no funciona»).
    So the head case uses the focal recipe of the global route: the crop
    with its context (no canvas), the anchored edit and the measured
    appearance text. The measured selection is what gets written.
    """
    run = execute("debe ser calvo", ids=("head-person",))
    assert run.output is not None, run.message
    loaded = [kwargs.get("flux_version") or (args[0] if args else "")
              for args, kwargs in run.client.loads]
    assert all("LongCat" not in str(version) for version in loaded), (
        f"the appearance change used the engine that replaces the head: {loaded}")
    request = run.client.generations[0]
    # The crop with its context, without the regional canvas. It is scaled up
    # so the model has something to work with (a head of 136 px in a photo of
    # 2841 makes the model invent the fragment), but it shows the SELECTED
    # region, not the whole photo.
    assert "reference_canvas" not in request, (
        "the head case kept the regional canvas: the anchored recipe copies")
    reference = request.get("image", request.get("source_image"))
    assert reference is not None, "the head case sent no reference"
    same = np.array_equal(
        np.asarray(reference.resize(run.source.size, Image.Resampling.NEAREST)),
        np.asarray(run.source))
    assert not same, "the head case reference is the whole photo"
    # The measured bald recipe (the one that avoids the documented "cap").
    assert "bald head means a completely shaved scalp" in str(request["prompt"]).lower(), (
        f"the measured bald recipe is missing: {request['prompt']}")
    assert "must be bald" in str(request["prompt"]).lower(), (
        f"the instruction was lost: {request['prompt']}")
    # And the marked zone is what gets written.
    source, out = np.asarray(run.source), np.asarray(run.output)
    selected = np.asarray(es.union_mask(run.analysis, ["head-person"])) > 0
    assert np.any(out[selected] != source[selected]), (
        "the selected head did not change")
    assert not np.any(out[~selected] != source[~selected]), (
        "pixels outside the explicit selection changed")


def pose_change_uses_the_vision_engine():
    """A measured pose/angle change reinterprets the frame with vision.

    Real user case (2026-10-22): «CAMBIA EL ANGULO, IMAGEN DE LATERAL»
    (changes angle, side image) fell into the ordinary path (Kontext with
    the regional canvas + anchor), which is measured as the copy machine:
    the photo came back unchanged. A pose change is not a local patch: the
    body is repositioned as a whole, so only the engine that follows the
    text can do it, with the FULL photo as input, no anchor and no hole.
    The faces of other people are preserved from the original photo.
    """
    run = execute("CAMBIA EL ANGULO, IMAGEN DE LATERAL", ids=("person",))
    assert run.output is not None, run.message
    assert len(run.client.generations) == 1, run.message
    # The engine is the measured one: the vision model, loaded explicitly.
    loaded = [kwargs.get("flux_version") or (args[0] if args else "")
              for args, kwargs in run.client.loads]
    assert any("LongCat" in str(version) for version in loaded), (
        f"the pose change did not load the vision engine: {loaded}")
    assert any(kwargs.get("allow_vision_engine")
               for _args, kwargs in run.client.loads), (
        "the vision engine was not requested explicitly")
    request = run.client.generations[0]
    # The full photo is the input: a pose change cannot be a local crop.
    assert "reference_canvas" not in request, (
        "the pose change kept the regional canvas: the anchored recipe copies")
    reference = request.get("image", request.get("source_image"))
    assert reference is not None, "the pose change sent no reference"
    assert reference.size == run.source.size, (
        f"the pose change did not send the full photo: {reference.size}")
    # The text is the only authority: no anchor, no hole.
    assert request.get("prompt_authority"), (
        "the pose change did not give the text the authority")
    assert request.get("no_kontext"), (
        "the pose change kept the latent anchor")
    assert request.get("mask_image") is None, (
        "the pose change used a hole: the body moves as a whole")
    assert not request.get("keep_source_pixels"), (
        "the pose change kept the source pixels: the frame must change")
    assert not request.get("preserve_frame"), (
        "the pose change preserved the frame: the pose cannot be a patch")
    # The instruction survives in the measured prompt.
    assert "changes angle, side image" in str(request["prompt"]), (
        f"the instruction was lost: {request['prompt']}")
    # The frame is reinterpreted: the whole photo changes...
    source, output = np.asarray(run.source), np.asarray(run.output)
    assert np.any(output != source), "the pose change did not change the frame"
    # ...but the faces of other people are preserved from the original.
    protected = np.asarray(es.protection_mask(run.analysis, ["person"]).convert("L")) > 0
    assert protected.any(), "the fixture has no other face to protect"
    assert np.array_equal(output[protected], source[protected]), (
        "the other person's face was redrawn by the pose change")
    # The message tells the user what happened.
    assert "postura" in run.message.lower(), (
        f"the pose message is missing: {run.message}")


def unmeasured_instruction_is_reported_not_hidden():
    """An instruction the analyser does not measure says so.

    The user's real case (00:09, «el hombre de la derecha debe ser
    cxalvo»): the analyser covers no concept, so nothing was measured and
    the photo came back untouched. The route does not hijack the engine for
    that (the measured local cases — quality, angle, posture — must keep
    theirs); it REPORTS it, so a silent no-op becomes a visible one.
    """
    run = execute("el hombre de la derecha debe ser cxalvo", ids=("head-person",))
    assert run.output is not None, run.message
    assert "sin señal medida" in run.message, (
        f"the unmeasured instruction was not reported: {run.message}")
    # The validated local recipe is not hijacked by an unmeasured wording.
    loaded = [kwargs.get("flux_version") or (args[0] if args else "")
              for args, kwargs in run.client.loads]
    assert all("LongCat" not in str(version) for version in loaded), (
        f"an unmeasured wording hijacked the engine: {loaded}")


def ordinary_edit_keeps_its_own_regime():
    """An instruction without a measured body change keeps the plain recipe."""
    run = execute("Make the selected element silver")
    assert run.output is not None, run.message
    loaded = [kwargs.get("flux_version") or (args[0] if args else "")
              for args, kwargs in run.client.loads]
    assert not any("LongCat" in str(version) for version in loaded), (
        f"an ordinary edit was pushed into the vision engine: {loaded}")
    request = run.client.generations[0]
    assert not request.get("prompt_authority"), (
        "an ordinary edit received the body authority")
    assert "is fully nude" not in str(request["prompt"]), (
        "an ordinary edit received a body end state")


def _locate(source, reference, blind):
    """Return the (x, y) of the reference crop inside the source."""
    ref = np.asarray(reference)
    height, width = ref.shape[0], ref.shape[1]
    grey = (ref[:, :, 0] == blind) & (ref[:, :, 1] == blind) & (ref[:, :, 2] == blind)
    for y in range(source.shape[0] - height + 1):
        for x in range(source.shape[1] - width + 1):
            window = source[y:y + height, x:x + width]
            changed = np.any(window != ref, axis=2)
            if not changed.any():
                return int(x), int(y)
            # The only difference allowed is the grey blind patch.
            if np.all(grey[changed]):
                return int(x), int(y)
    return None


def selected_head_is_not_blinded():
    """The head of the selected subject stays visible in the reference."""
    run = execute("Make the selected element silver")
    request = run.client.generations[0]
    reference = request.get("image", request.get("source_image"))
    assert reference is not None, "the reference was not sent"
    source = np.asarray(run.source)
    blind = int(es.load_settings()["edit"]["blind_fill"][0])
    place = _locate(source, reference, blind)
    assert place is not None, "the reference was not a crop of the source"
    x0, y0 = place
    ref = np.asarray(reference)
    grey = (ref[:, :, 0] == blind) & (ref[:, :, 1] == blind) & (ref[:, :, 2] == blind)
    # The selected subject's head must arrive visible.
    head = np.asarray(run.analysis.regions[1].mask) > 0
    head_crop = head[y0:y0 + ref.shape[0], x0:x0 + ref.shape[1]]
    assert head_crop.any(), "the fixture does not put the head inside the crop"
    assert not np.any(grey & head_crop), (
        "the head of the selected subject was blinded in the reference")
    # The other head must still be blinded: the crop stays unambiguous.
    other = np.asarray(run.analysis.regions[2].mask) > 0
    other_crop = other[y0:y0 + ref.shape[0], x0:x0 + ref.shape[1]]
    if other_crop.any():
        assert np.any(grey & other_crop), (
            "the unselected head was not blinded in the reference")


def ordinary_edit_exterior_stays_pixel_exact():
    """Without a measured body change, the support is the only written area."""
    run = execute("Make the selected element silver")
    source, output = np.asarray(run.source), np.asarray(run.output)
    selected = np.asarray(es.union_mask(run.analysis, ["person"])) > 0
    assert np.any(output[selected] != source[selected]), "the selected region did not change"
    assert not np.any(output[~selected] != source[~selected]), (
        "pixels outside the explicit selection changed")


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    cases = [measured_body_change_uses_the_vision_engine,
             body_change_with_other_people_preserves_the_frame,
             body_change_without_other_people_reinterprets_the_frame,
             appearance_change_on_a_head_uses_the_focal_recipe,
             pose_change_uses_the_vision_engine,
             unmeasured_instruction_is_reported_not_hidden,
             ordinary_edit_keeps_its_own_regime,
             selected_head_is_not_blinded,
             ordinary_edit_exterior_stays_pixel_exact]
    failures = []
    for check in cases:
        try:
            check()
            print(f"PASS {check.__name__}", flush=True)
        except Exception as error:
            failures.append(check.__name__)
            print(f"FAIL {check.__name__}: {error}", flush=True)
            traceback.print_exc()
    print(f"H-EDIT-SELECT regime CPU: {len(cases) - len(failures)}/{len(cases)} PASS; failures={failures}",
          flush=True)
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
