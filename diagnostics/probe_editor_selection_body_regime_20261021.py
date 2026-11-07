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
    original 0.2675, Kontext 0.0676, vision engine 0.4346). The prompt is
    written on the measured clothing zone. The rest of the photo stays.
    Conserving the face is not the goal (user, 2026-10-23);
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


class VestSegmenter:
    """Only the top garment measured: the rest of the outfit stays outside.

    The real photo measured 21568 px of clothing on a 66243 px person, with
    the trousers outside the fragment.
    """

    def is_available(self):
        return True

    def segment_clothing(self, image, threshold=0.45):
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).rectangle((300, 70, 455, 200), fill=255)
        return mask, {}


TRANSLATION = {
    "la mujer esta desnuda, sin zapatos": "The woman is naked, without shoes...",
    "debe ir descalza, sin pantalon, sin blusa":
        "must go barefoot, without pants, without blouse",
    # La frase que el usuario repitió en vivo (logs 12:06 y 12:13):
    # coordinada con «y», sin comas, así que el criterio de cláusula
    # no medida partido solo por comas no veía «without pants».
    "debe ir descalza y sin pantalones":
        "must go barefoot and without pants",
    "Make the selected element silver": "Make the selected element silver",
    "Change the person's blouse to red": "Change the person's blouse to red",
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


def execute(prompt, ids=("person",), mutator=None, others=True, client=None,
            segmenter=None, engine="kontext"):
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
                                         return_value=segmenter or MeasuredSegmenter()))
        for name in ("_get_clipseg_masker", "_apply_prompt_rewriter",
                     "_route_edit_engine", "_route_body_engine",
                     "_explicit_viewpoint_request", "_build_camera_pose_request"):
            if hasattr(manager, name):
                stack.enter_context(patch.object(manager, name,
                    side_effect=AssertionError(f"explicit selection reached textual inference: {name}")))
        stack.enter_context(patch.object(manager, "_release_local_gpu_before_comfy"))
        output, message, mask = manager.generate_intelligent(
            image=photo, prompt=prompt, engine=engine,
            selection_analysis=analysis, selected_region_ids=list(ids),
            use_semantic=True, use_rewriter=True, enhance_faces=False, auto_upscale=False,
        )
    return SimpleNamespace(source=photo, analysis=analysis, client=client, output=output,
                           message=message, mask=mask)


def measured_body_change_writes_the_measured_zone():
    """A measured body change runs img2img puro on the measured zone.

    v5.1251 — la ruta conserva la foto (decisión del usuario 2026-10-26:
    calidad y conservación), así que el cambio se escribe con Kontext SIN
    ReferenceLatent: el ancla con tokens limpios devuelve la foto vestida
    (medido, registro 0,999; log real 12:00, ropa intacta).
    """
    run = execute("la mujer esta desnuda, sin zapatos")
    assert run.output is not None, run.message
    assert len(run.client.generations) == 1, run.message
    loaded = [kwargs.get("flux_version") or (args[0] if args else "")
              for args, kwargs in run.client.loads]
    assert any("kontext" in str(version).lower() for version in loaded), (
        f"the measured body change did not load Kontext: {loaded}")
    request = run.client.generations[0]
    prompt = str(request["prompt"])
    # Sin ancla: el cliente construye el img2img puro solo con este kwarg.
    img2img = request.get("kontext_img2img_denoise")
    assert isinstance(img2img, (int, float)) and 0.80 <= float(img2img) <= 0.90, (
        f"the measured body change kept the latent anchor (img2img={img2img})")
    assert request.get("mask_image") is not None, (
        "the measured body change left the measured zone empty")
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


def body_change_without_other_people_writes_the_measured_zone():
    """Without other people, the prompt is still written on the garment zone.

    Reinterpreting the frame obeyed the text on a different photo
    (edit_1792703256.png, FastEdit_00836_). The user (2026-10-23): fulfilling
    the prompt and placing it on the measured zone matters more than keeping
    the face. A non-pose instruction does not replace the rest of the image.
    """
    run = execute("la mujer esta desnuda, sin zapatos", others=False)
    assert run.output is not None, run.message
    request = run.client.generations[0]
    assert request.get("mask_image") is not None, (
        "the measured body change replaced the whole photo")
    assert request.get("keep_source_pixels"), (
        "the measured body change did not keep the photo around the zone")
    img2img = request.get("kontext_img2img_denoise")
    assert isinstance(img2img, (int, float)) and 0.80 <= float(img2img) <= 0.90, (
        f"the measured body change kept the latent anchor (img2img={img2img})")
    zone = np.asarray(request["mask_image"].convert("L")) > 0
    assert zone.any(), "the measured zone of the body change is empty"
    source, output = np.asarray(run.source), np.asarray(run.output)
    selected = np.asarray(es.union_mask(run.analysis, ["person"])) > 0
    assert np.any(output[selected] != source[selected]), (
        "the selected region did not change")
    assert not np.any(output[~selected] != source[~selected]), (
        "pixels outside the explicit selection changed")


def garment_fragment_widens_to_the_person():
    """«sin pantalón, sin blusa» sobre una persona seleccionada.

    Caso real 2026-10-23 (edit_1792748877.png): object-1 era la mujer entera
    (66 243 px) pero la ropa medida solo 21 568 px (el top), así que el texto
    se escribió en un fragmento: el top se volvió camiseta y el pantalón
    quedó. «without pants» puntúa 0 en el analizador; entonces decide la foto
    y la zona pasa a ser la persona medida menos la cabeza. La comparación
    se hace contra la misma persona con la ropa COMPLETA medida: sin el
    ensanche, la zona del fragmento es una fracción de la otra.

    v5.1248 — se prueban las dos formas en que el usuario lo escribió: con
    comas y coordinada con «y» (log 12:13: «debe ir descalza y sin
    pantalones»), que no partía por cláusulas y dejaba la zona en 21 568 px.
    """
    failures = []
    for prompt in ("debe ir descalza, sin pantalon, sin blusa",
                   "debe ir descalza y sin pantalones"):
        try:
            _garment_fragment_case(prompt)
        except AssertionError as error:
            failures.append(f"{prompt}: {error}")
    assert not failures, failures


def _garment_fragment_case(prompt):
    run = execute(prompt, segmenter=VestSegmenter())
    assert run.output is not None, run.message
    assert len(run.client.generations) == 1, run.message
    request = run.client.generations[0]
    assert request.get("mask_image") is not None, "no measured zone"
    zone = np.asarray(request["mask_image"].convert("L")) > 0
    whole = execute(prompt)
    reference = np.asarray(whole.client.generations[0]["mask_image"].convert("L")) > 0
    assert zone.shape == reference.shape, (
        f"the zone crop changed: {zone.shape} vs {reference.shape}")
    ratio = float(zone.sum()) / max(1, int(reference.sum()))
    assert ratio > 0.85, (
        f"the zone stopped at the measured fragment ({ratio:.2f} of the person)")
    source, output = np.asarray(run.source), np.asarray(run.output)
    selected = np.asarray(es.union_mask(run.analysis, ["person"])) > 0
    assert np.any(output[selected] != source[selected]), (
        "the selected person did not change")
    assert not np.any(output[~selected] != source[~selected]), (
        "pixels outside the explicit selection changed")


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


def pose_change_reinterprets_the_frame():
    """A measured pose/angle change reinterprets the frame with Kontext.

    Real user case (2026-10-22): «CAMBIA EL ANGULO, IMAGEN DE LATERAL»
    (changes angle, side image) fell into the ordinary path (Kontext with
    the regional canvas + anchor), which is measured as the copy machine:
    the photo came back unchanged. A pose change is not a local patch: the
    body is repositioned as a whole, so the sampler must start from the FULL
    photo with high noise and NO ReferenceLatent (the anchor keeps the old
    pose). The faces of other people are preserved from the original photo.
    """
    run = execute("CAMBIA EL ANGULO, IMAGEN DE LATERAL", ids=("person",))
    assert run.output is not None, run.message
    assert len(run.client.generations) == 1, run.message
    loaded = [kwargs.get("flux_version") or (args[0] if args else "")
              for args, kwargs in run.client.loads]
    assert any("kontext" in str(version).lower() for version in loaded), (
        f"the pose change did not load Kontext: {loaded}")
    request = run.client.generations[0]
    # The full photo is the input: a pose change cannot be a local crop.
    assert "reference_canvas" not in request, (
        "the pose change kept the regional canvas: the anchored recipe copies")
    reference = request.get("image", request.get("source_image"))
    assert reference is not None, "the pose change sent no reference"
    assert reference.size == run.source.size, (
        f"the pose change did not send the full photo: {reference.size}")
    # Sin ancla y sin agujero: la postura es del marco entero.
    img2img = request.get("kontext_img2img_denoise")
    assert isinstance(img2img, (int, float)) and 0.80 <= float(img2img) <= 0.90, (
        f"the pose change kept the latent anchor (img2img={img2img})")
    assert request.get("mask_image") is None, (
        "the pose change used a hole: the body moves as a whole")
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


def clothing_change_does_not_reinterpret_the_frame():
    """A blouse edit must stay local instead of activating the pose route."""
    prompt = "Change the person's blouse to red"
    manager = ImgEditorManager()
    facts = manager._prompt_authority_facts(prompt, prompt)
    assert not facts["pose"], f"clothing request was classified as pose: {facts}"
    assert not facts["structural_remove"], (
        f"clothing request was classified as structural removal: {facts}")
    assert facts["region_ambiguous"], (
        f"clothing request did not remain eligible for measured localization: {facts}")
    assert manager._prompt_authority_facts(
        "Make the person sit", "Make the person sit")["pose"], (
        "a concrete pose request stopped using the measured pose route")
    for removal in ("Remove the lamp", "Remove a person from the photo"):
        assert manager._prompt_authority_facts(
            removal, removal)["structural_remove"], (
            f"a concrete removal stopped using the structural route: {removal}")

    run = execute(prompt)
    assert run.output is not None, run.message
    request = run.client.generations[0]
    reference = request.get("image", request.get("source_image"))
    assert reference is not None, "the local edit sent no reference"
    assert "reference_canvas" in request, (
        "a clothing edit skipped the ordinary anchored regional recipe")
    assert request.get("kontext_img2img_denoise") is None, (
        "a clothing edit used the high-denoise pose route")
    assert "sin señal medida" not in run.message, (
        f"the explicit selection was not accepted as local support: {run.message}")
    source, output = np.asarray(run.source), np.asarray(run.output)
    selected = np.asarray(es.union_mask(run.analysis, ["person"])) > 0
    assert np.any(output[selected] != source[selected]), (
        "the selected clothing region did not change")
    assert not np.any(output[~selected] != source[~selected]), (
        "pixels outside the selected person changed")


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


def explicit_longcat_selection_uses_the_measured_native_recipe():
    """An explicit LongCat selection reaches Turbo without losing its mask."""
    params = ImgEditorManager().auto_detect_params({"magnitude": 0.65}, "longcat")
    assert params["num_inference_steps"] == 8
    assert params["guidance_scale"] == 1.0
    assert params["denoise"] == 1.0
    run = execute(
        "la mujer esta desnuda, sin zapatos",
        engine="longcat",
    )
    assert run.output is not None, run.message
    loaded = [kwargs.get("flux_version") or (args[0] if args else "")
              for args, kwargs in run.client.loads]
    assert any("LongCat-Image-Edit-Turbo-Q4_K_S.gguf" in str(version)
               for version in loaded), f"LongCat Turbo was not loaded: {loaded}"
    request = run.client.generations[0]
    assert request.get("mask_image") is not None, (
        "the selected body edit lost its measured write zone")
    assert request.get("keep_source_pixels"), (
        "the selected body edit stopped preserving the photo around its zone")
    assert request.get("prompt_authority"), (
        "the vision edit did not receive the measured instruction")
    assert request.get("no_kontext"), (
        "the selected edit was routed through the Kontext workflow")
    assert request.get("num_inference_steps") == 8, (
        f"LongCat did not receive its configured native step count: {request}")
    assert request.get("guidance_scale") == 1.0, (
        f"LongCat did not receive its configured CFG: {request}")
    source, output = np.asarray(run.source), np.asarray(run.output)
    selected = np.asarray(es.union_mask(run.analysis, ["person"])) > 0
    assert np.any(output[selected] != source[selected]), (
        "the selected LongCat region did not change")
    assert not np.any(output[~selected] != source[~selected]), (
        "LongCat changed pixels outside the explicit selection")


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


def multi_clause_uncover_writes_full_body_suffix():
    """«sin zapatos, sin pantalones» (sin «desnuda/naked») → cuerpo completo.

    Ninguna cláusula mide body_exposure_full >= 0.3 individualmente, pero dos
    cláusulas de uncover juntas piden desnudar a la persona. El sufijo de
    estado final debe escribirse y la zona debe ser toda la persona.
    """
    manager = ImgEditorManager()
    # Full-body exposure is confirmed by the multi-clause uncover fallback.
    assert manager.measured_full_body_exposure("debe ir descalza y sin pantalones"), (
        "multi-clause uncover did not confirm full body exposure")
    assert manager.measured_full_body_exposure("sin zapatos, sin pantalones"), (
        "multi-clause uncover (ES) did not confirm full body exposure")
    # The end-state sentence is written from the measured clauses.
    suffix = manager._body_state_clause(
        "debe ir descalza y sin pantalones", "uncover", 0.74)
    assert "is fully nude" in suffix, f"full-body suffix missing: {suffix!r}"
    assert "bare skin" in suffix, f"bare skin sentence missing: {suffix!r}"
    # A single uncover clause is NOT promoted to full exposure (regression guard).
    assert not manager.measured_full_body_exposure("sin zapatos"), (
        "single uncover clause was promoted to full body exposure")
    single_suffix = manager._body_state_clause(
        "sin zapatos", "uncover", 0.74)
    assert "is fully nude" not in single_suffix, (
        f"single clause got full-body suffix: {single_suffix!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    cases = [measured_body_change_writes_the_measured_zone,
             body_change_with_other_people_preserves_the_frame,
             body_change_without_other_people_writes_the_measured_zone,
             garment_fragment_widens_to_the_person,
             multi_clause_uncover_writes_full_body_suffix,
             appearance_change_on_a_head_uses_the_focal_recipe,
             pose_change_reinterprets_the_frame,
             clothing_change_does_not_reinterpret_the_frame,
             unmeasured_instruction_is_reported_not_hidden,
             ordinary_edit_keeps_its_own_regime,
             explicit_longcat_selection_uses_the_measured_native_recipe,
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
