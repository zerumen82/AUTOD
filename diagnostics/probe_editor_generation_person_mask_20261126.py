"""Check that local appearance generation receives the measured subject.

Uses a real neutral photo and segmenter. The client records arguments; no image
generation or UI restart occurs.
"""

import io
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
from skimage import data

from roop.face_util import get_face_analyser_cpu
from roop.img_editor.gen_prompt_config import get_longcat_model_config
from roop.img_editor.img_editor_manager import ImgEditorManager


class CaptureClient:
    def __init__(self):
        self.calls = []

    def load(self, version):
        return True, "diagnostic client ready"

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(image=kwargs["image"].copy()), "diagnostic capture"


def main():
    manager = ImgEditorManager()
    client = CaptureClient()
    image = Image.fromarray(data.astronaut()).convert("RGB")
    observed = []
    pasted = []
    original_region = manager._appearance_work_region
    original_integrate = manager._integrate_local_generation

    def capture_region(photo, mask, person=None):
        result = original_region(photo, mask, person)
        if person is not None:
            observed.append((mask, person, result[1]))
        return result

    def capture_integrate(photo, generated, mask, box, **kwargs):
        pasted.append((mask, kwargs.get("limit_mask"), kwargs))
        return original_integrate(photo, generated, mask, box, **kwargs)

    buffer = io.StringIO()
    with (
        redirect_stdout(buffer),
        patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
        patch("roop.face_util.get_face_analyser", new=lambda *a, **k: get_face_analyser_cpu()),
        patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
        patch.object(manager, "_release_local_gpu_before_comfy"),
        patch("roop.img_editor.prompt_translator.translate_prompt", new=lambda text, *a, **k: text),
        patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client", return_value=client),
        patch.object(manager, "_appearance_work_region", side_effect=capture_region),
        patch.object(manager, "_integrate_local_generation", side_effect=capture_integrate),
    ):
        manager.generate_intelligent(
            image=image, prompt="The person must be bald.", engine="longcat",
        )
    assert observed and client.calls, "local appearance did not reach the client"
    proposed, person, work_mask = observed[0]
    subject = np.asarray(person.convert("L")) >= 128
    focus = np.asarray(proposed.convert("L")) >= 128
    assert np.array_equal(focus, subject), (
        "generation reverted from the measured person to the smaller head"
    )
    sent = np.asarray(work_mask.convert("L")) >= 128
    recipe = get_longcat_model_config()
    assert client.calls[0]["mask_image"] is None
    assert client.calls[0]["num_inference_steps"] == int(recipe["steps"])
    assert client.calls[0]["reference_latents_method"] == recipe["reference_latents_method"]
    assert pasted and pasted[0][2]["keep_measured_mask"] is True
    if pasted:
        output = Path("diagnostics/out/editor_person_mask_20261126")
        output.mkdir(parents=True, exist_ok=True)
        image.save(output / "source.png")
        proposed.save(output / "generation_person.png")
        pasted[0][0].save(output / "paste_head.png")
        if pasted[0][1] is not None:
            pasted[0][1].save(output / "paste_limit.png")
    print(
        f"measured person={int(subject.sum())} px; "
        f"generation focus={int(focus.sum())} px; crop mask={int(sent.sum())} px"
    )
    print("PASS: local generation receives the measured subject")


if __name__ == "__main__":
    main()
