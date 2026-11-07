"""A subject mask without face overlap must still drive a localized body edit."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from roop.img_editor.img_editor_manager import ImgEditorManager


class FakeMasker:
    def __init__(self, size):
        self.subject = Image.new("L", size, 0)
        self.subject.paste(255, (30, 16, 54, 58))
        self.shoes = Image.new("L", size, 0)
        self.shoes.paste(255, (34, 52, 50, 58))
        self.subject_calls = []

    def generate_subject_mask(self, image, subject, template):
        self.subject_calls.append((subject, template))
        return self.subject.copy()

    def generate_mask(self, image, prompt):
        return self.shoes.copy() if "shoe" in prompt.casefold() else None


class FakeClient:
    def __init__(self):
        self.calls = []

    def load(self, *args, **kwargs):
        self.calls.append(("load", args, kwargs))
        return True, "OK"

    def generate(self, **kwargs):
        self.calls.append(("generate", kwargs))
        image = kwargs["image"]
        return SimpleNamespace(
            image=Image.new("RGB", image.size, (220, 30, 30)),
        ), "OK (stub)"


def main():
    source = Image.new("RGB", (96, 64), (90, 90, 90))
    masker = FakeMasker(source.size)
    client = FakeClient()
    manager = ImgEditorManager()
    other_faces = [
        SimpleNamespace(bbox=(1, 1, 10, 10), det_score=0.9),
        SimpleNamespace(bbox=(84, 1, 94, 10), det_score=0.8),
    ]

    with (
        patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
        patch("roop.face_util.get_face_analyser", return_value=object()),
        patch("roop.face_util.detect_faces_robust", return_value=other_faces),
        patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
        patch.object(manager, "_release_local_gpu_before_comfy"),
        patch.object(manager, "_get_clipseg_masker", return_value=masker),
        patch("roop.img_editor.prompt_translator.translate_prompt",
              return_value="The woman in blue must go without shoes"),
        patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
              return_value=client),
    ):
        output, message, _mask = manager.generate_intelligent(
            image=source,
            prompt="LA MUJER DE AZUL DEBE IR SIN ZAPATOS",
            engine="qwen_camera",
        )

    generations = [
        args for name, args, *_ in client.calls if name == "generate"
    ]
    assert output is not None, message
    assert generations, client.calls
    assert any("a photo of {subject}" == template
               for _subject, template in masker.subject_calls), masker.subject_calls
    assert all(call["mask_image"] is not None for call in generations), generations
    assert all(call["keep_source_pixels"] is True for call in generations), generations
    assert all(call["preserve_frame"] is True for call in generations), generations
    result = np.asarray(output.convert("RGB"))
    original = np.asarray(source.convert("RGB"))
    assert not np.array_equal(result[55, 40], original[55, 40]), result[55, 40]
    assert np.array_equal(result[40, 5], original[40, 5]), result[40, 5]
    print("OK: el sujeto sin solape facial se edita con máscara local y marco preservado")


if __name__ == "__main__":
    main()
