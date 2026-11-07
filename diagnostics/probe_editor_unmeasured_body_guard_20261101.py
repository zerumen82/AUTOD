"""A body edit without a measured subject must not regenerate the whole photo."""
from unittest.mock import patch

from PIL import Image

from roop.img_editor.img_editor_manager import ImgEditorManager


def main():
    prompt = "LA MUJER DE AZUL DEBE IR SIN ZAPATOS"
    translated = "The woman in blue must go without shoes"
    source = Image.new("RGB", (48, 32), "white")
    manager = ImgEditorManager()

    class FakeClient:
        def __init__(self):
            self.calls = []

        def load(self, *args, **kwargs):
            self.calls.append(("load", args, kwargs))
            return True, "OK"

        def generate(self, **kwargs):
            self.calls.append(("generate", kwargs))
            return None, "unexpected generation"

    client = FakeClient()
    with (
        patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
        patch("roop.face_util.get_face_analyser", return_value=None),
        patch("roop.face_util.detect_faces_robust", return_value=[]),
        patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
        patch.object(manager, "_release_local_gpu_before_comfy"),
        patch.object(manager, "_measured_subject_passes", return_value=[]) as measure,
        patch("roop.img_editor.prompt_translator.translate_prompt", return_value=translated),
        patch(
            "roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
            return_value=client,
        ),
    ):
        output, message, _mask = manager.generate_intelligent(
            image=source, prompt=prompt, engine="qwen_camera",
        )

    facts = manager._prompt_authority_facts(prompt, translated)
    assert facts["body"] and not facts["pose"], facts
    measure.assert_called_once()
    assert output is not None and list(output.getdata()) == list(source.getdata())
    assert "cuerpo no medido; se conserva la foto" in message, message
    assert client.calls == [], client.calls
    print("OK: cuerpo sin sujeto medido conserva la foto y no carga ComfyUI")


if __name__ == "__main__":
    main()
