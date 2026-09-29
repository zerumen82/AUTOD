"""Exercise ordinary LongCat dispatch without running an image generator."""
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from roop.img_editor.img_editor_manager import ImgEditorManager


class FakeClient:
    def __init__(self):
        self.calls = []

    def load(self, version):
        self.calls.append(("load", version))
        return True, "OK"

    def generate(self, **kwargs):
        self.calls.append(("generate", kwargs))
        return SimpleNamespace(image=Image.new("RGB", (32, 20), "red")), "OK (stub)"


def main():
    cases = (
        (
            "la mujer va descalza y sin medias, el hombre blanco que está de rodillas es calvo",
            "the woman goes barefoot and without stockings, "
            "the white man who is on his knees is bald",
        ),
        ("Add a person near the doorway", "Add a person near the doorway"),
        ("Remove the person on the left", "Remove the person on the left"),
        ("Add a lamp to the table", "Add a lamp to the table"),
        ("Remove the chair", "Remove the chair"),
    )
    source = Image.new("RGB", (48, 32), "white")
    manager = ImgEditorManager()
    assert manager._explicit_viewpoint_request("cambia el pov", "change the viewpoint")
    assert manager._explicit_viewpoint_request("cambia el ángulo a vista cenital", "show a top view")
    assert manager._explicit_viewpoint_request("cambia la vista de la habitación", "change the room view")
    assert manager._explicit_viewpoint_request("Show the room from the opposite angle", "Show the room from the opposite angle")
    assert not manager._explicit_viewpoint_request(
        "el hombre de perfil debe llevar sombrero", "the man in profile must wear a hat"
    )
    assert not manager._explicit_viewpoint_request(
        "cambia el pelo de la mujer de perfil", "change the hair of the woman in profile"
    )
    assert not manager._explicit_viewpoint_request(
        "añade una ventana con vista al mar", "add a window with a sea view"
    )
    assert not manager._explicit_viewpoint_request("Pon la pared azul", "Make the wall blue")
    with (patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
          patch("roop.face_util.get_face_analyser", return_value=None),
          patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
          patch.object(manager, "_release_local_gpu_before_comfy")):
        for prompt, translated in cases:
            client = FakeClient()
            with (patch("roop.img_editor.prompt_translator.translate_prompt", return_value=translated),
                  patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client", return_value=client)):
                output, message, mask = manager.generate_intelligent(
                    image=source, prompt=prompt, engine="longcat", use_rewriter=True,
                    use_semantic=True,
                )
            assert output is not None and output.size == source.size and mask is None
            assert "cumplimiento visual no verificado" in message, (prompt, message)
            if manager._explicit_viewpoint_request(prompt, translated):
                assert "POV no verificado" in message, (prompt, message)
            assert [name for name, _ in client.calls] == ["load", "generate"]
            generation = client.calls[-1][1]
            assert translated in generation["prompt"]
            assert "Keep people anatomically coherent" in generation["prompt"]
            assert generation["num_inference_steps"] == 12
            assert generation["no_kontext"] is True
            print(f"OK: {prompt} -> {message}")

        client = FakeClient()
        finish_calls = []
        def fake_finish(image, original, **kwargs):
            finish_calls.append((image.size, original.size, kwargs))
            return Image.new("RGB", (64, 42), "red"), "acabado simulado", {"scores": {}}
        with (patch("roop.img_editor.prompt_translator.translate_prompt", return_value="Make the wall blue"),
              patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client", return_value=client),
              patch("roop.img_editor.top_tier_finish.run_quality_enhance_after_edit", side_effect=fake_finish)):
            output, message, mask = manager.generate_intelligent(
                image=source, prompt="Pon la pared azul", engine="longcat",
                quality_mode=True, enhance_tier="4k", quality_preserve_faces=False,
            )
        assert output is not None and output.size == (64, 42) and mask is None
        assert [name for name, _ in client.calls] == ["load", "generate"]
        assert "Keep every person, object, room detail" in client.calls[-1][1]["prompt"]
        assert "Keep people anatomically coherent" in client.calls[-1][1]["prompt"]
        assert finish_calls[0][0:2] == ((32, 20), source.size)
        assert finish_calls[0][2]["enhance_tier"] == "4k"
        assert finish_calls[0][2]["preserve_faces"] is False
        assert finish_calls[0][2]["use_generative"] is False
        assert "Turbo" in message
        print("OK: Calidad con prompt -> Turbo único + acabado clásico, sin restauración facial")


if __name__ == "__main__":
    main()
