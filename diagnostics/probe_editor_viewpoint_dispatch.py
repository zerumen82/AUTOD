"""Check that camera and compound edits use the two-step route without GPU work."""

from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from roop.img_editor.img_editor_manager import ImgEditorManager
from roop.img_editor.qwen_camera_comfy_client import camera_pose, remaining_edit


class FakeClient:
    def __init__(self):
        self.calls = []

    def load(self, flux_version):
        self.calls.append(("load", flux_version))
        return True, "OK"

    def generate(self, **kwargs):
        self.calls.append(("generate", kwargs))
        return SimpleNamespace(image=Image.new("RGB", (48, 32), "blue")), "OK (stub)"


class FakeCamera:
    calls = []

    def generate(self, image, pose, **kwargs):
        self.calls.append((image.size, pose))
        return SimpleNamespace(image=Image.new("RGB", (48, 32), "green")), "OK"


def main():
    manager = ImgEditorManager()
    client = FakeClient()
    camera = FakeCamera()
    with (
        patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
        patch("roop.face_util.get_face_analyser", return_value=None),
        patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
        patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client", return_value=client),
        patch("roop.img_editor.qwen_camera_comfy_client.QwenCameraComfyClient", return_value=camera),
        patch.object(manager, "_release_local_gpu_before_comfy"),
    ):
        output, message, _ = manager.generate_intelligent(
            image=Image.new("RGB", (48, 32), "white"),
            prompt="Show the room from an overhead camera angle, make the wall blue",
            engine="longcat",
            use_semantic=False,
            use_rewriter=False,
            enhance_faces=False,
        )
    assert output is not None, message
    assert camera.calls == [((48, 32), "<sks> front view high-angle shot medium shot")]
    assert any(
        name == "load" and version == "LongCat-Image-Edit-Turbo-Q4_K_S.gguf"
        for name, version in client.calls
    ), client.calls
    assert any(name == "generate" for name, _ in client.calls), client.calls
    assert "Qwen Camera" in message and "LongCat Turbo" in message, message
    assert "make the wall blue" in client.calls[-1][1]["prompt"].lower()
    assert "camera angle" not in client.calls[-1][1]["prompt"].lower()
    assert remaining_edit("cambia el POV", "change the viewpoint") == ""
    assert camera_pose("desde arriba, plano general") == "<sks> front view high-angle shot wide shot"
    assert camera_pose("desde atras a la izquierda") == (
        "<sks> back-left quarter view eye-level shot medium shot"
    )

    class FailedCamera:
        def generate(self, image, pose, **kwargs):
            return None, "Sin muestreo (150s)"

    failed = FailedCamera()
    with (
        patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
        patch("roop.face_util.get_face_analyser", return_value=None),
        patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
        patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client", return_value=client),
        patch("roop.img_editor.qwen_camera_comfy_client.QwenCameraComfyClient", return_value=failed),
        patch.object(manager, "_release_local_gpu_before_comfy"),
    ):
        client.calls.clear()
        source = Image.new("RGB", (48, 32), "white")
        output, message, _ = manager.generate_intelligent(
            image=source,
            prompt="cambia el POV",
            engine="longcat",
            use_semantic=False,
            use_rewriter=False,
            enhance_faces=False,
        )
    assert list(output.getdata()) == list(source.getdata()), "a failed camera pass must keep the input pixels"
    assert client.calls == [], client.calls
    assert message.startswith("POV no aplicado"), message
    print("OK: camera first, remaining edit second, explicit pose and uncertainty reported")


if __name__ == "__main__":
    main()
