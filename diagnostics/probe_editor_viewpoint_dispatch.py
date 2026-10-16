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
            return None, "Sin muestreo (300s)"

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

    # Presupuesto real de la ruta de cámara: 2026-10-16 la carga fría + prep
    # tardó 202 s en llegar al KSampler (job interrumpido con los 14 nodos
    # previos ejecutados) y el stall de 150 s lo mataba justo antes de
    # muestrear, así que "cambia el pov" salía sin cambios.
    from roop.img_editor.qwen_camera_comfy_client import QwenCameraComfyClient

    captured = []

    def _capturing_wait(base_url, prompt_id, **kwargs):
        captured.append(kwargs)
        return None, f"Sin muestreo ({kwargs.get('stall_timeout', 0):.0f}s)"

    def _post(url, **kwargs):
        body = ({"name": "cam.png"} if url.endswith("/upload/image")
                else {"prompt_id": "probe-budget"})
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: body)

    with (
        patch("roop.img_editor.qwen_camera_comfy_client.get_comfyui_url",
              return_value="http://comfy.test"),
        patch("roop.img_editor.qwen_camera_comfy_client._server_vram_free_mb",
              return_value=(8000, 8192)),
        patch("roop.img_editor.qwen_camera_comfy_client._queue_is_idle",
              return_value=True),
        patch("roop.img_editor.qwen_camera_comfy_client._sync_unload",
              return_value=True),
        patch("roop.img_editor.qwen_camera_comfy_client.wait_for_comfy_image",
              side_effect=_capturing_wait),
        patch("roop.img_editor.qwen_camera_comfy_client.requests.post",
              side_effect=_post),
        patch.object(QwenCameraComfyClient, "available",
                     new=lambda self: (True, "OK")),
    ):
        budget_client = QwenCameraComfyClient()
        budget_result, budget_message = budget_client.generate(
            Image.new("RGB", (48, 32), "white"),
            "<sks> front-right quarter view elevated shot medium shot",
        )
    assert budget_result is None, budget_message
    assert captured, "wait_for_comfy_image must be reached"
    first = captured[0]
    assert first.get("stall_timeout", 0) >= 240, (
        f"stall_timeout={first.get('stall_timeout')} kills the measured 202 s "
        "cold load before its first sampling step"
    )
    assert first["timeout"] > first["stall_timeout"], first
    assert 1 <= len(captured) <= 2, captured
    for call in captured:
        assert 240 <= call["stall_timeout"] < call["timeout"] <= 720, call
    print(
        "OK: camera budget covers the measured 202 s cold load "
        f"(stall={first['stall_timeout']:.0f}s, timeout={first['timeout']:.0f}s, "
        f"attempts={len(captured)})"
    )
    print("OK: camera first, remaining edit second, explicit pose and uncertainty reported")


if __name__ == "__main__":
    main()
