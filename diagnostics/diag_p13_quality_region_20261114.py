"""Reproduce the P13 quality-mode case with a MEASURED region (probe contract)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from roop.img_editor.img_editor_manager import get_img_editor_manager


ROOT = Path(__file__).resolve().parents[1]


class FakeClient:
    def __init__(self):
        self.calls = []

    def load(self, version=None, **params):
        self.calls.append(("load", version or params.get("flux_version")
                           or params.get("qwen_version")))
        return True, "OK"

    def generate(self, **kwargs):
        self.calls.append(("generate", kwargs))
        return SimpleNamespace(image=Image.new("RGB", (32, 20), "red")), "OK (stub)"


def main():
    source = Image.new("RGB", (48, 32), "white")
    measured_room = Image.new("L", source.size, 0)
    measured_room.paste(255, (5, 5, 25, 20))
    manager = get_img_editor_manager()
    client = FakeClient()
    finish_calls = []

    def fake_finish(image, original, **kwargs):
        finish_calls.append((image.size, original.size, kwargs))
        return Image.new("RGB", (64, 42), "red"), "acabado simulado", {"scores": {}}

    with patch(
        "roop.img_editor.prompt_translator.translate_prompt",
        return_value="Make the wall blue",
    ), patch(
        "roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
        return_value=client,
    ), patch(
        "roop.img_editor.top_tier_finish.run_quality_enhance_after_edit",
        side_effect=fake_finish,
    ), patch.object(
        manager, "_get_clipseg_masker",
        return_value=SimpleNamespace(generate_mask=lambda *_args: measured_room),
    ):
        output, message, mask = manager.generate_intelligent(
            image=source, prompt="Pon la pared azul", engine="qwen_camera",
            quality_mode=True, enhance_tier="4k", quality_preserve_faces=False,
        )
    print("output:", None if output is None else output.size)
    print("mask:", mask)
    print("message:", message)
    print("calls:", [name for name, _ in client.calls])
    print("finish_calls:", finish_calls)


if __name__ == "__main__":
    main()
