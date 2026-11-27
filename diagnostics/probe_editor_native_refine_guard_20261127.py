"""Replay a recorded UI generation to catch a destructive automatic second pass.

The source is the latest saved UI image and the model output is the recorded
first LongCat sample. No new model inference or UI restart is performed.
"""

from contextlib import nullcontext, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402


class RecordedClient:
    def __init__(self, sample):
        self.sample = sample
        self.calls = []

    def load(self, *_args, **_kwargs):
        return True, "recorded sample"

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(image=self.sample), "recorded first pass"


def main():
    force_retry = "--retry" in sys.argv
    source = Image.open(ROOT / "img_editor/edit_1795733254.png").convert("RGB")
    raw = Image.open(ROOT / "ui/tob/ComfyUI/output/FastEdit_01438_.png").convert("RGB")
    client = RecordedClient(raw)
    manager = ImgEditorManager()
    capture = StringIO()
    paste_calls = []
    integrate_original = manager._integrate_local_generation

    def record_paste(*args, **kwargs):
        paste_calls.append(kwargs)
        return integrate_original(*args, **kwargs)

    retry_gate = (
        patch.object(manager, "_edit_is_nearly_unchanged", side_effect=[True, False])
        if force_retry else nullcontext()
    )
    with (
        redirect_stdout(capture),
        retry_gate,
        patch.object(manager, "_integrate_local_generation", side_effect=record_paste),
        patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
        patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client", return_value=client),
        patch("roop.img_editor.prompt_translator.translate_prompt", return_value="Man must be bald."),
    ):
        result, message, _mask = manager.generate_intelligent(
            image=source,
            prompt="EL HOMBRE DEBE SER CALVO",
            engine="auto",
            scope_hint="features",
            use_rewriter=False,
            enhance_faces=False,
            auto_upscale=False,
        )
    log = capture.getvalue()
    print("\n".join(line for line in log.splitlines() if (
        "Selector «features»" in line
        or "Apariencia local: edición nativa" in line
        or "Refino focal" in line
        or "Región integrada" in line
    )))
    if not force_retry:
        assert result is not None, message
    assert "Selector «features»: eje local_appearance elevado" in log
    assert "Apariencia local: edición nativa" in log
    assert "Refino focal omitido: la edición local nativa" in log
    expected_calls = 2 if force_retry else 1
    assert len(client.calls) == expected_calls, (
        f"expected {expected_calls} model calls, got {len(client.calls)}"
    )
    assert client.calls[0].get("mask_image") is None
    if force_retry:
        assert "Refino focal del reintento omitido" in log
        assert client.calls[1].get("mask_image") is None
        assert len(paste_calls) >= 2
        assert paste_calls[0].get("expand_head_support") == paste_calls[1].get(
            "expand_head_support"
        )
        assert paste_calls[0].get("source_faces") is paste_calls[1].get(
            "source_faces"
        )
    print(f"PASS: {expected_calls} native generation(s), no destructive focused refinement")


if __name__ == "__main__":
    main()
