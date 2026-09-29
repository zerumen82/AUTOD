"""Check that measured hair failures retry and keep the best real candidate."""
from unittest.mock import patch

from PIL import Image

from ui.tabs.img_editor import events, state


class FakeManager:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = 0
        self._last_instruction_compliance = None

    def generate_intelligent(self, **kwargs):
        self.calls += 1
        image, message, score = next(self.replies)
        self._last_instruction_compliance = (
            {"compliant": False, "source": "hair_mask", "hair_after": score}
            if score is not None else {"compliant": True}
        )
        return image, message, None

    def _release_local_gpu_before_comfy(self):
        pass


def run(replies):
    manager = FakeManager(replies)
    with (patch.object(events, "get_img_editor_manager", return_value=manager),
          patch.object(events, "_cleanup_vram"),
          patch.object(events, "get_img_editor_output_dir", return_value="diagnostics"),
          patch.object(Image.Image, "save")):
        result, message, _ = events.on_generate(
            Image.new("RGB", (8, 8)), "make the man bald",
            "longcat", 0.0, False, 3, False, "hd",
        )
    assert not state.is_generating
    return manager, result, message


red = Image.new("RGB", (8, 8), "red")
green = Image.new("RGB", (8, 8), "green")
blue = Image.new("RGB", (8, 8), "blue")

manager, result, message = run(((red, "hair remains", 0.6), (green, "less hair", 0.2)))
assert manager.calls == 2 and result is green and "less hair" in message

manager, result, message = run(((red, "hair remains", 0.6), (blue, "done", None)))
assert manager.calls == 2 and result is blue and "done" in message

manager, result, message = run(((red, "hair remains", 0.6), (None, "no se reintenta", None)))
assert manager.calls == 2 and result is red and "hair remains" in message

print("[PROBE] measured hair retry 3/3")
