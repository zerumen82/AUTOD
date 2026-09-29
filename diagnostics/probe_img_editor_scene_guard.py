"""CPU-only checks for the compound location edit guard."""
from types import SimpleNamespace

import numpy as np
from PIL import Image

from roop.img_editor.img_editor_manager import ImgEditorManager


clauses = ImgEditorManager._split_compound_instruction(
    "the man must be bald, and they're in an office"
)
assert clauses == ["the man must be bald", "they're in an office"], clauses
assert ImgEditorManager._is_scene_location_clause(clauses[1])
assert ImgEditorManager._is_scene_location_clause("she is at a beach")
assert not ImgEditorManager._is_scene_location_clause("make the walls gray")

photo = Image.new("RGB", (100, 80))
people = np.zeros((80, 100), dtype=np.uint8)
people[20:70, 10:35] = 255
people[15:70, 55:85] = 255
faces = [
    SimpleNamespace(bbox=(14, 25, 30, 43)),
    SimpleNamespace(bbox=(62, 22, 78, 40)),
]


class FakeMasker:
    def __init__(self, mask):
        self.mask = mask

    def generate_mask(self, image, query):
        return Image.fromarray(self.mask, "L") if query == "person" else None


mask = ImgEditorManager._background_mask_from_people(photo, FakeMasker(people), faces)
assert mask is not None
array = np.asarray(mask)
assert array[30, 20] == 0 and array[30, 70] == 0
assert array[5, 5] == 255 and array[5, 95] == 255
original = Image.new("RGB", photo.size, "blue")
generated = Image.new("RGB", photo.size, "red")
composed = np.asarray(Image.composite(generated, original, mask))
assert tuple(composed[30, 20]) == (0, 0, 255)
assert tuple(composed[5, 5]) == (255, 0, 0)
missing_person = people.copy()
missing_person[:, 55:85] = 0
assert ImgEditorManager._background_mask_from_people(
    photo, FakeMasker(missing_person), faces,
) is None


class SceneMasker:
    def __init__(self, generated_people):
        self.generated_people = generated_people

    def generate_mask(self, image, query):
        if query != "person":
            return None
        pixels = self.generated_people if image.getpixel((0, 0)) == (255, 0, 0) else people
        return Image.fromarray(pixels, "L")


manager = ImgEditorManager()
scene_masker = SceneMasker(people)
source_background = manager._background_mask_from_people(original, scene_masker, faces)
assert source_background is not None
scene, reason, head_mask = manager._compose_scene_preserving_heads(
    original, generated, source_background, faces, scene_masker,
)
assert scene is not None, reason
assert head_mask is not None
assert scene.getpixel((5, 5)) == (255, 0, 0)
shifted = np.zeros_like(people)
shifted[20:70, 50:99] = 255
scene, reason, head_mask = manager._compose_scene_preserving_heads(
    original, generated, source_background, faces, SceneMasker(shifted),
)
assert scene is None, reason
assert head_mask is None
print("[PROBE] scene guard 11/11, CPU only")
