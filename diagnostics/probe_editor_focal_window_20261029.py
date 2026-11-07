"""Real CPU geometry regressions for local editing; no model or inference doubles."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402


class FocalWindowContracts(unittest.TestCase):
    def setUp(self):
        self.manager = ImgEditorManager()

    def scene(self, size, region):
        width, height = size
        y, x = np.indices((height, width))
        photo = Image.fromarray(np.stack((x % 256, y % 256, (x + y) % 256), axis=2).astype(np.uint8))
        mask = Image.new("L", size, 0)
        ImageDraw.Draw(mask).rectangle(region, fill=255)
        return photo, mask

    def assert_window(self, photo, mask, person=None):
        source = np.array(photo)
        source_mask = np.array(mask)
        work, work_mask, box = self.manager._appearance_work_region(photo, mask, person)
        self.assertIsNotNone(box)
        x1, y1, x2, y2 = box
        self.assertTrue(0 <= x1 < x2 <= photo.width)
        self.assertTrue(0 <= y1 < y2 <= photo.height)
        self.assertEqual(work.size, work_mask.size)
        np.testing.assert_array_equal(np.asarray(work), source[y1:y2, x1:x2])
        np.testing.assert_array_equal(np.asarray(work_mask), source_mask[y1:y2, x1:x2])
        self.assertEqual(np.count_nonzero(work_mask), np.count_nonzero(source_mask))
        if person is not None:
            self.assertEqual(np.count_nonzero(person.crop(box)), np.count_nonzero(person))
        np.testing.assert_array_equal(np.asarray(photo), source)
        np.testing.assert_array_equal(np.asarray(mask), source_mask)
        return box

    def test_local_edit_without_person(self):
        for size, region in (((640, 420), (280, 80, 350, 150)),
                             ((400, 600), (250, 80, 350, 240)),
                             ((1600, 1200), (760, 540, 839, 659))):
            with self.subTest(size=size):
                photo, mask = self.scene(size, region)
                self.assertNotEqual(self.assert_window(photo, mask), (0, 0, *size))

    def test_context_contains_measured_person(self):
        photo, mask = self.scene((640, 420), (280, 80, 350, 150))
        person = Image.new("L", photo.size, 0)
        ImageDraw.Draw(person).rectangle((70, 15, 590, 400), fill=255)
        self.assert_window(photo, mask, person)

    def test_local_edit_at_photo_edges(self):
        for region in ((0, 0, 55, 55), (584, 364, 639, 419)):
            with self.subTest(region=region):
                photo, mask = self.scene((640, 420), region)
                self.assert_window(photo, mask)

    def test_no_mask_preserves_original(self):
        photo = Image.new("RGB", (640, 420), "blue")
        work, mask, box = self.manager._appearance_work_region(photo, None)
        self.assertIs(work, photo)
        self.assertIsNone(mask)
        self.assertIsNone(box)

    def test_empty_mask_preserves_original(self):
        photo = Image.new("RGB", (640, 420), "blue")
        mask = Image.new("L", photo.size, 0)
        work, result_mask, box = self.manager._appearance_work_region(photo, mask)
        self.assertIs(work, photo)
        self.assertIs(result_mask, mask)
        self.assertIsNone(box)


if __name__ == "__main__":
    unittest.main(verbosity=2)
