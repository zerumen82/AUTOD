"""CPU dispatch contract: a measured region is not replaced by face boxes.

Synthetic pixels and explicit detector/client fixtures only. The real
generate_intelligent dispatcher, window geometry and paste are executed;
no model inference, server requests, downloads or UI startup are allowed.
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402


class CaptureClient:
    """Declared CPU fixture, never sends work to a backend."""

    def __init__(self):
        self.calls = []

    def load(self, *args, **kwargs):
        return True, "CPU fixture"

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(image=Image.new("RGB", kwargs["image"].size, (30, 70, 230))), "CPU fixture"


class MeasuredScopeDispatch(unittest.TestCase):
    def test_measured_core_and_write_support_survive_dispatch(self):
        width, height = 480, 360
        pixels = np.random.default_rng(411).integers(90, 170, (height, width, 3), dtype=np.uint8)
        photo = Image.fromarray(pixels)
        region = np.zeros((height, width), dtype=np.uint8)
        region[50:130, 95:250] = 255
        measured = Image.fromarray(region, "L")
        person_arr = np.zeros_like(region)
        person_arr[30:330, 80:260] = 255
        person = Image.fromarray(person_arr, "L")
        other_person = np.zeros_like(region, dtype=bool)
        other_person[30:330, 270:420] = True
        target = SimpleNamespace(bbox=np.array([150, 80, 225, 125]), det_score=0.99, age=35)
        other = SimpleNamespace(bbox=np.array([280, 60, 340, 140]), det_score=0.99, age=35)
        faces = [target, other]
        item = {"mask": measured, "person": person, "text": "The person on the left must be bald.",
                "measured_request": True, "deliver_region": False, "coverage": float((region > 0).mean())}
        manager = ImgEditorManager()
        client = CaptureClient()
        log = io.StringIO()
        with (
            patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
            patch("ui.tabs.comfy_launcher.start", side_effect=AssertionError("no backend startup")),
            patch("roop.face_util.get_face_analyser", return_value=object()),
            patch("roop.face_util.detect_faces_robust", return_value=faces),
            patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
            patch("roop.img_editor.prompt_translator.translate_prompt", return_value=item["text"]),
            patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client", return_value=client),
            patch.object(manager, "_credible_faces", return_value=faces),
            patch.object(manager, "_release_local_gpu_before_comfy"),
            patch.object(manager, "_measured_subject_passes", return_value=[item]),
            patch.object(manager, "_narrow_appearance_passes", side_effect=lambda image, passes, *args: passes),
            patch.object(manager, "_head_zone_for_subject", return_value=None),
            patch.object(manager, "_with_covered_reference", side_effect=lambda sample, *args: (sample, None)),
            patch.object(manager, "_detect_edit_faces", return_value=faces),
            patch.object(manager, "_added_face_mask", return_value=None),
            patch.object(manager, "_degenerate_faces_mask", return_value=None),
            patch.object(manager, "_appearance_work_region", wraps=manager._appearance_work_region) as window,
            contextlib.redirect_stdout(log),
        ):
            output, message, _ = manager.generate_intelligent(
                image=photo, prompt=item["text"], engine="longcat", seed=424242,
            )
        print(log.getvalue())
        self.assertIsNotNone(output, message)
        self.assertEqual(len(client.calls), 1, message)
        self.assertEqual(window.call_count, 1, message)
        dispatched = np.asarray(window.call_args.args[1].convert("L"))
        changed = np.any(np.asarray(output) != pixels, axis=2)
        core = region > 0
        erased = int(np.count_nonzero(core & (dispatched < 96)))
        added = int(np.count_nonzero((dispatched >= 96) & ~core))
        outside_measurement = int(np.count_nonzero(changed & ~core))
        outside_person = int(np.count_nonzero(changed & (person_arr == 0)))
        other_changed = int(np.count_nonzero(changed & other_person))
        print(f"scope: erased_core={erased}, added_pixels={added}, outside_measurement={outside_measurement}, outside_person={outside_person}, other_person={other_changed}")
        with self.subTest("measured core is not erased by target face bbox"):
            self.assertEqual(erased, 0)
        with self.subTest("no unmeasured dilation replaces measured mask"):
            self.assertEqual(added, 0)
        with self.subTest("write support stays inside measured region"):
            self.assertEqual(outside_measurement, 0)
        with self.subTest("write support stays inside measured person"):
            self.assertEqual(outside_person, 0)
        with self.subTest("other person remains identical"):
            self.assertEqual(other_changed, 0)
        with self.subTest("requested measured core really reaches output"):
            self.assertTrue(changed[90:115, 170:210].all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
