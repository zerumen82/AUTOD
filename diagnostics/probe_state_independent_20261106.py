"""Independent CPU contracts for residual rejection and retry support.

Images and masks are synthetic. Diffusion, semantic measurements and backend
availability are explicit doubles; production caller/composition dispatch is
executed. No inference, backend startup or downloads are permitted.
"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roop.img_editor import selected_region_edit as selected  # noqa: E402
from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402
from diagnostics import probe_editor_kontext_people_protection_20261018 as fixture  # noqa: E402


class StopBeforeInference(RuntimeError):
    """Stop the real caller on reaching its next existing route."""


class ResidualDispatch(unittest.TestCase):
    def described(self, *, both_fail=False, quality=False):
        manager = ImgEditorManager()
        yy, xx = np.indices((80, 80))
        source = Image.fromarray(np.stack((80 + xx, 50 + yy, 60 + xx), axis=-1).astype(np.uint8))
        subject = np.zeros((80, 80), dtype=bool)
        subject[10:70, 10:70] = True
        zone = np.zeros_like(subject)
        zone[35:45, 35:45] = True
        calls = []
        client = SimpleNamespace(load=Mock(return_value=(True, "CPU diffusion double")))

        def generate(_manager, _client, photo, measured, *_args, **_kwargs):
            calls.append(np.asarray(measured, dtype=bool).copy())
            altered = Image.fromarray(255 - np.asarray(photo))
            mask = Image.fromarray(np.asarray(measured, dtype=np.uint8) * 255)
            result = Image.composite(altered, photo, mask)
            note = selected.TARGET_RESIDUAL_FAIL if len(calls) == 1 or both_fail else "CPU target absent"
            return result, note

        with ExitStack() as stack:
            contexts = (
                patch.object(manager, "_prompt_authority_facts", return_value={
                    "body": True, "pose": False, "body_direction": "uncover"}),
                patch.object(manager, "_split_compound_instruction", return_value=[]),
                patch.object(manager, "_subject_queries", return_value=["person with scarf"]),
                patch.object(manager, "measured_body_regime", return_value={
                    "body": True, "direction": "uncover", "magnitude": 0.8,
                    "intensity": 0.8, "enforce_clause": "", "negative_prompt": "", "engine": {}}),
                patch.object(manager, "measured_full_body_exposure", return_value=False),
                patch.object(manager, "_requested_removal_target", return_value="scarf"),
                patch.object(manager, "_compose_generation_prompt", return_value="Remove the scarf"),
                patch.object(manager, "_compose_local_target_prompt", return_value="Remove the scarf"),
                patch.object(manager, "_release_local_gpu_before_comfy"),
                patch.object(manager, "_validate_visual_edit", side_effect=AssertionError(
                    "Residual failure must not be accepted as pixel-change success")),
                patch.object(selected, "_described_subject_mask", return_value=(subject, "person with scarf")),
                patch.object(selected, "_measured_clothing_zone", return_value=None),
                patch.object(selected, "_measured_target_zone", return_value=zone),
                patch.object(selected, "_vision_zone_pass", side_effect=generate),
                patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
                patch("ui.tabs.comfy_launcher.start", side_effect=AssertionError("Backend startup forbidden")),
                patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client", return_value=client),
                patch("roop.img_editor.top_tier_finish.run_top_tier_quality", side_effect=AssertionError(
                    "Residual failure must skip finishing")),
                patch("requests.sessions.Session.request", side_effect=AssertionError("HTTP forbidden")),
            )
            for context in contexts:
                stack.enter_context(context)
            result = selected.edit_described_subject(
                manager, source, "Remove the scarf", "Remove the scarf", [],
                engine="longcat", automatic=True, quality_mode=quality,
            )
        return source, result, calls, subject

    def test_retry_mask_describes_all_pixels_written(self):
        source, result, calls, subject = self.described()
        self.assertEqual(len(calls), 2)
        self.assertGreater(int(calls[1].sum()), int(calls[0].sum()))
        self.assertFalse(np.any(calls[1] & ~subject))
        written = np.any(np.asarray(result[0]) != np.asarray(source), axis=-1)
        declared = np.asarray(result[2].convert("L")) > 0
        self.assertFalse(np.any(written & ~declared), "Successful residual retry wrote outside its returned mask")
        np.testing.assert_array_equal(declared, calls[1])

    def test_persistent_residual_skips_quality_finish(self):
        _, result, calls, _ = self.described(both_fail=True, quality=True)
        self.assertEqual(len(calls), 2)
        self.assertIn(selected.TARGET_RESIDUAL_FAIL, result[1])

    def test_real_caller_does_not_accept_residual_as_visual_success(self):
        def described(_manager, image, *_args, **_kwargs):
            return Image.fromarray(255 - np.asarray(image)), selected.TARGET_RESIDUAL_FAIL, None

        with (
            patch.object(selected, "edit_described_subject", side_effect=described),
            patch.object(ImgEditorManager, "_validate_visual_edit", side_effect=AssertionError(
                "Residual marker must bypass pixel validation acceptance")),
            patch.object(ImgEditorManager, "_automatic_appearance_engine", side_effect=StopBeforeInference("next route")),
        ):
            with self.assertRaisesRegex(StopBeforeInference, "next route"):
                fixture._execute(engine="auto")

    def test_growth_preserves_holes_and_disjoint_neighbor_exclusion(self):
        zone = np.zeros((80, 90), dtype=bool)
        zone[35:40, 25:30] = True
        support = np.zeros_like(zone)
        support[10:70, 10:45] = True
        support[20:50, 22:24] = False  # excluded hole inside the selected subject
        grown = selected._expand_zone_within_support(zone, support)
        self.assertTrue(np.all(grown[zone]))
        self.assertGreater(int(grown.sum()), int(zone.sum()))
        self.assertFalse(np.any(grown & ~support))
        self.assertFalse(np.any(grown[:, 45:]))  # disjoint adjacent subject
        self.assertFalse(np.any(grown[20:50, 22:24]))


class IndependentResidualPolicy(unittest.TestCase):
    def setUp(self):
        self.settings = deepcopy(selected.selection.load_settings())
        self.settings["edit"].update({
            "residual_failure_fraction": 0.4,
            "residual_retry_growth_fraction": 0.35,
            "residual_retry_min_radius_pixels": 8,
        })
        self.zone = np.zeros((30, 30), dtype=bool)
        self.zone[10:20, 10:20] = True
        reader = patch.object(selected.selection, "load_settings", return_value=self.settings)
        reader.start()
        self.addCleanup(reader.stop)

    def test_explicit_overrides_reject_non_numeric_and_non_finite_values(self):
        for value in (float("nan"), float("inf"), True, "0.4", -0.1, 1.1):
            with self.subTest(parameter="threshold", value=value):
                with self.assertRaises(ValueError):
                    selected._residual_ratio_failed(50, self.zone, threshold=value)
            with self.subTest(parameter="growth", value=value):
                with self.assertRaises(ValueError):
                    selected._expand_zone_within_support(self.zone, self.zone, grow_frac=value)

    def test_zero_growth_still_cannot_write_protected_holes(self):
        self.settings["edit"].update({
            "residual_retry_growth_fraction": 0.0,
            "residual_retry_min_radius_pixels": 0,
        })
        support = self.zone.copy()
        support[12:16, 12:16] = False
        result = selected._expand_zone_within_support(self.zone, support)
        np.testing.assert_array_equal(result, support)


class IndependentVisionComposition(unittest.TestCase):
    def test_real_compositor_preserves_holes_neighbor_and_frame_edge(self):
        yy, xx = np.indices((100, 100))
        pixels = np.stack((50 + xx, 65 + yy, 80 + (xx + yy) // 2), axis=-1).astype(np.uint8)
        source = Image.fromarray(pixels)

        class DiffusionDouble:
            def generate(self, **kwargs):
                generated = np.asarray(kwargs["image"]).copy()
                generated[::2, :, 0] = 255
                generated[:, ::2, 1] = 0
                return SimpleNamespace(image=Image.fromarray(generated)), "CPU adversarial diffusion"

        class ResidualDouble:
            def generate_mask(self, image, _query):
                return Image.new("L", image.size, 0)

        manager = SimpleNamespace(
            _release_local_gpu_before_comfy=lambda: None,
            _get_clipseg_masker=lambda: ResidualDouble(),
        )
        for edge in (False, True):
            with self.subTest(touches_frame_edge=edge):
                zone = np.zeros((100, 100), dtype=bool)
                if edge:
                    zone[:30, :30] = True
                else:
                    zone[30:70, 30:70] = True
                support = zone.copy()
                if edge:
                    support[10:15, 10:15] = False
                else:
                    support[40:50, 40:50] = False
                allowed = zone & support
                with patch("requests.sessions.Session.request", side_effect=AssertionError("HTTP forbidden")):
                    result, _ = selected._vision_zone_pass(
                        manager, DiffusionDouble(), source, zone, support,
                        "Remove the measured object", "", 7,
                        selected.selection.load_settings()["edit"], None, None,
                        analysis=SimpleNamespace(regions=[]), target_query="object",
                    )
                self.assertIsNotNone(result)
                changed = np.any(np.asarray(result) != pixels, axis=-1)
                self.assertTrue(np.any(changed & allowed), "Fixture must include a real in-region change")
                self.assertFalse(np.any(changed & ~allowed), "Compositor escaped the measured write support")
                np.testing.assert_array_equal(np.asarray(source), pixels)


if __name__ == "__main__":
    unittest.main(verbosity=2)
