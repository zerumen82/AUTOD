"""CPU contracts: validate a described candidate before expensive finishing.

Semantic facts, masks and inference are declared doubles. The described route,
pixel validation and finish dispatch run for real; no backend/model is started.
"""
from __future__ import annotations

from contextlib import ExitStack
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


class DescribedFinishGate(unittest.TestCase):
    def execute(self, *, unchanged=False, quality_mode=True, auto_upscale=False,
                cancel_after_validation=False, reject_ghost=False):
        manager = ImgEditorManager()
        yy, xx = np.indices((64, 64))
        pixels = np.stack((40 + xx, 60 + yy, 90 + (xx + yy) % 30), axis=-1).astype(np.uint8)
        source = Image.fromarray(pixels)
        candidate = source.copy() if unchanged else Image.fromarray(255 - pixels)
        finished = candidate.resize((128, 128), Image.Resampling.NEAREST)
        subject = np.ones((64, 64), dtype=bool)
        zone = np.zeros_like(subject)
        zone[36:60, 20:44] = True
        client = SimpleNamespace(load=Mock(return_value=(True, "CPU inference double")))
        trace = []
        cancelled = False
        real_validate = manager._validate_visual_edit

        def validate(original, edited, prompt):
            nonlocal cancelled
            trace.append("validate")
            result = real_validate(original, edited, prompt)
            if cancel_after_validation:
                cancelled = True
            return result

        def finish(image, **kwargs):
            trace.append("finish")
            self.assertIs(image, candidate)
            self.assertEqual(kwargs["enhance_tier"], "4k")
            self.assertEqual(kwargs["quality_steps"], {"upscale": True})
            self.assertFalse(kwargs["use_generative"])
            return finished, "CPU finishing double", {"tested": True}

        with ExitStack() as stack:
            contexts = (
                patch.object(manager, "_prompt_authority_facts", return_value={
                    "body": True, "pose": False, "body_direction": "uncover"}),
                patch.object(manager, "_split_compound_instruction", return_value=[]),
                patch.object(manager, "_subject_queries", return_value=["the person"]),
                patch.object(manager, "measured_body_regime", return_value={
                    "body": True, "direction": "uncover", "magnitude": 1.0,
                    "intensity": 1.0, "enforce_clause": "", "negative_prompt": "", "engine": {}}),
                patch.object(manager, "measured_full_body_exposure", return_value=False),
                patch.object(manager, "_requested_removal_target", return_value="scarf"),
                patch.object(manager, "_compose_generation_prompt", return_value="Remove scarf"),
                patch.object(manager, "_compose_local_target_prompt", return_value="Remove scarf"),
                patch.object(manager, "_release_local_gpu_before_comfy"),
                patch.object(manager, "_validate_visual_edit", side_effect=validate),
                patch.object(ImgEditorManager, "_edit_leaves_a_ghost", return_value=reject_ghost),
                patch.object(selected, "_described_subject_mask", return_value=(subject, "person")),
                patch.object(selected, "_measured_clothing_zone", return_value=None),
                patch.object(selected, "_measured_target_zone", return_value=zone),
                patch.object(selected, "_vision_zone_pass", return_value=(candidate, "CPU generated")),
                patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
                patch("ui.tabs.comfy_launcher.start", side_effect=AssertionError("must not start backend")),
                patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client", return_value=client),
                patch("roop.img_editor.top_tier_finish.run_top_tier_quality", side_effect=finish),
            )
            for context in contexts:
                stack.enter_context(context)
            result = selected.edit_described_subject(
                manager, source, "remove the scarf", "remove the scarf", [],
                engine="longcat", automatic=True, quality_mode=quality_mode,
                auto_upscale=auto_upscale, enhance_tier="4k",
                quality_steps={"upscale": True}, cancel_check=lambda: cancelled,
            )
        return result, candidate, finished, trace, zone, manager

    def test_unchanged_candidate_skips_top_and_retains_rejected_pixels(self):
        result, candidate, _, trace, zone, manager = self.execute(unchanged=True)
        self.assertEqual(trace, ["validate"])
        self.assertIs(result[0], candidate)
        np.testing.assert_array_equal(np.asarray(result[2]) > 0, zone)
        self.assertNotIn("CPU finishing double", result[1])
        self.assertIsNot(manager._last_instruction_compliance.get("compliant"), True)

    def test_ghost_candidate_skips_top_without_replacing_it_with_source(self):
        result, candidate, _, trace, _, _ = self.execute(reject_ghost=True)
        self.assertEqual(trace, ["validate"])
        self.assertIs(result[0], candidate)

    def test_valid_candidate_keeps_finish_and_settings(self):
        result, _, finished, trace, _, manager = self.execute()
        self.assertEqual(trace, ["validate", "finish"])
        self.assertIs(result[0], finished)
        self.assertIn("CPU finishing double", result[1])
        self.assertEqual(manager._last_quality_analysis, {"tested": True})

    def test_auto_upscale_uses_same_gate(self):
        result, candidate, _, trace, _, _ = self.execute(
            unchanged=True, quality_mode=False, auto_upscale=True)
        self.assertEqual(trace, ["validate"])
        self.assertIs(result[0], candidate)

    def test_no_finish_requested_keeps_existing_dispatch(self):
        result, candidate, _, trace, _, _ = self.execute(quality_mode=False)
        self.assertEqual(trace, [])
        self.assertIs(result[0], candidate)

    def test_cancel_after_validation_prevents_finish(self):
        result, _, _, trace, _, _ = self.execute(cancel_after_validation=True)
        self.assertEqual(trace, ["validate"])
        self.assertIsNone(result[0])
        self.assertEqual(result[1], "Cancelado")


class RetryReached(RuntimeError):
    """CPU test signal: stop before the next route starts inference."""


class FinalDescribedAcceptance(unittest.TestCase):
    def execute(self, valid):
        candidates = []
        validations = []
        real_validate = ImgEditorManager._validate_visual_edit

        def described(_manager, image, *_args, **_kwargs):
            candidate = Image.fromarray(255 - np.asarray(image)) if valid else image.copy()
            candidates.append(candidate)
            return candidate, "CPU post-finish candidate", None

        def validate(original, candidate, prompt):
            validations.append(candidate)
            return real_validate(original, candidate, prompt)

        with (
            patch.object(selected, "edit_described_subject", side_effect=described),
            patch.object(ImgEditorManager, "_validate_visual_edit", side_effect=validate),
            patch.object(ImgEditorManager, "_edit_leaves_a_ghost", return_value=False),
            patch.object(ImgEditorManager, "_automatic_appearance_engine", side_effect=RetryReached("existing next route")),
        ):
            if valid:
                run = fixture._execute(engine="auto")
                self.assertIs(run.output, candidates[0])
                self.assertEqual(run.message, "CPU post-finish candidate")
            else:
                with self.assertRaisesRegex(RetryReached, "existing next route"):
                    fixture._execute(engine="auto")
        self.assertEqual(validations, candidates)
        self.assertEqual(len(validations), 1)

    def test_finished_invalid_candidate_is_rejected_by_real_caller(self):
        self.execute(valid=False)

    def test_valid_finished_candidate_is_accepted_without_new_generation(self):
        self.execute(valid=True)


if __name__ == "__main__":
    unittest.main()
