"""CPU regression for an unclassified local request and its constraints.

Production semantic facts, compound dispatch, proposal grounding, crop and paste
execute. LLM proposal names, segmentation, diffusion and pixel acceptance are
explicit doubles. No model, backend startup or downloads are permitted.
"""
from contextlib import ExitStack
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diagnostics.probe_editor_versatility_integrity_20261105 import (  # noqa: E402
    CPUContracts, RecordingDiffusion,
)


class UnknownScopeContracts(CPUContracts):
    def setUp(self):
        super().setUp()
        yy, xx = np.indices((160, 240))
        pixels = np.stack((40 + xx // 3, 70 + yy // 2, 100 + xx // 5), axis=-1).astype(np.uint8)
        self.source = Image.fromarray(pixels)
        self.target = Image.new("L", self.source.size, 0)
        self.target.paste(255, (100, 60, 140, 100))
        self.client = RecordingDiffusion()
        self.stack.enter_context(patch(
            "roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
            return_value=self.client,
        ))
        self.stack.enter_context(patch.object(self.manager, "_second_focused_pass", return_value=None))
        self.stack.enter_context(patch.object(self.manager, "_added_face_mask", return_value=None))
        self.stack.enter_context(patch.object(
            self.manager, "_validate_visual_edit", return_value=(True, "CPU pixel-acceptance double"),
        ))

    def edit(self, prompt, *, quality=False):
        return self.manager.generate_intelligent(
            self.source, prompt, engine="auto", use_semantic=False,
            use_rewriter=False, enhance_faces=False, quality_mode=quality,
            quality_preserve_faces=False,
        )

    def test_unclassified_compound_without_grounded_support_never_loads_or_generates(self):
        prompt = "Make the ceramic coffee cup cobalt blue. Keep everything else unchanged."
        with (
            patch.object(self.manager, "_compound_clause_zone", return_value=None),
            patch.object(self.manager, "_proposed_region_passes", return_value=[]),
        ):
            output, message, _ = self.edit(prompt)
        self.assertIsNotNone(output, message)
        self.assertEqual(self.client.loads, [], "Unknown scope must abstain before loading diffusion")
        self.assertEqual(self.client.generations, [], "Unknown scope must never become an unmasked global edit")
        np.testing.assert_array_equal(np.asarray(output), np.asarray(self.source))

    def test_partially_measured_compound_does_not_silently_drop_unknown_request(self):
        known = "Paint the cup blue"
        unknown = "Apply the requested effect to the spoon"
        prompt = f"{known}. {unknown}."

        def measured(_image, clause, *_args, **_kwargs):
            return self.target if clause == known else None

        with (
            patch.object(self.manager, "_compound_clause_zone", side_effect=measured),
            patch.object(self.manager, "_proposed_region_passes", side_effect=AssertionError(
                "A partial measured compound must not replace missing-clause coverage with a whole-prompt guess")),
        ):
            output, message, _ = self.edit(prompt)
        self.assertIsNotNone(output, message)
        self.assertEqual(self.client.loads, [], "Missing unknown clause cannot authorize a partial generation")
        self.assertEqual(self.client.generations, [])
        np.testing.assert_array_equal(np.asarray(output), np.asarray(self.source))

    def test_full_instruction_proposal_must_be_grounded_before_one_masked_generation(self):
        prompt = "Make the ceramic coffee cup cobalt blue. Keep everything else unchanged."
        queries = []

        def segment(_image, query):
            queries.append(query)
            return self.target if query == "cup" else None

        masker = SimpleNamespace(generate_mask=Mock(side_effect=segment))
        with ExitStack() as stack:
            stack.enter_context(patch.object(self.manager, "_compound_clause_zone", return_value=None))
            stack.enter_context(patch.object(self.manager, "_get_clipseg_masker", return_value=masker))
            planner = stack.enter_context(patch.object(self.manager, "_region_proposals", return_value=["cup"]))
            stack.enter_context(patch.object(self.manager, "_continue_measured_region", return_value=self.target))
            output, message, _ = self.edit(prompt)
        self.assertIsNotNone(output, message)
        planner.assert_called_once()
        self.assertEqual(planner.call_args.args[0], prompt, "Preservation constraints must reach the support planner")
        self.assertEqual(queries, ["cup"], "An LLM noun alone must not authorize writing without segmentation")
        self.assertEqual(len(self.client.generations), 1, message)
        generation = self.client.generations[0]
        self.assertIsNotNone(generation.get("mask_image"), "A measured local request must carry a sampler mask")
        self.assertIn("Make the ceramic coffee cup cobalt blue", generation["prompt"])
        self.assertIn("Keep everything else unchanged", generation["prompt"])
        self.assertNotIn("The face of the edited person may change", generation["prompt"],
                         "An object-only instruction cannot introduce an edited person")
        before, after = np.asarray(self.source), np.asarray(output)
        target = np.asarray(self.target) > 0
        self.assertTrue(np.any(before[target] != after[target]), "The measured target must actually change")
        self.assertFalse(np.any(before[~target] != after[~target]), "The unrequested scene must be preserved")

    def full_instruction_with_two_supports(self, *, second_measured):
        prompt = "Make the ceramic cup cobalt blue. Apply the requested effect to the spoon."
        second = Image.new("L", self.source.size, 0)
        second.paste(255, (160, 65, 200, 105))

        def segment(_image, query):
            if query == "cup":
                return self.target
            if query == "spoon" and second_measured:
                return second
            return None

        with (
            patch.object(self.manager, "_compound_clause_zone", return_value=None),
            patch.object(self.manager, "_get_clipseg_masker",
                         return_value=SimpleNamespace(generate_mask=Mock(side_effect=segment))),
            patch.object(self.manager, "_region_proposals", return_value=["cup", "spoon"]),
            patch.object(self.manager, "_continue_measured_region", side_effect=lambda measured, *_args: measured),
        ):
            output, message, _ = self.edit(prompt)
        return output, message, second

    def test_complete_proposal_cannot_hide_one_unmeasured_support(self):
        output, message, _ = self.full_instruction_with_two_supports(second_measured=False)
        self.assertIsNotNone(output, message)
        self.assertEqual(self.client.loads, [], "One grounded region cannot stand in for a missing companion")
        self.assertEqual(self.client.generations, [])
        np.testing.assert_array_equal(np.asarray(output), np.asarray(self.source))

    def test_complete_proposal_keeps_both_distinct_measured_changes(self):
        output, message, second = self.full_instruction_with_two_supports(second_measured=True)
        self.assertIsNotNone(output, message)
        self.assertEqual(len(self.client.generations), 1, message)
        generation = self.client.generations[0]
        self.assertIsNotNone(generation.get("mask_image"))
        self.assertIn("ceramic cup cobalt blue", generation["prompt"])
        self.assertIn("requested effect to the spoon", generation["prompt"])
        before, after = np.asarray(self.source), np.asarray(output)
        first_mask, second_mask = np.asarray(self.target) > 0, np.asarray(second) > 0
        self.assertTrue(np.any(before[first_mask] != after[first_mask]), "First requested support was dropped")
        self.assertTrue(np.any(before[second_mask] != after[second_mask]), "Second requested support was dropped")
        self.assertFalse(np.any(before[~(first_mask | second_mask)] != after[~(first_mask | second_mask)]))

    def test_single_unclassified_request_without_support_abstains_before_model_load(self):
        with patch.object(self.manager, "_proposed_region_passes", return_value=[]):
            output, message, _ = self.edit("Make the ceramic coffee cup cobalt blue")
        self.assertIsNotNone(output, message)
        self.assertEqual(self.client.loads, [])
        self.assertEqual(self.client.generations, [])
        np.testing.assert_array_equal(np.asarray(output), np.asarray(self.source))

    def test_single_unclassified_request_uses_one_grounded_masked_edit(self):
        prompt = "Make the ceramic coffee cup cobalt blue"
        with (
            patch.object(self.manager, "_get_clipseg_masker", return_value=SimpleNamespace(
                generate_mask=lambda _image, query: self.target if query == "cup" else None)),
            patch.object(self.manager, "_region_proposals", return_value=["cup"]) as planner,
            patch.object(self.manager, "_continue_measured_region", return_value=self.target),
        ):
            output, message, _ = self.edit(prompt)
        self.assertIsNotNone(output, message)
        planner.assert_called_once()
        self.assertEqual(planner.call_args.args[0], prompt)
        self.assertEqual(len(self.client.generations), 1, message)
        generation = self.client.generations[0]
        self.assertIsNotNone(generation.get("mask_image"))
        self.assertTrue(generation.get("preserve_frame"))
        self.assertTrue(generation.get("prompt_authority"))
        changed = np.any(np.asarray(output) != np.asarray(self.source), axis=-1)
        target = np.asarray(self.target) > 0
        self.assertTrue(np.any(changed & target))
        self.assertFalse(np.any(changed & ~target))

    def test_quality_preserves_real_semantic_region_dispatch(self):
        prompt = "Paint the cup blue"
        self.assertTrue(self.manager._prompt_authority_facts(prompt, prompt)["surface_recolor"],
                        "Fixture must exercise a real measured recolor fact")
        generations = []
        quality_target = Image.new("L", self.source.size, 0)
        quality_target.paste(255, (80, 50, 160, 110))
        measured_pass = {"mask": quality_target, "text": prompt, "query": "cup",
                         "coverage": 4800 / (240 * 160), "measured_request": True}

        def identity_finish(image, *_args, **_kwargs):
            return image, "CPU finishing identity double", {}

        for quality in (False, True):
            with (
                self.subTest(quality=quality),
                patch.object(self.manager, "_measured_region_passes", return_value=[measured_pass]) as measure,
                patch.object(self.manager, "_complete_request_surface", return_value=quality_target),
                patch.object(self.manager, "_complete_surface_recolor", side_effect=lambda _original, edited, *_args: edited),
                patch("roop.img_editor.top_tier_finish.run_quality_enhance_after_edit", side_effect=identity_finish),
                patch("roop.img_editor.top_tier_finish.run_top_tier_quality", side_effect=identity_finish),
            ):
                before_calls = len(self.client.generations)
                output, message, _ = self.edit(prompt, quality=quality)
                self.assertIsNotNone(output, message)
                self.assertTrue(measure.call_args_list, "Quality mode bypassed the measured-region route")
                self.assertTrue(measure.call_args.kwargs["surface_recolor"])
                self.assertEqual(len(self.client.generations) - before_calls, 1, message)
                generation = self.client.generations[-1]
                self.assertIsNotNone(generation.get("mask_image"))
                generations.append(generation)
                changed = np.any(np.asarray(output) != np.asarray(self.source), axis=-1)
                self.assertTrue(np.any(changed & (np.asarray(quality_target) > 0)))
                self.assertFalse(np.any(changed & (np.asarray(quality_target) == 0)))
        self.assertEqual(generations[0]["image"].size, generations[1]["image"].size)
        np.testing.assert_array_equal(
            np.asarray(generations[0]["mask_image"]), np.asarray(generations[1]["mask_image"]),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
