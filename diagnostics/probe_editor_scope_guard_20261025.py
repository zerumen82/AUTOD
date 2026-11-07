"""CPU regressions for unintended instruction expansion; no image inference."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402
from roop.img_editor import element_selection  # noqa: E402


class ScopeGuardContracts(unittest.TestCase):
    def setUp(self):
        self.manager = ImgEditorManager()
        self.scorer = Mock()
        self.scorer.score_concepts.return_value = {
            "body_exposure_full": 0.0, "body_exposure_local": 0.0,
        }
        self.analyzer = patch.object(
            self.manager, "_get_semantic_analyzer", return_value=self.scorer,
        )
        self.analyzer.start()
        self.addCleanup(self.analyzer.stop)

    def test_recolor_compounds_never_imply_full_exposure(self):
        for text in (
            "The walls must be gray, the sofa must be blue",
            "The walls must be gray; the sofa must be blue",
            "The walls must be gray, the sofa must be blue, the lamp must be red",
        ):
            with self.subTest(text=text):
                self.assertFalse(self.manager.measured_full_body_exposure(text))

    def test_unconfirmed_state_is_not_added_to_first_subject(self):
        self.scorer.score_concepts.return_value = {
            "body_exposure_full": 0.0, "body_exposure_local": 0.8,
        }
        result = self.manager._body_state_clause(
            "The sofa must be gray, the woman must go barefoot", "uncover", 0.8,
        )
        self.assertEqual(result, "")

    def test_analyzer_failure_does_not_raise_or_invent_state(self):
        with patch.object(self.manager, "_get_semantic_analyzer",
                          side_effect=RuntimeError("analyzer unavailable")):
            self.assertEqual(self.manager._body_state_clause(
                "The walls must be gray, the sofa must be blue", "uncover", 0.8,
            ), "")
            self.assertFalse(self.manager.measured_full_body_exposure("a, b"))

    def test_scoring_failure_does_not_invent_state(self):
        self.scorer.score_concepts.side_effect = RuntimeError("measurement failed")
        text = "The walls must be gray, the sofa must be blue"
        self.assertFalse(self.manager.measured_full_body_exposure(text))
        self.assertEqual(self.manager._body_state_clause(text, "uncover", 0.8), "")

    def test_empty_instruction_cannot_expand_scope(self):
        self.assertFalse(self.manager.measured_full_body_exposure(""))
        self.assertEqual(self.manager._body_state_clause("", "uncover", 0.8), "")

    def test_scope_threshold_comes_from_configuration(self):
        self.scorer.score_concepts.return_value = {
            "body_exposure_full": 0.4, "body_exposure_local": 0.0,
        }
        settings = element_selection.load_settings()
        settings["instruction_scope"] = {
            "full_exposure_min_score": 0.9,
            "full_exposure_local_ratio": 1.15,
            "body_state_min_intensity": 0.3,
        }
        with patch.object(element_selection, "load_settings", return_value=settings):
            self.assertFalse(self.manager.measured_full_body_exposure("change this"))
            self.assertEqual(self.manager._body_state_clause("change this", "uncover", 0.8), "")

    def test_existing_positive_measurement_remains_available(self):
        self.scorer.score_concepts.return_value = {
            "body_exposure_full": 0.8, "body_exposure_local": 0.1,
        }
        self.assertTrue(self.manager.measured_full_body_exposure("measured instruction"))

    def test_real_cpu_recolor_measurement_does_not_expand(self):
        self.analyzer.stop()
        text = "The walls must be gray, the sofa must be blue"
        self.assertFalse(self.manager.measured_full_body_exposure(text))
        self.assertEqual(self.manager._body_state_clause(text, "uncover", 0.8), "")

    def test_invalid_scope_configuration_cannot_expand_scope(self):
        self.scorer.score_concepts.return_value = {
            "body_exposure_full": 1.0, "body_exposure_local": 0.0,
        }
        for invalid in (float("nan"), -1.0, True, "invalid"):
            with self.subTest(value=invalid):
                settings = element_selection.load_settings()
                settings["instruction_scope"] = {
                    "full_exposure_min_score": invalid,
                    "full_exposure_local_ratio": 1.15,
                    "body_state_min_intensity": 0.3,
                }
                with patch.object(element_selection, "load_settings", return_value=settings):
                    self.assertFalse(self.manager.measured_full_body_exposure("change this"))
                    self.assertEqual(self.manager._body_state_clause("change this", "uncover", 0.8), "")

    def test_nonfinite_measurement_cannot_expand_scope(self):
        for invalid in (float("inf"), float("nan")):
            with self.subTest(value=invalid):
                self.scorer.score_concepts.return_value = {
                    "body_exposure_full": invalid, "body_exposure_local": 0.0,
                }
                self.assertFalse(self.manager.measured_full_body_exposure("change this"))
                self.assertEqual(self.manager._body_state_clause("change this", "uncover", 0.8), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
