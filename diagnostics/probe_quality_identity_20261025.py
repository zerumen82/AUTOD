"""CPU contracts for preserving EVERY face across quality restoration.

Embeddings substitute only detector output. The production acceptance function
and assignment algorithm execute normally; this does not certify visual quality.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roop.img_editor import top_tier_finish as finish  # noqa: E402


class QualityIdentityContracts(unittest.TestCase):
    def setUp(self):
        self.original = Image.new("RGB", (16, 16), "red")
        self.before = Image.new("RGB", (16, 16), "green")
        self.after = Image.new("RGB", (16, 16), "blue")
        self.settings = {"identity": {
            "verify_after_face_restore": True, "max_side": 2048, "max_drop": 0.02,
        }}
        self.ref = np.eye(3, dtype=np.float32)[:2]

    def run_gate(self, ref, pre, post):
        with patch.object(finish, "_face_embeddings", side_effect=[ref, pre, post]):
            return finish._faces_keep_identity(
                self.original, self.before, self.after, self.settings,
            )

    def assert_reverted(self, ref, pre, post):
        result, reason = self.run_gate(ref, pre, post)
        self.assertIs(result, self.before)
        self.assertTrue(reason)

    def test_second_face_degrades_while_first_remains_perfect(self):
        post = np.array([[1, 0, 0], [0, 0.6, 0.8]], dtype=np.float32)
        self.assert_reverted(self.ref, self.ref, post)

    def test_face_order_does_not_change_acceptance(self):
        result, reason = self.run_gate(self.ref, self.ref[::-1], self.ref)
        self.assertIs(result, self.after)
        self.assertEqual(reason, "")

    def test_missing_one_face_reverts(self):
        self.assert_reverted(self.ref, self.ref, self.ref[:1])

    def test_duplicate_of_good_face_cannot_cover_lost_identity(self):
        self.assert_reverted(self.ref, self.ref, np.repeat(self.ref[:1], 2, axis=0))

    def test_new_face_after_restoration_reverts(self):
        self.assert_reverted(self.ref, self.ref, np.eye(3, dtype=np.float32))

    def test_single_face_degradation_reverts(self):
        post = np.array([[0.6, 0.8, 0]], dtype=np.float32)
        self.assert_reverted(self.ref[:1], self.ref[:1], post)

    def test_all_faces_improve(self):
        pre = np.array([[0.8, 0, 0.6], [0, 0.8, 0.6]], dtype=np.float32)
        result, reason = self.run_gate(self.ref, pre, self.ref)
        self.assertIs(result, self.after)
        self.assertEqual(reason, "")

    def test_per_face_tolerance_is_configured(self):
        post = np.array([[1, 0, 0], [0, 0.98, np.sqrt(1 - 0.98**2)]])
        self.settings["identity"]["max_drop"] = 0.03
        result, reason = self.run_gate(self.ref, self.ref, post)
        self.assertIs(result, self.after)
        self.assertEqual(reason, "")
        self.settings["identity"]["max_drop"] = 0.01
        self.assert_reverted(self.ref, self.ref, post)

    def test_normalizes_embeddings_instead_of_accepting_large_norm(self):
        post = np.array([[1, 0, 0], [0, 6, 8]], dtype=np.float32)
        self.assert_reverted(self.ref, self.ref, post)

    def test_invalid_or_missing_evidence_never_approves_restoration(self):
        for bad in [None, np.empty((0, 3)), np.zeros((2, 3)),
                    np.full((2, 3), np.nan), np.full((2, 3), np.inf),
                    np.ones((2, 2)), np.ones(3)]:
            for stage in range(3):
                evidence = [self.ref, self.ref, self.ref]
                evidence[stage] = bad
                with self.subTest(stage=stage, evidence=str(bad)):
                    self.assert_reverted(*evidence)

    def test_invalid_tolerance_cannot_disable_identity_gate(self):
        for tolerance in [float("nan"), float("inf"), -1, "invalid"]:
            with self.subTest(tolerance=tolerance):
                self.settings["identity"]["max_drop"] = tolerance
                self.assert_reverted(self.ref, self.ref, self.ref)

    def test_assignment_does_not_greedily_steal_another_face(self):
        # A greedy first-row match chooses candidate0 and strands reference1.
        # Optimal one-to-one matching preserves the only useful second match.
        ref = np.eye(2, dtype=np.float32)
        candidates = np.array([[0.8, 0.6], [0.7, -np.sqrt(1 - 0.7**2)]])
        result, reason = self.run_gate(ref, candidates, candidates[::-1])
        self.assertIs(result, self.after)
        self.assertEqual(reason, "")

    def test_disabled_check_preserves_explicit_configuration(self):
        self.settings["identity"]["verify_after_face_restore"] = False
        with patch.object(finish, "_face_embeddings") as detector:
            result, reason = finish._faces_keep_identity(
                self.original, self.before, self.after, self.settings,
            )
        self.assertIs(result, self.after)
        self.assertEqual(reason, "")
        detector.assert_not_called()

    def test_max_side_preserves_existing_budget(self):
        self.settings["identity"]["max_side"] = 8
        with patch.object(finish, "_face_embeddings") as detector:
            result, reason = finish._faces_keep_identity(
                self.original, self.before, self.after, self.settings,
            )
        self.assertIs(result, self.after)
        self.assertEqual(reason, "")
        detector.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
