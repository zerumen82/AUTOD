"""CPU contracts for occlusion coverage in its measured face domain.

Executes the production denominator extracted by AST, without loading models.
--source can point to the exact pre-fix snapshot to reproduce the regression.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument("--source", type=Path, default=ROOT / "roop/ProcessMgr.py")
ARGS, UNITTEST_ARGS = PARSER.parse_known_args()
TREE = ast.parse(ARGS.source.read_text(encoding="utf-8-sig"))
MANAGER = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "ProcessMgr")
SWAP = next(n for n in MANAGER.body if isinstance(n, ast.FunctionDef) and n.name == "_process_face_swap_v21")
SETTINGS = json.loads((ROOT / "config/model_configs.json").read_text(encoding="utf-8-sig"))["faceswap"]


def production_fraction(mask, interior):
    namespace = {
        "np": np, "cv2": cv2, "occ_support": SETTINGS["occlusion_support_threshold"],
        "_occ_interior": interior,
    }
    helper = next((n for n in ast.walk(SWAP) if isinstance(n, ast.FunctionDef)
                   and n.name == "_occlusion_fraction"), None)
    if helper is None:
        # Baseline: execute its actual final fraction assignment, not a replica.
        assignment = next(n for n in ast.walk(SWAP) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "occ_frac" for t in n.targets))
        namespace["occ_mask_aligned"] = mask
        exec(compile(ast.Module(body=[copy.deepcopy(assignment)], type_ignores=[]),
                     str(ARGS.source), "exec"), namespace)
        return namespace["occ_frac"]
    exec(compile(ast.Module(body=[copy.deepcopy(helper)], type_ignores=[]),
                 str(ARGS.source), "exec"), namespace)
    return namespace[helper.name](mask)


class OcclusionDomainContracts(unittest.TestCase):
    def setUp(self):
        self.interior = np.zeros((20, 20), dtype=np.float32)
        self.interior[5:15, 5:15] = 1
        self.wide = np.zeros_like(self.interior)
        self.wide[5:11, 5:15] = 1
        self.local = np.zeros_like(self.interior)
        self.local[5:7, 5:15] = 1

    def test_broad_mask_is_not_admitted_by_padding(self):
        fraction = production_fraction(self.wide, self.interior)
        self.assertAlmostEqual(fraction, 0.6)
        self.assertGreater(fraction, SETTINGS["occlusion_max_fraction"])

    def test_local_measured_mask_remains_eligible(self):
        fraction = production_fraction(self.local, self.interior)
        self.assertAlmostEqual(fraction, 0.2)
        self.assertLessEqual(SETTINGS["occlusion_min_fraction"], fraction)
        self.assertLessEqual(fraction, SETTINGS["occlusion_max_fraction"])

    def test_padding_does_not_change_fraction(self):
        for padding in [1, 13, 40]:
            with self.subTest(padding=padding):
                fraction = production_fraction(np.pad(self.local, padding), np.pad(self.interior, padding))
                self.assertAlmostEqual(fraction, 0.2)

    def test_uniform_resize_preserves_fraction(self):
        for scale in [2, 3, 4]:
            with self.subTest(scale=scale):
                mask = np.repeat(np.repeat(self.wide, scale, axis=0), scale, axis=1)
                domain = np.repeat(np.repeat(self.interior, scale, axis=0), scale, axis=1)
                self.assertAlmostEqual(production_fraction(mask, domain), 0.6)

    def test_domain_is_resized_to_final_mask_resolution(self):
        mask = cv2.resize(self.local, (40, 40), interpolation=cv2.INTER_NEAREST)
        self.assertAlmostEqual(production_fraction(mask, self.interior), 0.2)

    def test_without_interior_keeps_original_crop_behavior(self):
        self.assertAlmostEqual(production_fraction(self.local, None), float(np.mean(self.local > 0.1)))

    def test_ignores_support_outside_measured_interior(self):
        mask = self.local.copy()
        mask[:5] = 1
        self.assertAlmostEqual(production_fraction(mask, self.interior), 0.2)

    def test_uses_existing_configured_support_threshold(self):
        mask = self.local * (SETTINGS["occlusion_support_threshold"] / 2)
        self.assertEqual(production_fraction(mask, self.interior), 0)

    def test_absent_or_empty_mask_has_no_occlusion(self):
        for mask in [None, np.empty((0, 0)), np.zeros((1, 1))]:
            with self.subTest(mask=str(mask)):
                self.assertEqual(production_fraction(mask, self.interior), 0)

    def test_empty_or_invalid_domain_cannot_pass_gate(self):
        for domain in [np.zeros_like(self.interior), np.empty((0, 0)),
                       np.full_like(self.interior, np.nan), np.full_like(self.interior, np.inf),
                       -self.interior, np.ones(20), np.ones((20, 20, 2))]:
            with self.subTest(domain=str(domain)):
                self.assertGreater(production_fraction(self.local, domain), SETTINGS["occlusion_max_fraction"])

    def test_invalid_mask_cannot_pass_gate(self):
        for mask in [np.full_like(self.local, np.nan), np.full_like(self.local, np.inf),
                     np.ones(20), np.ones((20, 20, 2))]:
            with self.subTest(mask=str(mask)):
                self.assertGreater(production_fraction(mask, self.interior), SETTINGS["occlusion_max_fraction"])

    def test_rescue_and_final_gate_use_same_domain_function(self):
        assignments = [n for n in ast.walk(SWAP) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "occ_frac" for t in n.targets)]
        self.assertEqual(len(assignments), 1)
        self.assertEqual(ast.unparse(assignments[0].value), "_occlusion_fraction(occ_mask_aligned)")
        rescues = [n for n in ast.walk(SWAP) if isinstance(n, ast.If)
                   and "self._localize_mouth_occlusion(" in "\n".join(ast.unparse(s) for s in n.body)
                   and "_occlusion_fraction(occ_mask_aligned)" in ast.unparse(n.test)]
        self.assertTrue(rescues, "oral rescue must use the same measured denominator")


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *UNITTEST_ARGS], verbosity=2)
