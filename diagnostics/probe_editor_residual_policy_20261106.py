"""CPU contracts for configurable residual limits and measured retry geometry.

Only the settings reader is substituted. Production residual arithmetic and
morphology run on synthetic non-person masks; no models or backend are used.
"""
from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from roop.img_editor import selected_region_edit as selected


class ResidualPolicyContracts(unittest.TestCase):
    def setUp(self):
        self.settings = deepcopy(selected.selection.load_settings())
        self.settings["edit"].update({
            "residual_failure_fraction": 0.4,
            "residual_retry_growth_fraction": 0.35,
            "residual_retry_min_radius_pixels": 8,
        })
        self.zone = np.zeros((80, 80), dtype=bool)
        self.zone[30:40, 30:40] = True
        self.support = np.zeros_like(self.zone)
        self.support[15:60, 15:60] = True
        self.reader = patch.object(selected.selection, "load_settings", return_value=self.settings)
        self.reader.start()
        self.addCleanup(self.reader.stop)

    def test_residual_uses_nondefault_configured_fraction(self):
        self.settings["edit"]["residual_failure_fraction"] = 0.7
        self.assertFalse(selected._residual_ratio_failed(50, self.zone))
        self.assertTrue(selected._residual_ratio_failed(70, self.zone))
        self.settings["edit"]["residual_failure_fraction"] = 0.2
        self.assertTrue(selected._residual_ratio_failed(20, self.zone))

    def test_explicit_residual_override_is_preserved(self):
        self.settings["edit"]["residual_failure_fraction"] = 0.7
        self.assertTrue(selected._residual_ratio_failed(50, self.zone, threshold=0.3))

    def test_growth_and_minimum_radius_use_configuration(self):
        self.settings["edit"].update({
            "residual_retry_growth_fraction": 0.1,
            "residual_retry_min_radius_pixels": 1,
        })
        small = selected._expand_zone_within_support(self.zone, self.support)
        self.settings["edit"]["residual_retry_growth_fraction"] = 0.5
        large = selected._expand_zone_within_support(self.zone, self.support)
        self.assertGreater(int(large.sum()), int(small.sum()))
        self.settings["edit"]["residual_retry_min_radius_pixels"] = 12
        larger = selected._expand_zone_within_support(self.zone, self.support)
        self.assertGreater(int(larger.sum()), int(large.sum()))
        self.assertFalse(np.any(larger & ~self.support))

    def test_zero_growth_and_radius_is_identity(self):
        self.settings["edit"].update({
            "residual_retry_growth_fraction": 0.0,
            "residual_retry_min_radius_pixels": 0,
        })
        result = selected._expand_zone_within_support(self.zone, self.support)
        np.testing.assert_array_equal(result, self.zone)

    def test_empty_and_incompatible_masks_do_not_expand(self):
        empty = np.zeros_like(self.zone)
        np.testing.assert_array_equal(
            selected._expand_zone_within_support(empty, self.support), empty,
        )
        np.testing.assert_array_equal(
            selected._expand_zone_within_support(self.zone, self.support[:10]), self.zone,
        )

    def test_unavailable_measurement_is_not_a_failure(self):
        self.assertFalse(selected._residual_ratio_failed(None, self.zone))
        self.assertFalse(selected._residual_ratio_failed(10, None))

    def test_invalid_fraction_is_not_silently_accepted(self):
        for key, operation in (
            ("residual_failure_fraction", lambda: selected._residual_ratio_failed(50, self.zone)),
            ("residual_retry_growth_fraction", lambda: selected._expand_zone_within_support(self.zone, self.support)),
        ):
            for value in (float("nan"), float("inf"), -0.1, 1.1, True, "0.4", None):
                with self.subTest(key=key, value=value):
                    previous = self.settings["edit"][key]
                    self.settings["edit"][key] = value
                    try:
                        with self.assertRaises(ValueError):
                            operation()
                    finally:
                        self.settings["edit"][key] = previous

    def test_invalid_radius_is_not_silently_accepted(self):
        for value in (-1, 1.5, True, "8", None):
            with self.subTest(value=value):
                self.settings["edit"]["residual_retry_min_radius_pixels"] = value
                with self.assertRaises(ValueError):
                    selected._expand_zone_within_support(self.zone, self.support)

    def test_missing_key_reports_configuration_error(self):
        del self.settings["edit"]["residual_failure_fraction"]
        with self.assertRaisesRegex(ValueError, "residual_failure_fraction"):
            selected._residual_ratio_failed(50, self.zone)


if __name__ == "__main__":
    unittest.main()
