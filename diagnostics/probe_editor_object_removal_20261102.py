"""Functional CPU removal contracts; synthetic segmentation is not visual proof."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.probe_editor_versatility_integrity_20261105 import RemovalContracts  # noqa: E402


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RemovalContracts)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
