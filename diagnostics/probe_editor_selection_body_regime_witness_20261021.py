"""Witness: the new contract FAILS against the previous behaviour.

The previous behaviours are simulated here without touching the production
code:

  * the blind ignored the measured inclusion rule, so the head of the
    selected subject arrived greyed out;
  * the regional route used a FIXED recipe (EDIT + 'offset' anchor, no
    measured negative, no end state), so a body change was answered with a
    copy of the photo;
  * the body change ran on Kontext alone: the negative is inert with CFG=1
    and the text alone leaves the garment on (measured skin share inside the
    garment zone: original 0.2675, Kontext 0.0676).

The probe of the regime must FAIL in all three cases. This is the
fail->pass evidence of the fix; it prints the failures and never edits
anything.
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roop.img_editor import element_selection as es  # noqa: E402
from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_editor_selection_body_regime_20261021 as probe  # noqa: E402


OLD_FIXED_REGIME = {
    "body": False,
    "direction": "change",
    "intensity": 0.0,
    "magnitude": 0.0,
    "state_suffix": "",
    "negative_prompt": "",
    "enforce_clause": "",
    # The old fixed recipe: the anchored EDIT, no measured zone.
    "engine": {"reference_latents_method": "offset"},
}


def _load_without_vision(self, *args, **kwargs):
    """The vision engine is not available: the route cannot execute."""
    self.loads.append((args, kwargs))
    if kwargs.get("allow_vision_engine"):
        return False, "vision engine not available"
    return True, "CPU inference double"


def main():
    failures = []

    # 1. The old blind: every head that is not selected is greyed out.
    with patch.object(es, "_head_inside_selection", return_value=False):
        try:
            probe.selected_head_is_not_blinded()
            failures.append("selected_head_is_not_blinded did NOT fail with the old blind")
        except AssertionError as error:
            print(f"EXPECTED FAIL selected_head_is_not_blinded: {error}", flush=True)

    # 2. The old fixed recipe: no measured regime at all.
    with patch.object(ImgEditorManager, "measured_body_regime",
                      return_value=dict(OLD_FIXED_REGIME)):
        try:
            probe.measured_body_change_uses_the_vision_engine()
            failures.append("measured_body_change_uses_the_vision_engine did NOT fail with the old regime")
        except AssertionError as error:
            print(f"EXPECTED FAIL measured_body_change_uses_the_vision_engine: {error}", flush=True)

    # 3. The body change without the engine that follows the text: the
    #    instruction is never executed.
    with patch.object(probe.RecordingInference, "load", _load_without_vision):
        try:
            probe.measured_body_change_uses_the_vision_engine()
            failures.append("measured_body_change_uses_the_vision_engine did NOT fail without the vision engine")
        except AssertionError as error:
            print(f"EXPECTED FAIL measured_body_change_uses_the_vision_engine (no vision engine): {error}", flush=True)

    # 3. The appearance change on a selected head with the OLD route: the
    #    regional canvas with the anchored recipe, which copies the photo and
    #    changes nothing (the user's real case, log 00:09).
    with patch.object(ImgEditorManager, "_prompt_authority_facts",
                      return_value={"body": False, "local_appearance": False,
                                    "local_region": True, "pose": False,
                                    "authority": False, "scenario": False,
                                    "structural_remove": False}):
        try:
            probe.appearance_change_on_a_head_uses_the_focal_recipe()
            failures.append("appearance_change_on_a_head_uses_the_focal_recipe did NOT fail with the old route")
        except AssertionError as error:
            print(f"EXPECTED FAIL appearance_change_on_a_head_uses_the_focal_recipe (old route): {error}", flush=True)

    if failures:
        for failure in failures:
            print(f"WITNESS FAIL: {failure}", flush=True)
        return 1
    print("WITNESS OK: the previous behaviours fail the new contract", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
