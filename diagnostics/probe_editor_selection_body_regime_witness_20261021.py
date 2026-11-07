"""Witness: the new contract FAILS against the previous behaviour.

The previous behaviours are simulated here without touching the production
code:

  * the blind ignored the measured inclusion rule, so the head of the
    selected subject arrived greyed out;
  * the regional route used a FIXED recipe (EDIT + 'offset' anchor, no
    measured zone, no end state), so a body change was answered with a copy
    of the photo (measured anchor: registro 0,999, ropa intacta; log real
    2026-10-26 12:00: el mensaje decía «img2img puro» y el workflow iba con
    ReferenceLatent);
  * la apariencia sobre una cabeza seleccionada usaba el lienzo regional
    anclado, que copia la foto y no cambia nada (caso real del usuario,
    log 00:09).

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


def main():
    failures = []

    # 1. The old blind: every head that is not selected is greyed out.
    with patch.object(es, "_head_inside_selection", return_value=False):
        try:
            probe.selected_head_is_not_blinded()
            failures.append("selected_head_is_not_blinded did NOT fail with the old blind")
        except AssertionError as error:
            print(f"EXPECTED FAIL selected_head_is_not_blinded: {error}", flush=True)

    # 2. The old fixed recipe: no measured regime, so the client keeps the
    #    ReferenceLatent and the anchored EDIT copies the clothed photo.
    with patch.object(ImgEditorManager, "measured_body_regime",
                      return_value=dict(OLD_FIXED_REGIME)):
        try:
            probe.measured_body_change_writes_the_measured_zone()
            failures.append("measured_body_change_writes_the_measured_zone did NOT fail with the old regime")
        except AssertionError as error:
            print(f"EXPECTED FAIL measured_body_change_writes_the_measured_zone: {error}", flush=True)

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
