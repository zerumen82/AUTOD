"""Measure the authority facts and measured zones of the P1/P2 prompts.

Read-only diagnostic: no generation, no ComfyUI. It prints the facts that
decide the route so the P1 failures can be traced to a measurement instead of
guessed from the picture.
"""

import json
from pathlib import Path

from PIL import Image

from roop.img_editor.img_editor_manager import get_img_editor_manager


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "diagnostics" / "out" / "editor_camera_neutral_20261110" / "before.png"
CASES = (
    ("pose", "Raise the astronaut's right arm"),
    ("clothing", "Change the astronaut suit to blue"),
    ("body", "Make the astronaut taller"),
    ("features", "Change the astronaut's hair to black"),
    ("background", "Change the background to a forest"),
    ("camera", "Show the astronaut from a high camera angle"),
    ("add", "Add a small flag beside the astronaut"),
    ("remove", "Remove the black helmet from the photo"),
    ("people", "Add another astronaut beside the first one"),
    ("barefoot", "The woman must go barefoot and without stockings"),
)


def main():
    source = Image.open(SOURCE).convert("RGB")
    manager = get_img_editor_manager()
    report = []
    for category, prompt in CASES:
        row = {"category": category, "prompt": prompt}
        try:
            facts = manager._prompt_authority_facts(prompt, prompt)
            row["facts"] = {
                key: value for key, value in facts.items()
                if key in (
                    "authority", "pose", "body", "body_direction",
                    "body_intensity", "local_appearance", "local_region",
                    "surface_recolor", "region_ambiguous", "structural_add",
                    "structural_remove", "scenario", "scenario_strength",
                )
            }
            row["visual_subject_descriptions"] = manager._visual_subject_descriptions(prompt)
            row["subject_queries"] = manager._subject_queries(prompt, prompt)
            row["visual_request_queries"] = manager._visual_request_queries([prompt])
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        report.append(row)
        print(json.dumps(row, ensure_ascii=False, default=str), flush=True)
    (ROOT / "diagnostics" / "out" / "editor_categories_20261114" /
     "facts_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
