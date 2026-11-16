"""Print the measured concept scores of the P1/P2 prompts (no generation)."""

import json
import sys

from roop.img_editor.nlp.semantic_analyzer import LightLocalIntentAnalyzer


PROMPTS = (
    "Raise the astronaut's right arm",
    "Change the astronaut suit to blue",
    "Make the astronaut taller",
    "Change the astronaut's hair to black",
    "Change the background to a forest",
    "Show the astronaut from a high camera angle",
    "Add a small flag beside the astronaut",
    "Remove the black helmet from the photo",
    "Add another astronaut beside the first one",
    "The woman must go barefoot and without stockings",
    "Quita el casco negro de la foto",
    "Pon una bandera pequeña al lado del astronauta",
)

CONCEPTS = (
    "structural_add", "structural_remove", "structural_background",
    "attribute_body", "attribute_uncover", "attribute_appearance",
    "attribute_recolor", "pose", "camera_view",
)


def main():
    analyzer = LightLocalIntentAnalyzer()
    for prompt in PROMPTS:
        scores = analyzer.score_concepts(prompt)
        picked = {name: round(float(scores.get(name, 0.0)), 4) for name in CONCEPTS}
        bias = analyzer.get_structural_bias(prompt)
        print(json.dumps({
            "prompt": prompt,
            "scores": picked,
            "structural_bias": bias,
            "magnitude": round(analyzer.get_magnitude(prompt), 3),
            "body_intensity": round(analyzer.get_body_transform_intensity(prompt), 3),
            "body_direction": analyzer.get_body_direction(prompt),
        }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.exit(main())
