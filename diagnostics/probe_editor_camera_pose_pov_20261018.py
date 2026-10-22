"""Camera pose mapping: a POV request maps to the viewer-facing trained pose.

The angle LoRA has no first-person pose among its 96 trained ones; the
closest trained approximation of a first-person POV request is 'front view
eye-level shot close-up'. Pure function contract: no models, GPU, network
or UI.
"""
from roop.img_editor.qwen_camera_comfy_client import camera_pose

POV_EXPECTED = "<sks> front view eye-level shot medium shot"


def main():
    checks = []

    pov_cases = [
        ("CAMBIA EL POV, CORNUDO POV", "change the pov, cuckold pov"),
        ("cambia el pov", "change the pov"),
        # «primera» ≈ «primer» (fuzzy del matcher) mide close-up en
        # distancia; documentado como comportamiento medido y estable.
        ("PRIMERA PERSONA", "first person view"),
    ]
    for raw, translated in pov_cases:
        got = camera_pose(raw, translated)
        expected = ("<sks> front view eye-level shot close-up"
                    if raw == "PRIMERA PERSONA" else POV_EXPECTED)
        checks.append((f"pov {raw[:28]!r} -> viewer-facing pose",
                       got == expected, got))
    got = camera_pose("plano sobre el hombro", "over the shoulder shot")
    checks.append(("pov 'plano sobre el hombro' -> viewer-facing pose",
                   got == POV_EXPECTED, got))

    regression = [
        (("cambia la vista", "change the view"),
         "<sks> front-right quarter view elevated shot medium shot"),
        (("vista cenital", "top down view"),
         "<sks> front view high-angle shot medium shot"),
        (("vista de perfil desde la izquierda", "left side profile view"),
         "<sks> left side view eye-level shot medium shot"),
        (("desde atras", "from behind"),
         "<sks> back view eye-level shot medium shot"),
        (("pov elevado", "elevated pov"),
         "<sks> front view elevated shot medium shot"),
        (("pov plano general", "wide pov shot"),
         "<sks> front view eye-level shot wide shot"),
    ]
    for (raw, translated), expected in regression:
        got = camera_pose(raw, translated)
        checks.append((f"no-regresion {raw[:28]!r}",
                       got == expected, got))

    for name, ok, got in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name} | got={got!r}")
    failed = [c for c in checks if not c[1]]
    assert not failed, f"{len(failed)} caso(s) de pose POV incorrecto(s)"
    print(f"OK: {len(checks)} comprobaciones de camera_pose POV")


if __name__ == "__main__":
    main()
