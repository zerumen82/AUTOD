#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe v5.1349 — selector «Tipo de edición» del Image Editor.

Comprueba, con el manager real y sus mediciones (sin GPU ni ComfyUI):
  1. «auto» no cambia hechos ni POV respecto al comportamiento anterior.
  2. Cada categoría eleva su eje con un prompt que no mide nada.
  3. Una medición fuerte del prompt gana al selector (se registra).
  4. «mostrando el culo a la cámara» con pose no activa la ruta de cámara.
  5. «camera» fuerza el POV explícito.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.img_editor.img_editor_manager import (  # noqa: E402
    _PINNED_ENGINE_BY_HINT, _SCOPE_HINT, _SCOPE_HINT_EFFECTS,
    get_img_editor_manager,
)


def facts_with(manager, hint, raw, translated=None):
    token = _SCOPE_HINT.set(hint)
    try:
        return manager._prompt_authority_facts(raw, translated or raw)
    finally:
        _SCOPE_HINT.reset(token)


def main():
    manager = get_img_editor_manager()
    ok = []

    def check(name, condition, detail=""):
        ok.append(bool(condition))
        print(f"{'PASS' if condition else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))

    plain = "hazlo bonito"
    auto_facts = facts_with(manager, "auto", plain)
    check(
        "auto: prompt neutro sin ejes",
        not any(auto_facts.get(k) for k in (
            "pose", "body", "scenario", "local_appearance", "local_region",
            "structural_add", "structural_remove",
        )),
        str({k: auto_facts.get(k) for k in ("pose", "body", "scenario")}),
    )

    clothing = facts_with(manager, "clothing", plain)
    check(
        "clothing: body=True sin inventar dirección (neutro → change)",
        clothing["body"] is True
        and clothing["body_direction"] == "change"
        and clothing["body_intensity"] >= 0.45,
        f"{clothing['body_direction']} {clothing['body_intensity']}",
    )

    background = facts_with(manager, "background", plain)
    check(
        "background: scenario=True",
        background["scenario"] is True and background["local_region"] is False,
        str(background["scenario"]),
    )

    poses = facts_with(manager, "pose", plain)
    check("pose: pose=True", poses["pose"] is True and poses["authority"] is True)

    features = facts_with(manager, "features", plain)
    check("features: local_appearance=True", features["local_appearance"] is True)

    added = facts_with(manager, "add", plain)
    check("add: structural_add=True", added["structural_add"] is True)

    removed = facts_with(manager, "remove", plain)
    check("remove: structural_remove=True", removed["structural_remove"] is True)

    people = facts_with(manager, "people", plain)
    check("people: sin add/remove medido → scenario", people["scenario"] is True)

    body_hint = facts_with(manager, "body", plain)
    check(
        "body: body=True dirección medida o change",
        body_hint["body"] is True
        and body_hint["body_direction"] in ("uncover", "cover", "change"),
        body_hint["body_direction"],
    )

    strong_body = "quita la ropa"
    measured = facts_with(manager, "auto", strong_body)
    conflicting = facts_with(manager, "clothing", strong_body)
    check(
        "clothing + «quita la ropa»: medición fuerte manda (sin cambio de eje)",
        conflicting.get("body") is True
        and conflicting.get("body_direction") == measured.get("body_direction"),
        f"{measured.get('body_intensity')} vs {conflicting.get('body_intensity')}",
    )

    pose_vs_camera = "cambiar de postura y mostrando el culo a la camara"
    check(
        "pose + «mostrando el culo a la cámara»: POV False",
        manager._explicit_viewpoint_request(pose_vs_camera, pose_vs_camera) is False,
    )
    check(
        "«cambia el pov» sigue siendo POV",
        manager._explicit_viewpoint_request("cambia el pov", "change the viewpoint") is True,
    )

    camera_plain = facts_with(manager, "camera", plain)
    check(
        "camera: no altera hechos (el POV se fuerza en la ruta)",
        not any(camera_plain.get(k) for k in (
            "pose", "body", "scenario", "local_appearance", "local_region",
        )),
    )

    check(
        "solo «camera» fija motor; el resto lo decide la ruta medida",
        _PINNED_ENGINE_BY_HINT == {"camera": "qwen_camera"},
        str(_PINNED_ENGINE_BY_HINT),
    )

    hint_axis = facts_with(manager, "pose", "quita la ropa")
    check(
        "selector pose cede ante cuerpo medido fuerte (no inventa postura)",
        hint_axis.get("pose") is False,
        str({k: hint_axis.get(k) for k in ("pose", "body")}),
    )

    weak_body = "ponle ropa"
    features_hint = facts_with(manager, "features", weak_body)
    check(
        "features manda sobre un cuerpo medido débil (no se borra el eje)",
        features_hint.get("local_appearance") is True
        and features_hint.get("body") is False,
        str({k: features_hint.get(k) for k in ("body", "local_appearance")}),
    )
    background_hint = facts_with(manager, "background", weak_body)
    check(
        "background manda sobre un cuerpo medido débil (no se borra el eje)",
        background_hint.get("scenario") is True
        and background_hint.get("body") is False,
        str({k: background_hint.get(k) for k in ("body", "scenario")}),
    )

    camera_prompt = "cambia el POV"
    camera_facts = facts_with(manager, "auto", camera_prompt)
    check(
        "señal fuerte: «cambia el POV» mide eje camera (no pose)",
        manager._strong_prompt_axis(camera_facts, camera_prompt, camera_prompt)
        == "camera",
    )
    posture_prompt = "CAMBIAR DE POSTURA A LA MUJER MOSTRANDO EL CULO A LA CAMARA"
    posture_facts = facts_with(manager, "auto", posture_prompt)
    check(
        "señal fuerte: postura real gana a cámara",
        manager._strong_prompt_axis(posture_facts, posture_prompt, posture_prompt)
        == "pose",
        str({k: posture_facts.get(k) for k in ("pose", "body")}),
    )

    cover_hint = facts_with(manager, "clothing", "con ropa puesta")
    check(
        "clothing no invierte una dirección medida (cover se conserva)",
        cover_hint["body_direction"] == "cover",
        cover_hint["body_direction"],
    )
    dress_hint = facts_with(manager, "clothing", "ponle ropa")
    check(
        "clothing no fuerza uncover cuando el prompt ya mide cuerpo",
        dress_hint["body_direction"] == "change",
        dress_hint["body_direction"],
    )

    # Cualquiera de las opciones del desplegable: cada una eleva solo su eje,
    # sin listas de palabras y sin casos especiales por frase.
    axis_keys = (
        "pose", "body", "scenario", "local_appearance", "local_region",
        "structural_add", "structural_remove",
    )
    for option, effect in _SCOPE_HINT_EFFECTS.items():
        got = facts_with(manager, option, "hazlo bonito")
        expected = set(k for k, value in effect["facts"].items() if value)
        if option == "people":
            expected.add("scenario")
        elevated = set(k for k in axis_keys if got.get(k))
        check(
            f"opción «{option}»: eleva solo {sorted(expected)}",
            elevated == expected,
            f"elevado={sorted(elevated)} esperado={sorted(expected)}",
        )

    # La ContextVar vuelve a "auto" tras cada medición.
    check("ContextVar restaurada a auto", _SCOPE_HINT.get() == "auto")

    total = len(ok)
    passed = sum(ok)
    print(f"\n{passed}/{total} comprobaciones")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
