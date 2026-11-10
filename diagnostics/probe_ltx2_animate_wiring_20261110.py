"""Probe CPU del cableado de LTX-2 en Animate (v5.1328).

No renderiza: comprueba que el motor nuevo llega al desplegable con los
parámetros medidos (8 pasos, CFG 1,0, 25 fps, fotogramas 8n+1), que el prompt
se compone como pide LTX-2 (un párrafo con acción, escena y sonidos) y que la
detección de modelos no deja pasar una ruta incompleta.

Uso: venv/Scripts/python.exe diagnostics/probe_ltx2_animate_wiring_20261110.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from roop.animate import ltx2_audio as ltx  # noqa: E402
from roop.animate.animate_manager import AnimateManager  # noqa: E402

failures = []


def check(name, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not condition:
        failures.append(name)


def main():
    action = "She looks into the camera and smiles, then says: \"Hola, ya estoy aquí.\""
    prompt = ltx.compose_prompt(action, "viento suave y pájaros lejanos")
    check("un_solo_parrafo", "\n" not in prompt, repr(prompt[:40]))
    check("audio_en_el_parrafo", "Audio: viento suave y pájaros lejanos." in prompt)
    check("dialogo_conservado", "Hola, ya estoy aquí." in prompt)
    check("limite_200_palabras", len(prompt.split()) <= 200, f"{len(prompt.split())} palabras")

    long_prompt = ltx.compose_prompt(" ".join(["word"] * 300))
    check("recorte_200", len(long_prompt.split()) <= 200 and long_prompt.endswith("."),
          f"{len(long_prompt.split())} palabras")

    spec_8gb = ltx.resolve_generation(8, 48)
    check("8gb_540p_y_49_frames",
          (spec_8gb["width"], spec_8gb["height"], spec_8gb["frames"]) == (960, 544, 49),
          str(spec_8gb))
    check("destilado_8_pasos_cfg1",
          spec_8gb["steps"] == 8 and spec_8gb["cfg"] == 1.0 and spec_8gb["fps"] == 25)
    spec_16gb = ltx.resolve_generation(16, 97)
    check("16gb_720p", (spec_16gb["width"], spec_16gb["height"]) == (1280, 704), str(spec_16gb))

    manager = AnimateManager()
    params = manager.resolve_params("ltx2_audio", frames=48, steps=None, cfg=None, magnitude=0.5)
    check("manager_ltx2_params",
          params["frames"] == 49 and params["steps"] == 8 and params["cfg"] == 1.0
          and params["fps"] == 25,
          str({k: params[k] for k in ("frames", "steps", "cfg", "fps")}))
    capped = manager.resolve_params("wan_video", frames=81, steps=28, cfg=6.0, magnitude=0.5)
    check("wan_sigue_capado", capped["frames"] == 49 and capped["steps"] == 20,
          f"{capped['frames']}f {capped['steps']}s")

    check("modelos_detectados", manager._check_models("ltx2_audio") == [],
          str(manager._check_models("ltx2_audio")))

    ui_source = open(os.path.join(ROOT, "ui", "tabs", "animate", "ui.py"), encoding="utf-8").read()
    check("ltx2_en_desplegable", '"ltx2_audio"' in ui_source)
    check("slider_hasta_97", "maximum=97" in ui_source)

    logic_source = open(os.path.join(ROOT, "ui", "tabs", "animate", "logic.py"), encoding="utf-8").read()
    check("logic_permite_ltx2", '201 if engine == "ltx2_audio"' in logic_source)

    print(f"\nRESULTADO: {'PASS' if not failures else 'FAIL'} ({len(failures)} fallos)")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
