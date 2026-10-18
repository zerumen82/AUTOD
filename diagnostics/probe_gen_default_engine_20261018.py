# -*- coding: utf-8 -*-
"""probe_gen_default_engine_20261018.py — Generate arranca con el modelo seguro (Q4).

Decisión del usuario 2026-10-18: default NO-NSFW por defecto; los modelos
NSFW (abliterated, Chroma, SDXL explícitos) solo en el desplegable.
El default era flux_dev_abliterated: obedece adultos pero manchó el castillo
benigno aun con NSFW OFF (H-GEN-002). Q4 clava benignos en 4 semillas.

Contrato:
  [1] generate_ui.default_engine = flux_dev_q4 — FLUX.1-dev seguro.
  [2] El default resuelve en un archivo INSTALADO en disco.
  [3] Su conf en model_configs.json: steps=20, cfg=3.5 (guía vía
      FluxGuidance; el muestreo queda a CFG=1) y explicit=false (NO adulto).
  [4] El default aparece en el desplegable de la UI (instalados + fallback APK).
  [5] Los alias NSFW siguen listados como opciones (abliterated, chroma).
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from roop.img_editor.flux_gen_comfy_client import (  # noqa: E402
    FluxGenComfyClient,
    get_installed_generation_engines,
)
from roop.img_editor.gen_prompt_config import get_generate_ui_config  # noqa: E402

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), ".."))
Q4_FILE = "flux1-dev-Q4_K.gguf"


def main():
    checks = []

    cfg = get_generate_ui_config()
    default_engine = str(cfg.get("default_engine", "")).strip()
    checks.append(("[1] default_engine = flux_dev_q4 (seguro, no adulto)",
                   default_engine == "flux_dev_q4", default_engine))

    gguf_path = os.path.join(PROJECT_ROOT, "ui", "tob", "ComfyUI", "models",
                             "diffusion_models", Q4_FILE)
    checks.append((f"[2] archivo instalado {Q4_FILE}",
                   os.path.isfile(gguf_path), gguf_path))

    with open(os.path.join(PROJECT_ROOT, "config", "model_configs.json"),
              "r", encoding="utf-8") as fh:
        model_cfg = json.load(fh).get("flux_dev_q4", {})
    checks.append(("[3a] conf steps=20", model_cfg.get("steps") == 20,
                   model_cfg.get("steps")))
    checks.append(("[3b] conf cfg=3.5", model_cfg.get("cfg") == 3.5,
                   model_cfg.get("cfg")))
    checks.append(("[3c] conf explicit=false (NO adulto por defecto)",
                   bool(model_cfg.get("explicit")) is False,
                   model_cfg.get("explicit")))

    loaded, msg = FluxGenComfyClient().load(default_engine)
    checks.append(("[2b] load() del default acepta el alias", bool(loaded), msg))

    installed = {value for _label, value in get_installed_generation_engines()}
    checks.append(("[4] el default está en el desplegable de la UI",
                   default_engine in installed, sorted(installed)))
    checks.append(("[5a] abliterated sigue como opción NSFW",
                   "flux_dev_abliterated" in installed, sorted(installed)))
    checks.append(("[5b] chroma sigue como opción NSFW",
                   "chroma" in installed, sorted(installed)))

    failed = []
    for name, ok, detail in checks:
        status = "PASS " if ok else "FAIL "
        print(f"{status} {name} | got={detail!r}")
        if not ok:
            failed.append(name)
    assert not failed, f"{len(failed)} comprobacion(es) del contrato de default_engine"
    print(f"OK: {len(checks)} comprobaciones de default_engine de Generate")


if __name__ == "__main__":
    main()
