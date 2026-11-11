# -*- coding: utf-8 -*-
"""Probe: tres mejoras acotadas del Modo Calidad (2026-11-11, aprobadas).

1. Verificador visual con VRAM: el bloque del verificador libera las sesiones
   ONNX locales (ESRGAN del finisher, CodeFormer) ANTES de cargar Moondream;
   en híbrido 8K se saltaba la verificación (277 MiB libres de 1891).
2. Desposterizar con la fuerza que pide la medida: con poster ≥0.50 el tope
   de heavy_tier (0.18) ya no aplasta la limpieza; la puerta (band/poster)
   sigue revirtiendo si empeora.
3. Medición de pelo residual también en la ruta Kontext focal: la zona es
   `_longcat_focal_mask` (final_mask pasa a None en esa ruta) y la medición
   es de aviso, nunca descarta el resultado.

Verificación CPU: ejecución real de `_classical_finish_strengths` + AST del
fuente del manager (sin stubs, sin listas paralelas).
"""

import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MANAGER = os.path.join(ROOT, "roop", "img_editor", "img_editor_manager.py")

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def main():
    src = open(MANAGER, encoding="utf-8").read()

    # --- Mejora 1: liberación de VRAM antes del verificador ---
    # El bloque v5.1331 está en la verificación FINAL (hay otra llamada del
    # verificador en la ruta compuesta ~9622, sin acabado TOP delante).
    fix_idx = src.find("# v5.1331")
    verify_idx = src.find("get_instruction_verifier().verify(", fix_idx)
    release_idx = src.find("release_gpu_caches()", fix_idx)
    preserve_idx = src.find("_preserve_faces._cf = None", fix_idx)
    check("bloque_fix_encontrado", fix_idx > 0, f"idx={fix_idx}")
    check("verificador_libera_esrgan", 0 < release_idx < verify_idx)
    check("verificador_libera_codeformer", 0 < preserve_idx < verify_idx)

    # --- Mejora 3: medición de pelo sin exclusión de Kontext ---
    hair_zone_src = src[src.find("_hair_plan_zone = None"):src.find("_hair_plan_zone = None") + 900]
    check(
        "zona_pelo_kontext_focal",
        "_longcat_focal_mask" in hair_zone_src,
    )
    gate_line = [l for l in hair_zone_src.splitlines() if "_is_bald_request" in l]
    check(
        "medicion_pelo_sin_exclusion_kontext",
        len(gate_line) == 1 and 'engine != "kontext"' not in gate_line[0],
        gate_line[0].strip() if gate_line else "",
    )
    # La selección PRE-resultado (qué cabeza editar) conserva su exclusión:
    # eso cambia la ruta, no es la medición de aviso.
    pre_idx = src.find('engine != "kontext" and self._is_bald_request')
    check("seleccion_pre_resultado_intacta", pre_idx > 0)

    # --- Mejora 2: fuerza de desposterizar guiada por la medida ---
    from roop.img_editor.top_tier_finish import _classical_finish_strengths

    def strengths(poster, sharpness=0.30, heavy=True, after_edit=True):
        scores = {
            "sharpness": sharpness, "poster": poster, "restore": 0.30,
            "max_side": 5586, "target_side": 5586, "noise": 0.10,
        }
        return _classical_finish_strengths(
            scores, {}, after_edit=after_edit, heavy_tier=heavy,
        )

    # Caso medido hoy (log 2026-11-11): poster 0.72, foto grande → dep > 0.18.
    high = strengths(0.72)
    check("poster_alto_sube_limpieza", high["depixelize"] > 0.18, f"dep={high['depixelize']:.3f}")
    check("poster_alto_con_tope_30", high["depixelize"] <= 0.30)
    # Foto grande limpia: el tope previo de heavy_tier se conserva.
    low = strengths(0.20)
    check("poster_bajo_tope_intacto", low["depixelize"] <= 0.18, f"dep={low['depixelize']:.3f}")
    # Frontera: poster 0.50 no baja del tope previo pero activa la subida.
    mid = strengths(0.50)
    check("frontera_050_en_tope_previo", abs(mid["depixelize"] - 0.18) < 1e-9, f"dep={mid['depixelize']:.3f}")
    # Caso gentle (nítida y al tamaño): sin cambio de comportamiento.
    gentle = strengths(0.60, sharpness=0.90)
    check("caso_gentle_intacto", abs(gentle["depixelize"] - 0.12) < 1e-9, f"dep={gentle['depixelize']:.3f}")

    # La puerta del desposterizar sigue comprobando bandas+poster (reversión).
    top_src = open(os.path.join(ROOT, "roop", "img_editor", "top_tier_finish.py"), encoding="utf-8").read()
    deposter_gate = 'checks=("band", "poster")' in top_src
    check("puerta_band_poster_presente", deposter_gate)

    # --- Manager compila e importa (ejecución real) ---
    from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: F401
    check("manager_importa", True)

    fails = [n for n, ok, _ in results if not ok]
    print(f"\nRESULTADO: {'PASS' if not fails else 'FAIL'} ({len(fails)} fallos)")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
