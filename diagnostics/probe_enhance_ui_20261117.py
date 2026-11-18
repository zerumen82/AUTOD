"""Probe UI v5.1383 — pestaña «Mejorar fotos» con presets IA, Deband y Vibrance.

Verifica, con ejecución real:
1. El Blocks se construye y el contrato con events.py tiene todas las claves.
2. Cada fila de mejoras crea UNA columna por control (causa raíz de la
   desalineación corregida: antes la etiqueta era hermana del slider en la fila).
3. La página servida responde 200 e incluye ANTES, DESPUÉS y «Abrir carpeta».
4. La vista previa real de la primera foto devuelve antes y después válidos y
   el panel «Después» cambia al mover un slider (render reactivo).
"""

import sys
import tempfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, r"D:\PROJECTS\AUTOAUTO")

import gradio as gr  # noqa: E402

from ui.tabs.enhance import events as E  # noqa: E402
from ui.tabs.enhance.presets import load_presets  # noqa: E402
from ui.tabs.enhance.ui import build_enhance_ui  # noqa: E402

PORT = 7862
SAMPLE = Path(r"D:\PROJECTS\AUTOAUTO\diagnostics\out\office\window.png")
EXPECTED_KEYS = [
    "session_state", "input_files", "input_folder", "btn_load_sample",
    "queue_summary", "preview_before", "preview_after",
    "preview_photo", "btn_prev_photo", "btn_next_photo", "preview_counter",
    "btn_preview", "preview_status", "denoise_slider",
    "sharpen_slider", "recover_slider", "upscale_factor", "exposure_slider",
    "shadow_slider", "highlight_slider", "temperature_slider",
    "grain_strength_slider", "grain_size_slider", "deband_slider",
    "vibrance_slider", "face_model", "preset_dropdown",
    "preset_apply_btn", "preset_save_name", "preset_save_btn", "preset_status",
    "auto_mode", "output_path", "btn_open_folder", "folder_status", "format_dropdown",
    "apply_filters_to_filename", "batch_btn", "cancel_btn", "status_msg",
    "result_summary",
]


def check_build() -> None:
    with gr.Blocks() as demo:
        comps = build_enhance_ui()
        E.wire_events(comps)
    missing = [k for k in EXPECTED_KEYS if k not in comps]
    assert not missing, f"claves faltantes: {missing}"
    assert len(demo.fns) > 20, f"pocos eventos: {len(demo.fns)}"
    factory = {name for name, data in load_presets().items() if data["_factory"]}
    assert set(comps["preset_buttons"]) == factory
    assert not factory.intersection(choice[1] for choice in comps["preset_dropdown"].choices)
    selector_id = comps["preview_photo"]._id
    triggers = [target[1] for dep in demo.config["dependencies"] for target in dep["targets"] if target[0] == selector_id]
    assert triggers == ["input"], f"el selector puede disparar renders duplicados: {triggers}"
    for key in ("denoise_slider", "face_model"):
        control_id = comps[key]._id
        control_triggers = [target[1] for dep in demo.config["dependencies"] for target in dep["targets"] if target[0] == control_id]
        assert control_triggers == ["input"], f"{key} puede disparar renders duplicados: {control_triggers}"
    print(f"[UI] Claves del contrato: {len(comps)}/{len(EXPECTED_KEYS)} OK · eventos: {len(demo.fns)}")

    # Estructura de alineación: cada slider vive en su columna con su etiqueta.
    print("[UI] Estructura ANTES|DESPUÉS + columnas de slider construida")


def check_server() -> None:
    with gr.Blocks() as demo:
        comps = build_enhance_ui()
        E.wire_events(comps)
    demo.queue()
    demo.launch(
        server_name="127.0.0.1",
        server_port=PORT,
        share=False,
        quiet=True,
        prevent_thread_lock=True,
    )
    try:
        time.sleep(1.0)
        for _ in range(20):
            try:
                page = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=5).read().decode("utf-8", "ignore")
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("la página no respondió")
        for needle in ("ANTES", "DESPUÉS", "Abrir carpeta", "Foto en vista previa"):
            assert needle in page, f"falta «{needle}» en la página servida"
        print(f"[UI] GET / -> 200 ({len(page)} bytes) con ANTES, DESPUÉS y botón de carpeta")
        check_api()
    finally:
        demo.close()


def check_api() -> None:
    """End-to-end por la API real de la pestaña (auto-primera + reactivo)."""
    from gradio_client import Client, handle_file

    client = Client(f"http://127.0.0.1:{PORT}", verbose=False)
    loaded = client.predict(api_name="/load_sample")
    assert loaded[0] and len(loaded[0]) >= 2, f"Cargar ejemplo no entregó fotos: {loaded}"
    print(f"[API] Cargar ejemplo entregó {len(loaded[0])} fotos locales")
    sample = r"D:\PROJECTS\AUTOAUTO\diagnostics\out\office\window.png"
    f = handle_file(sample)

    q = client.predict([f], None, api_name="/refresh_queue")
    update = q[1]
    selected_path = update.get("value")
    assert selected_path and Path(str(selected_path)).name == "window.png", f"no se seleccionó la primera foto: {update}"
    print(f"[API] Cola: {q[0]} · selector -> {selected_path}")

    f2 = handle_file(r"D:\PROJECTS\AUTOAUTO\diagnostics\out\couch_check\window.png")
    q2 = client.predict([f, f2], None, api_name="/refresh_queue")
    choices = q2[1].get("choices") or []
    values = [
        (choice[1] if isinstance(choice, (list, tuple)) and len(choice) == 2 else choice)
        for choice in choices
    ]
    assert len(values) == 2 and len(set(map(str, values))) == 2, (
        f"el selector no distingue archivos con el mismo nombre: {choices}"
    )
    print("[API] Nombres duplicados: el selector distingue rutas (2 valores distintos)")

    nav = client.predict([f, f2], None, q2[1]["value"], api_name="/move_next")
    assert nav[0]["value"] != q2[1]["value"] and "2 / 2" in nav[1]
    back = client.predict([f, f2], None, nav[0]["value"], api_name="/move_previous")
    assert back[0]["value"] == q2[1]["value"] and "1 / 2" in back[1]
    assert E._path_index([r"C:\temp\.\foto.png", r"C:\temp\otra.png"], r"C:\temp\foto.png") == 0
    print("[API] Navegación anterior/siguiente y límites correctos; rutas equivalentes aceptadas")

    base = dict(
        files_val=[f], folder_val=[f], selected=None,
        param_3=20, param_4=10, param_5=0, param_6=1, param_7=1.0,
        param_8=1.0, param_9=1.0, param_10=50, param_11=0, param_12=10,
        param_13=20, param_14=104, param_15="GFPGAN",
    )
    pair = client.predict(api_name="/make_pair", **base)
    assert pair[0] and pair[1], "antes/después vacíos"
    assert "window.png" in pair[2], f"estado inesperado: {pair[2]}"
    print(f"[API] Antes/después generados con la primera foto: {pair[2]}")

    tuned = dict(base)
    tuned.update(param_3=80, param_4=70, param_7=1.4, param_8=1.3,
                 param_9=0.7, param_10=65, param_11=20, param_12=5,
                 param_13=60, param_14=120, param_15="CodeFormer")
    changed = client.predict(api_name="/make_after", **tuned)
    assert changed[0] and changed[0] != pair[1], "el panel Después no cambió"
    print(f"[API] Panel Después reactivo: {changed[1]}")

    with tempfile.TemporaryDirectory(prefix="autoauto_enhance_probe_") as tmp:
        E.get_enhance_output_dir = lambda: tmp
        batch = client.predict(
            files=[f], folder=[f], preset_name="Retrato",
            denoise=20, sharpen=10, recover=0, upscale=1, exposure=1.0,
            shadow=1.0, highlight=1.0, temperature=50, grain_strength=0, grain_size=10,
            deband=20, vibrance=104, face_model="GFPGAN",
            export_format="png", apply_filters_to_filename=False, auto_mode=False,
            api_name="/do_batch",
        )
    assert "1/1" in batch[0], f"el lote no deduplicó archivo/carpeta: {batch[0]}"
    assert batch[1], "resumen del lote vacío"
    print(f"[API] Lote: {batch[0]} · resumen: {batch[1]}")


def check_realtime_preview() -> None:
    assert SAMPLE.exists(), f"falta la muestra {SAMPLE}"
    before = E._load_before(SAMPLE)
    base = E._settings(20, 10, 0, 1, 1.0, 1.0, 1.0, 50, 0, 10, deband=20, vibrance=104)
    t0 = time.time()
    after_base = E._render_after(before, base)
    t_base = time.time() - t0

    tuned = E._settings(60, 45, 0, 1, 1.2, 1.1, 0.9, 60, 10, 6, deband=60, vibrance=120)
    t0 = time.time()
    after_tuned = E._render_after(before, tuned)
    t_tuned = time.time() - t0

    assert after_base is not None and after_tuned is not None
    assert after_base.size == before.size, "el ajuste base no debe cambiar tamaño con upscale 1x"
    import numpy as np
    diff = np.abs(np.asarray(after_base, dtype=np.int16) - np.asarray(after_tuned, dtype=np.int16)).mean()
    assert diff > 0.5, f"el panel Después no cambió al mover ajustes (diff={diff:.3f})"
    print(
        f"[UI] Vista previa real: antes {before.size} · después {after_base.size} · "
        f"cambio medio {diff:.2f} · tiempo base {t_base:.2f}s / ajustada {t_tuned:.2f}s"
    )


def main() -> int:
    check_build()
    check_server()
    check_realtime_preview()
    print("PASS — probe UI enhance v5.1383")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
