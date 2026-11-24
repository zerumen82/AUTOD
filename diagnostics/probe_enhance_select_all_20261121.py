"""Probe v5.1404 — «Seleccionar todas», presets, carpeta, destino y sRGB/254 PPP.

Verifica con ejecución real (servidor aislado, sin tocar la UI activa):
1. El tab construye y el contrato incluye la casilla `select_all`.
2. Los 15 presets de fábrica cargan con su receta e icono.
3. El modo carpeta (payload de lista de archivos) deduplica y descarta no-imágenes.
4. El lote exporta PNG y JPG con perfil ICC sRGB y 254 PPP configurados.
5. Con «Seleccionar todas» ON las dos fotos reciben la MISMA receta (salidas
   idénticas); con OFF conservan la receta por foto (salidas distintas).
6. El destino elegido se crea y se usa; una ruta inválida cae a enhance/ con aviso.
"""

import hashlib
import io
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402
from PIL import Image, ImageCms  # noqa: E402

from ui.tabs.enhance import events as E  # noqa: E402
from ui.tabs.enhance.logic import load_enhance_config, process_batch  # noqa: E402
from ui.tabs.enhance.presets import SLIDER_KEYS, icon_for_preset, load_presets  # noqa: E402
from ui.tabs.enhance.ui import build_enhance_ui  # noqa: E402

PORT = 7863
SAMPLE = ROOT / "diagnostics" / "out" / "office" / "window.png"
EXPECTED_KEYS = [
    "session_state", "input_files", "input_folder", "select_all",
    "btn_load_sample", "queue_summary", "preview_before", "preview_after",
    "preview_photo", "btn_prev_photo", "btn_next_photo", "preview_counter",
    "btn_preview", "preview_status", "denoise_slider", "sharpen_slider",
    "recover_slider", "upscale_factor", "exposure_slider", "shadow_slider",
    "highlight_slider", "temperature_slider", "grain_strength_slider",
    "grain_size_slider", "deband_slider", "vibrance_slider", "face_model",
    "btn_reset_sliders", "preset_dropdown", "preset_apply_btn",
    "preset_save_name", "preset_save_btn", "preset_status", "preset_buttons",
    "auto_mode", "output_path", "btn_browse_folder", "btn_open_folder", "folder_status",
    "format_dropdown", "apply_filters_to_filename", "batch_btn", "cancel_btn",
    "status_msg", "result_summary",
]


def check_build():
    with gr.Blocks() as demo:
        comps = build_enhance_ui()
        E.wire_events(comps)
    missing = [k for k in EXPECTED_KEYS if k not in comps]
    assert not missing, f"faltan claves del contrato: {missing}"
    assert comps["select_all"].value is False, "la casilla debe nacer desactivada"
    assert len(demo.fns) > 20, f"pocos eventos cableados: {len(demo.fns)}"
    factory = {name for name, data in load_presets().items() if data["_factory"]}
    assert set(comps["preset_buttons"]) == factory, "los botones de preset no coinciden"
    print(f"[BUILD] Contrato {len(comps)}/{len(EXPECTED_KEYS)} · eventos {len(demo.fns)} · select_all OK")


def check_presets():
    catalog = load_presets()
    factory = {name: data for name, data in catalog.items() if data["_factory"]}
    assert len(factory) == 15, f"catálogo incompleto: {list(factory)}"
    for name, data in factory.items():
        assert icon_for_preset(data).is_file(), f"sin icono: {name}"
        for key in SLIDER_KEYS:
            assert key in data["sliders"], f"receta incompleta {name}: {key}"
    assert factory["Blanco y negro"]["sliders"]["vibrance"] == 0, "blanco y negro roto"
    print(f"[PRESETS] {len(factory)}/15 válidos con receta e icono; vibrance=0 correcto")


def check_folder_mode():
    with tempfile.TemporaryDirectory(prefix="autoauto_folder_probe_") as td:
        a, b, junk = (Path(td) / name for name in ("a.png", "b.png", "notas.txt"))
        Image.new("RGB", (8, 8), (1, 2, 3)).save(a)
        Image.new("RGB", (8, 8), (4, 5, 6)).save(b)
        junk.write_text("no soy una foto", encoding="utf-8")
        # gr.File(file_count="directory") entrega una lista de rutas de archivo.
        queued = E._queue_paths(None, [str(a), str(b), str(junk)])
        assert len(queued) == 2, f"la carpeta no se enumeró bien: {queued}"
        assert all(Path(p).suffix.lower() == ".png" for p in queued)
        dedup = E._queue_paths([str(a)], [str(a), str(b)])
        assert len(dedup) == 2, f"archivo+carpeta no deduplicó: {dedup}"
    print("[CARPETA] payload de directorio: 2 imágenes, 1 no-imagen descartada, duplicados unidos")


def _settings_flat():
    """Ajustes neutros para las pruebas de exportación (rápidos y deterministas)."""
    return E._settings(0, 0, 0, 1, 1.0, 1.0, 1.0, 50, 0, 10, deband=0,
                       vibrance=100, face_model="GFPGAN")


def check_export_metadata():
    export_cfg = load_enhance_config()["export"]
    dpi = int(export_cfg["dpi"])
    assert str(export_cfg["color_profile"]).casefold() == "srgb"
    with tempfile.TemporaryDirectory(prefix="autoauto_meta_probe_") as td:
        src = Path(td) / "source.png"
        with Image.open(SAMPLE) as opened:
            img = opened.convert("RGB")
        img.thumbnail((64, 64), Image.Resampling.LANCZOS)
        img.save(src)
        for fmt in ("png", "jpg"):
            out_dir = Path(td) / fmt
            results = process_batch([src], _settings_flat(), out_dir, fmt,
                                    apply_filters_to_filename=False,
                                    allow_overwrite=True)
            assert results and results[0]["success"], f"falló el lote {fmt}: {results}"
            with Image.open(results[0]["output"]) as saved:
                assert saved.info.get("dpi") == (float(dpi), float(dpi)), (
                    f"{fmt} sin {dpi} PPP: {saved.info.get('dpi')}"
                )
                icc = saved.info.get("icc_profile")
                assert icc, f"{fmt} sin perfil ICC"
                desc = ImageCms.getProfileDescription(
                    ImageCms.ImageCmsProfile(io.BytesIO(icc))
                )
                assert "sRGB" in desc, f"{fmt} con perfil {desc!r}"
            print(f"[EXPORT] {fmt}: {dpi} PPP + perfil {desc.strip()!r}")


def check_batch_select_all():
    with gr.Blocks() as demo:
        comps = build_enhance_ui()
        E.wire_events(comps)
    demo.queue()
    demo.launch(
        server_name="127.0.0.1", server_port=PORT, share=False,
        quiet=True, prevent_thread_lock=True,
    )
    try:
        for _ in range(30):
            try:
                page = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=5).read().decode("utf-8", "ignore")
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("la página aislada no respondió")
        assert "Examinar" in page and "Carpeta de destino" in page, "falta el selector de destino"

        from gradio_client import Client, handle_file

        client = Client(f"http://127.0.0.1:{PORT}", verbose=False)
        with tempfile.TemporaryDirectory(prefix="autoauto_selectall_probe_") as td:
            root = Path(td)
            files = []
            for name in ("copy_a.png", "copy_b.png"):
                path = root / name
                with Image.open(SAMPLE) as opened:
                    img = opened.convert("RGB")
                img.thumbnail((96, 96), Image.Resampling.LANCZOS)
                img.save(path)
                files.append(handle_file(str(path)))

            queue = client.predict(files_val=files, folder_val=None,
                                   api_name="/refresh_queue")
            selected = [
                choice[1] if isinstance(choice, (list, tuple)) else choice
                for choice in (queue[1]["choices"] or [])
            ]
            assert len(selected) == 2, f"cola inesperada: {queue[1]}"

            def set_per_photo(photo, denoise):
                client.predict(
                    files_val=files, folder_val=None, selected=photo,
                    param_3=denoise, param_4=0, param_5=0, param_6=1,
                    param_7=1.0, param_8=1.0, param_9=1.0, param_10=50,
                    param_11=0, param_12=10, param_13=0, param_14=100,
                    param_15="GFPGAN", api_name="/make_after",
                )

            # Dos fotos idénticas con recetas distintas: A sin denoise, B con 80.
            set_per_photo(selected[0], 0)
            set_per_photo(selected[1], 80)

            def run_batch(select_all, out_name, dest=None):
                default_dir = root / out_name
                E.get_enhance_output_dir = lambda: str(default_dir)
                status, summary = client.predict(
                    files=files, folder=None, preset_name=None,
                    denoise=0, sharpen=0, recover=0, upscale=1,
                    exposure=1.0, shadow=1.0, highlight=1.0, temperature=50,
                    grain_strength=0, grain_size=10, deband=0, vibrance=100,
                    face_model="GFPGAN", export_format="png",
                    apply_filters_to_filename=False, auto_mode=False,
                    select_all=select_all, output_path=dest,
                    api_name="/do_batch",
                )
                assert "2/2" in status, f"lote incompleto: {status}"
                expected = Path(dest) if dest and Path(dest).is_dir() else default_dir
                outputs = sorted(expected.glob("*.png"))
                assert len(outputs) == 2, f"salidas inesperadas en {expected}: {outputs}"
                return [hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs], summary

            hashes_off, _ = run_batch(False, "off")
            hashes_on, summary_on = run_batch(True, "on")
            assert hashes_off[0] != hashes_off[1], (
                "OFF debe conservar la receta por foto (salidas distintas)"
            )
            assert hashes_on[0] == hashes_on[1], (
                "ON debe aplicar UNA MISMA receta a todas (salidas idénticas)"
            )
            assert "mismo ajuste para todas" in summary_on
            print("[LOTE] OFF → 2 recetas distintas · ON → 2 salidas idénticas · resumen marcado")

            # Destino elegido: carpeta nueva (se crea) y ruta inválida (fallback).
            custom_dir = root / "destino_elegido"
            hashes_custom, summary_custom = run_batch(True, "custom_default", str(custom_dir))
            assert custom_dir.is_dir(), "la carpeta elegida no se creó"
            assert hashes_custom[0] == hashes_custom[1], "en el destino elegido no se aplicó una sola receta"
            assert str(custom_dir) in summary_custom, "el resumen no muestra el destino elegido"

            invalid_dest = root / "copy_a.png"  # archivo existente, no carpeta
            _, summary_fallback = run_batch(False, "fallback", str(invalid_dest))
            assert "Se mantiene" in summary_fallback, "destino inválido sin aviso de fallback"
            print("[DESTINO] carpeta elegida creada y usada · ruta inválida → fallback a enhance/ con aviso")
    finally:
        demo.close()


def main():
    check_build()
    check_presets()
    check_folder_mode()
    check_export_metadata()
    check_batch_select_all()
    print("PASS — probe select-all/destino/sRGB v5.1404")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
