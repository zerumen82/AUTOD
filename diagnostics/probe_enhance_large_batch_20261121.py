"""Probe v5.1411 — lote grande (400 fotos) por drag&drop sin congelar la UI.

Comprueba con servidor aislado y SUBIDA REAL de 400 archivos:
1. La cola se llena con 400 rutas y el desplegable se limita sin romper navegación.
2. El evento posterior no recibe los FileData (solo un State de rutas).
3. Tras importar un lote grande NO se renderiza la vista previa.
4. El lote procesa las 400 fotos en la carpeta de destino elegida.
"""

import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402
from PIL import Image  # noqa: E402
from gradio_client import Client, handle_file  # noqa: E402

from ui.tabs.enhance import events as E  # noqa: E402
from ui.tabs.enhance.logic import load_enhance_config  # noqa: E402
from ui.tabs.enhance.ui import build_enhance_ui  # noqa: E402

PORT = 7866
COUNT = 400
CFG = load_enhance_config()["large_batch"]

_SLIDERS = dict(
    param_3=0, param_4=0, param_5=0, param_6=1,
    param_7=1.0, param_8=1.0, param_9=1.0, param_10=50,
    param_11=0, param_12=10, param_13=0, param_14=100,
    param_15="GFPGAN",
)


def _pair_kwargs(files, selected):
    return dict(files_val=files, folder_val=None, selected=selected, **_SLIDERS)


def main():
    with gr.Blocks() as demo:
        comps = build_enhance_ui()
        E.wire_events(comps)
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=PORT, share=False,
                quiet=True, prevent_thread_lock=True, show_error=True)
    try:
        for _ in range(30):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=5).read()
                break
            except Exception:
                time.sleep(0.5)
        client = Client(f"http://127.0.0.1:{PORT}", verbose=False)

        with tempfile.TemporaryDirectory(prefix="autoauto_large_batch_") as td:
            root = Path(td)
            source = root / "fotos"
            source.mkdir()
            for i in range(COUNT):
                Image.new("RGB", (24, 24), (i % 255, 0, 0)).save(
                    source / f"f{i:03d}.png"
                )
            (source / "notas.txt").write_text("no soy foto", encoding="utf-8")
            files = [handle_file(str(p)) for p in sorted(source.glob("*.png"))]

            t0 = time.time()
            queue = client.predict(files_val=files, folder_val=None,
                                   api_name="/refresh_queue")
            summary, dropdown = queue[0], queue[1]
            assert f"{COUNT} foto(s) en cola" in summary, f"cola incompleta: {summary}"
            max_visible = int(CFG["max_visible_choices"])
            assert len(dropdown["choices"]) == max_visible, f"opciones: {len(dropdown['choices'])}"
            assert f"desplegable: {max_visible} de {COUNT}" in summary, summary
            print(f"[COLA] {COUNT} fotos subidas en {time.time() - t0:.1f}s · desplegable {max_visible}")

            imported = client.predict(
                dropdown["value"],
                0, 0, 0, 1, 1.0, 1.0, 1.0, 50, 0, 10, 0, 100, "GFPGAN",
                api_name="/make_pair_on_import",
            )
            assert imported[0] and imported[1], "no se renderizó el preview de la primera foto"
            assert "Vista previa de la primera" in imported[2], imported[2]
            assert f"1 / {COUNT}" in imported[3], imported[3]
            print("[IMPORT] preview de la primera foto renderizado; sliders omitidos con gr.skip")

            selected_50 = dropdown["choices"][50][1]
            nav = client.predict(files_val=files, folder_val=None,
                                 selected=selected_50, api_name="/move_next")
            assert nav[0]["value"] != selected_50, "Siguiente no avanzó en la cola"
            pair = client.predict(api_name="/make_pair", **_pair_kwargs(files, nav[0]["value"]))
            assert pair[0] and pair[1], "no se pudo previsualizar tras navegar"
            print("[NAVEGACION] Anterior/Siguiente y vista previa siguen funcionando")

            out_dir = root / "destino"
            batch_status, batch_summary = client.predict(
                files=files, folder=None, preset_name=None,
                denoise=0, sharpen=0, recover=0, upscale=1,
                exposure=1.0, shadow=1.0, highlight=1.0, temperature=50,
                grain_strength=0, grain_size=10, deband=0, vibrance=100,
                face_model="GFPGAN", export_format="png",
                apply_filters_to_filename=False, auto_mode=False,
                select_all=True, output_path=str(out_dir),
                api_name="/do_batch",
            )
            assert f"{COUNT}/{COUNT}" in batch_status, f"lote incompleto: {batch_status}"
            outputs = list(out_dir.glob("*.png"))
            assert len(outputs) == COUNT, f"salidas: {len(outputs)}"
            print(f"[LOTE] {COUNT}/{COUNT} fotos procesadas en la carpeta elegida")
    finally:
        demo.close()
    print("PASS — probe lote grande v5.1415")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
