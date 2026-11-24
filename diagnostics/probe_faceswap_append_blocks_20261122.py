"""Probe v5.1408 — añadir bloques de fotos en origen y destino de FaceSwap.

Comprueba con la UI real (servidor aislado):
1. Un segundo bloque en «Caras Origen» suma caras al banco y no duplica las ya cargadas.
2. Un segundo bloque en «Caras Destino» suma archivos a la cola sin borrar el anterior.
3. Repetir un archivo no duplica nada.
4. Limpiar deja el registro a cero y permite volver a cargar.
"""

import socket
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402
from gradio_client import Client, handle_file  # noqa: E402

import roop.globals  # noqa: E402
import ui.globals as ui_globals  # noqa: E402
import ui.tabs.faceswap.state as state  # noqa: E402
from ui.tabs.faceswap import events, ui  # noqa: E402

FACES = Path(r"D:\PROJECTS\uee\FACES\AATL")
A = FACES / "grok_image_1773140604875.jpg"
B = FACES / "paloma.jpg"


def main():
    assert A.is_file() and B.is_file(), (A, B)
    with gr.Blocks(title="FaceSwap append blocks probe") as demo:
        comps = ui.build_faceswap_ui()
        events.wire_events(comps)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    demo.launch(server_name="127.0.0.1", server_port=port,
                prevent_thread_lock=True, quiet=True)
    try:
        for _ in range(30):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5).read()
                break
            except Exception:
                time.sleep(0.5)
        client = Client(f"http://127.0.0.1:{port}", verbose=False)
        a, b = handle_file(str(A)), handle_file(str(B))

        # --- Origen: primer bloque y segundo bloque (gr.Files acumula) ---
        client.predict([a], api_name="/_safe_on_src_changed")
        first_faces = len(roop.globals.INPUT_FACESETS)
        assert first_faces > 0, "el primer bloque de origen no dio caras"
        client.predict([a, b], api_name="/_safe_on_src_changed")
        second_faces = len(roop.globals.INPUT_FACESETS)
        assert second_faces > first_faces, (
            f"el segundo bloque no añadió caras: {first_faces} -> {second_faces}"
        )
        client.predict([a, b], api_name="/_safe_on_src_changed")
        assert len(roop.globals.INPUT_FACESETS) == second_faces, "duplicó caras al repetir"
        assert len(ui_globals.ui_input_thumbs) == second_faces, "miniaturas desincronizadas"
        print(f"[ORIGEN] {first_faces} caras + segundo bloque -> {second_faces} (sin duplicados)")

        # --- Destino: primer bloque, segundo bloque y repetido ---
        client.predict(files=[a], api_name="/on_dest_upload")
        first_dest = len(state.list_files_process)
        assert first_dest == 1, f"destino inicial: {first_dest}"
        client.predict(files=[a, b], api_name="/on_dest_upload")
        second_dest = len(state.list_files_process)
        assert second_dest == 2, f"el segundo bloque no se sumó: {second_dest}"
        client.predict(files=[b], api_name="/on_dest_upload")
        assert len(state.list_files_process) == 2, "duplicó destino al repetir"
        print(f"[DESTINO] 1 archivo + segundo bloque -> {second_dest} (sin duplicados)")

        # --- Limpieza y recarga ---
        client.predict(api_name="/clear_source")
        client.predict(api_name="/clear_target")
        assert not roop.globals.INPUT_FACESETS and not state.list_files_process
        client.predict([a], api_name="/_safe_on_src_changed")
        assert len(roop.globals.INPUT_FACESETS) == first_faces, "no se pudo recargar tras limpiar"
        client.predict(files=[a], api_name="/on_dest_upload")
        assert len(state.list_files_process) == 1, "no se pudo recargar destino tras limpiar"
        print("[LIMPIEZA] registros a cero y recarga correcta")
    finally:
        demo.close()
    print("PASS — probe bloques origen/destino v5.1408")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
