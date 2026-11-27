"""Probe v5.1408 — añadir bloques de fotos en origen y destino de FaceSwap.

Comprueba con la UI real (servidor aislado):
1. Un segundo bloque en «Caras Origen» suma caras al banco y no duplica las ya cargadas.
2. Un segundo bloque en «Caras Destino» suma archivos a la cola sin borrar el anterior.
3. Repetir un archivo no duplica nada.
4. Limpiar deja el registro a cero y permite volver a cargar.
"""

import socket
import os
import sys
import time
from types import SimpleNamespace
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402
from gradio_client import Client, handle_file  # noqa: E402

import roop.globals  # noqa: E402
import ui.globals as ui_globals  # noqa: E402
from ui.tabs.faceswap import logic  # noqa: E402
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
        # v5.1465: sin caja de subida de destino; se carga por ruta directa.
        def load_dest(*paths):
            for path in paths:
                client.predict(str(path), api_name="/on_add_target_path")

        load_dest(A)
        first_dest = len(state.list_files_process)
        assert first_dest == 1, f"destino inicial: {first_dest}"
        assert state.SELECTION_FACES_DATA, "no aparecieron candidatas del destino activo"
        # v5.1463: cargar un destino no elige caras; solo las muestra.
        assert not ui_globals.ui_target_thumbs, "la carga eligió una cara sola"
        click = next(fn.fn for fn in demo.fns.values()
                     if getattr(fn.fn, "__name__", "") == "on_face_selection_click")
        click(SimpleNamespace(index=0), 1)
        assert len(ui_globals.ui_target_thumbs) == 1, "el clic no añadió la cara elegida"
        load_dest(A, B)
        second_dest = len(state.list_files_process)
        assert second_dest == 2, f"el segundo bloque no se sumó: {second_dest}"
        assert len(ui_globals.ui_target_thumbs) == 1, (
            "cargar un bloque de destino eligió caras otras sola"
        )
        load_dest(B)
        assert len(state.list_files_process) == 2, "duplicó destino al repetir"
        print(f"[DESTINO] 1 archivo + segundo bloque -> {second_dest} (sin duplicados)")

        # --- Limpieza y recarga ---
        client.predict(api_name="/clear_source")
        client.predict(api_name="/clear_target")
        assert not roop.globals.INPUT_FACESETS and not state.list_files_process
        client.predict([a], api_name="/_safe_on_src_changed")
        assert len(roop.globals.INPUT_FACESETS) == first_faces, "no se pudo recargar tras limpiar"
        load_dest(A)
        assert len(state.list_files_process) == 1, "no se pudo recargar destino tras limpiar"
        client.predict(api_name="/clear_target")
        load_dest(A)
        assert len(state.list_files_process) == 1, "el segundo vaciado impidió recargar el mismo destino"
        print("[LIMPIEZA] registros a cero y recarga correcta")

        # Drop nativo: el servidor recibe rutas, y la UI solo devuelve nombres.
        client.predict(api_name="/clear_source")
        client.predict(api_name="/clear_target")
        with roop.globals.NATIVE_DROP_LOCK:
            roop.globals.NATIVE_DROP_QUEUES["faceswap_src"].extend([str(A), str(B)])
            roop.globals.NATIVE_DROP_QUEUES["faceswap_dest"].extend([str(A), str(B)])
        client.predict(api_name="/poll_native_src_drop")
        client.predict(api_name="/poll_native_dest_drop")
        assert len(roop.globals.INPUT_FACESETS) == 2
        assert len(ui_globals.ui_input_thumbs) == 2, "el origen no listó las dos caras"
        assert len(state.list_files_process) == 2
        names = [os.path.basename(entry.filename) for entry in state.list_files_process]
        assert A.name in names and B.name in names
        assert not ui_globals.ui_target_thumbs, "el drop nativo eligió una cara solo"
        print("[DROP NATIVO] dos bloques listados en origen y destino")

        # Una referencia por archivo se elige a mano (v5.1463).
        for entry in state.list_files_process:
            name = os.path.basename(entry.filename)
            if events._get_file_bank(name, entry.filename):
                continue
            faces = logic.extract_face_images(
                entry.filename, (True, 1), target_face_detection=True, ui_padding=0.5,
            )
            events._register_destination_face(entry, faces[0][0], faces[0][1])

        # A manually chosen reference on another frame must survive a trip
        # to the next file and back, even when frame 1 has one different face.
        previous, following = state.list_files_process
        first_name = os.path.basename(previous.filename)
        second_name = os.path.basename(following.filename)
        key = f"selected_face_ref_{first_name}"
        original_bank = roop.globals.selected_face_references[key]
        manual = dict(events._get_file_bank(second_name, following.filename)[0])
        manual["source_path"] = os.path.normcase(os.path.abspath(previous.filename))
        navigate = {}
        for block_fn in demo.fns.values():
            fn = block_fn.fn
            defaults = getattr(fn, "__defaults__", None)
            closure = getattr(fn, "__closure__", None) or ()
            if defaults in (("prev",), ("next",)) and any(
                getattr(cell.cell_contents, "__name__", "") == "navigate_file"
                for cell in closure
            ):
                navigate[defaults[0]] = fn
        assert set(navigate) == {"prev", "next"}, "callbacks de navegación no encontrados"
        try:
            roop.globals.selected_face_references[key] = [manual]
            navigate["next"]()
            navigate["prev"]()
            bank = events._get_file_bank(first_name, previous.filename)
            assert len(bank) == 1 and bank[0] is manual, (
                "al volver se añadió la cara de frame 1 sobre la referencia manual"
            )
        finally:
            roop.globals.selected_face_references[key] = original_bank
        print("[DESTINO] selección manual conservada tras ida/vuelta")
    finally:
        demo.close()
    print("PASS — probe bloques origen/destino v5.1408")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
