"""Exercise the real Gradio source upload event in an isolated UI process."""

import socket
from pathlib import Path

import gradio as gr
from gradio_client import Client, handle_file

from ui.tabs.faceswap import events, ui
import roop.globals


ROOT = Path(__file__).resolve().parents[1]
PHOTO = Path(r"D:\PROJECTS\uee\FACES\MARETA\P1010496.JPG")


def main():
    assert PHOTO.is_file(), PHOTO
    with gr.Blocks(title="FaceSwap upload probe") as demo:
        components = ui.build_faceswap_ui()
        events.wire_events(components)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    try:
        demo.launch(server_name="127.0.0.1", server_port=port,
                    prevent_thread_lock=True, show_error=True)
        client = Client(f"http://127.0.0.1:{port}", verbose=False)
        result = client.predict([handle_file(str(PHOTO))],
                                api_name="/_safe_on_src_changed")
        assert roop.globals.INPUT_FACESETS, "upload event loaded no source faces"
        assert len(result[0]) == len(roop.globals.INPUT_FACESETS), result
        print(f"[PROBE] upload -> {len(roop.globals.INPUT_FACESETS)} source face(s) -> gallery: OK")
    finally:
        demo.close()


if __name__ == "__main__":
    main()
