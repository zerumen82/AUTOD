"""Serve the actual Editor and exercise real selection callbacks over HTTP.

One production CPU analysis, no generation doubles and no diffusion render.
Separate Gradio clients verify session isolation. The test server always closes.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import gradio as gr
import requests
from gradio_client import Client, handle_file
from gradio_client.exceptions import AppError
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ui.tabs.img_editor import events  # noqa: E402
from ui.tabs.img_editor.ui import build_img_editor_ui  # noqa: E402


def save_image(value, target):
    path = value if isinstance(value, str) else value["path"]
    with Image.open(path) as image:
        image.save(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    source = str(args.source.resolve())
    destination = ROOT / "diagnostics/out/editor_selection_served_20261019" / str(time.time_ns())
    destination.mkdir(parents=True, exist_ok=False)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    with gr.Blocks() as app:
        build = build_img_editor_ui()
        events.wire_events(build)
    app.queue()
    clients = []
    try:
        app.launch(server_name="127.0.0.1", server_port=port, share=False,
                   prevent_thread_lock=True, allowed_paths=[str(ROOT)], quiet=True, show_error=True)
        url = f"http://127.0.0.1:{port}"
        response = requests.get(f"{url}/config", timeout=10)
        response.raise_for_status()
        config = response.json()
        (destination / "served_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        endpoints = {entry["api_name"]: entry for entry in config["dependencies"]}
        for name, count in (("edit", 8), ("edit_options", 9), ("edit_select", 13), ("edit_region", 14)):
            assert len(endpoints[name]["inputs"]) == count, (name, endpoints[name])
        client_a, client_b = Client(url, verbose=False), Client(url, verbose=False)
        clients.extend([client_a, client_b])
        started = time.monotonic()
        overlay, choices, message = client_a.predict(handle_file(source), api_name="/analyze_regions")
        assert overlay is not None and choices.get("choices"), message
        save_image(overlay, destination / "inventory.png")
        ids = [item[1] for item in choices["choices"][:2]]
        selected, selected_message = client_a.predict(ids, handle_file(source), api_name="/select_regions")
        assert selected is not None and str(len(ids)) in selected_message, selected_message
        save_image(selected, destination / "selected.png")
        rejection = ""
        try:
            other_overlay, other_message = client_b.predict(ids, handle_file(source), api_name="/select_regions")
            assert other_overlay is None, "Another session accepted the selection"
        except AppError as error:
            if "list of choices" not in str(error):
                raise
            rejection = str(error)
        other_overlay, other_message = client_b.predict([], handle_file(source), api_name="/select_regions")
        assert other_overlay is None and "anal" in other_message.casefold(), other_message
        kept, kept_message = client_a.predict(ids, handle_file(source), api_name="/select_regions")
        assert kept is not None, kept_message
        # Reusing a result invalidates the old selection in the SAME session.
        client_a.predict(handle_file(source), api_name="/reuse_editor_result")
        stale_overlay, stale_message = client_a.predict([], handle_file(source), api_name="/select_regions")
        assert stale_overlay is None, stale_message
        metrics = {"url": url, "source": source, "elapsed_seconds": time.monotonic() - started,
                   "analysis_message": message, "choices": choices["choices"],
                   "selected_ids": ids, "selected_message": selected_message,
                   "other_session_preprocess_rejection": rejection,
                   "other_session_message": other_message, "reuse_message": stale_message,
                   "scope": "Actual HTTP server + local CPU analysis/list/isolation/reuse; no diffusion render"}
        (destination / "results.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
        print(f"EVIDENCE {destination}", flush=True)
    finally:
        for client in clients:
            close = getattr(client, "close", None)
            if callable(close):
                close()
        app.close()


if __name__ == "__main__":
    main()
