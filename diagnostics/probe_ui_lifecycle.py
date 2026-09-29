"""Check that reopening AutoAuto reuses a live server and UI cleanup preserves ComfyUI."""

import json
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roop.core import find_existing_autoauto_server


class ConfigHandler(BaseHTTPRequestHandler):
    title = "AutoAuto - AI Editor"

    def do_GET(self):
        body = json.dumps({"title": self.title}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


server = ThreadingHTTPServer(("127.0.0.1", 0), ConfigHandler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
try:
    port = server.server_address[1]
    assert find_existing_autoauto_server(port, 1) == f"http://127.0.0.1:{port}"
    ConfigHandler.title = "Unrelated service"
    assert find_existing_autoauto_server(port, 1) is None
finally:
    server.shutdown()
    server.server_close()

from ui.main import cleanup_comfyui

with patch("ui.tabs.comfy_launcher.stop", side_effect=AssertionError("shared backend stopped")):
    cleanup_comfyui()

import roop.core as core

fake_webview = types.ModuleType("webview")
fake_webview.created_urls = []
fake_webview.create_window = lambda title, url, **kwargs: (
    fake_webview.created_urls.append(url) or SimpleNamespace(uid="probe-window")
)
fake_webview.start = lambda **kwargs: None
fake_drop_patch = types.ModuleType("webview_drop_patch")
fake_drop_patch.apply_drop_patch = lambda: None

class NoopThread:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass

with patch.object(core, "find_existing_autoauto_server", return_value="http://127.0.0.1:7861"), \
     patch.dict(sys.modules, {"webview": fake_webview, "webview_drop_patch": fake_drop_patch}), \
     patch.object(core.threading, "Thread", NoopThread), \
     patch("ui.tabs.comfy_launcher.start", side_effect=AssertionError("second backend started")), \
     patch("ui.tabs.comfy_launcher.stop", side_effect=AssertionError("shared backend stopped")):
    reused_server = core.run()

assert fake_webview.created_urls == ["http://127.0.0.1:7861"]
assert reused_server is True
print("UI lifecycle probe: reuse server, reject unrelated server, preserve backend on close")
