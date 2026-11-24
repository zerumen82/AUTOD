"""Probe: lanza un drop real de archivo sobre la ventana WebView2 (CDP).

Requiere la app arrancada con ``AUTOAUTO_DIAG_CDP_PORT=9222``. Comprueba la
zona de arrastre de «Mejorar fotos» y despacha un drop con un archivo real del
disco para ver si pywebview entrega la ruta (y no un evento vacío).

Uso:
    python diagnostics/probe_native_drop_cdp_20261123.py [ruta-de-una-foto]
"""

import json
import os
import re
import sys
import time
import urllib.request

import websocket

PORT = 9222
DEFAULT_FILE = r"D:\PROJECTS\AUTOAUTO\enhance\window.png"
TARGET = os.environ.get("PROBE_DROP_TARGET", "#enhance_input_files")


class CDP:
    def __init__(self, ws_url):
        self.ws = websocket.create_connection(ws_url, timeout=15)
        self.next_id = 0

    def call(self, method, **params):
        self.next_id += 1
        self.ws.send(json.dumps({"id": self.next_id, "method": method, "params": params}))
        while True:
            message = json.loads(self.ws.recv())
            if message.get("id") == self.next_id:
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                return message.get("result", {})

    def evaluate(self, expression):
        result = self.call("Runtime.evaluate", expression=expression, returnByValue=True)
        return (result.get("result") or {}).get("value")


def _page_ws():
    for _ in range(30):
        try:
            pages = json.loads(
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/list", timeout=5).read()
            )
            for page in pages:
                if page.get("type") == "page" and "7861" in (page.get("url") or ""):
                    return page["webSocketDebuggerUrl"]
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("no se encontró la página de AutoAuto en CDP")


INFO = """
(function(){
  var zone = document.querySelector(__SEL__);
  var queue = document.getElementById('enhance_input_files');
  var rect = zone ? zone.getBoundingClientRect() : null;
  return JSON.stringify({
    zone: !!zone,
    tag: zone ? zone.tagName : null,
    rect: rect ? {x: Math.round(rect.x), y: Math.round(rect.y), w: Math.round(rect.width), h: Math.round(rect.height)} : null,
    enabled: !!window.__autoautoDropEnabled,
    sink: !!document.getElementById('autoauto_drop_enhance'),
    text: queue ? (queue.innerText || '').trim().slice(0, 160) : null
  });
})()
"""


def main():
    paths = sys.argv[1:] or [DEFAULT_FILE]
    cdp = CDP(_page_ws())
    cdp.call("Runtime.enable")
    info_js = INFO.replace("__SEL__", repr(TARGET))
    # Gradio monta solo la pestaña activa: hay que abrir «Mejorar fotos».
    cdp.evaluate(
        "(function(){var b=Array.prototype.find.call("
        "document.querySelectorAll('button[role=tab]'),function(x){"
        "return (x.innerText||'').indexOf('Mejorar fotos')>=0;});"
        "if(b){b.click();}return !!b;})()"
    )
    time.sleep(2.0)
    cdp.evaluate(
        "(function(){var z=document.querySelector(" + repr(TARGET) + ");"
        "if(z){z.scrollIntoView({block:'center'});}return true;})()"
    )
    time.sleep(0.6)

    info = json.loads(cdp.evaluate(info_js))
    print("[CDP] zona:", json.dumps(info, ensure_ascii=False))
    if not info["rect"]:
        print("[CDP] RESULT: FAIL sin zona")
        return 1

    x = info["rect"]["x"] + info["rect"]["w"] // 2
    y = info["rect"]["y"] + info["rect"]["h"] // 2
    data = {"items": [], "files": paths, "dragOperationsMask": 1}
    for kind in ("dragEnter", "dragOver", "drop"):
        cdp.call("Input.dispatchDragEvent", type=kind, x=x, y=y, data=data)
        time.sleep(0.4)
    print(f"[CDP] drop despachado en ({x},{y}) con {len(paths)} archivo(s)")

    time.sleep(4)
    info_after = json.loads(cdp.evaluate(info_js))
    print("[CDP] tras el drop:", json.dumps(info_after, ensure_ascii=False))
    def _queue_count(text):
        match = re.search(r"(\d+)\s+foto\(s\) en cola", text or "")
        return int(match.group(1)) if match else 0

    before = _queue_count(info.get("text"))
    after = _queue_count(info_after.get("text"))
    ok = after == len(paths)
    print(f"[CDP] cola antes={before} después={after} (esperado {len(paths)})")
    print("[CDP] RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
