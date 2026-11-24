"""Probe: qué recibe pywebview al reenviar un drop al receptor oculto.

La vía de ``roop/core.py`` intercepta el drop real y lo reenvía a un div oculto
con ``DOMEventHandler``. Este probe reproduce esa cadena sin arrastre manual y
compara tres formas de reenviar el evento:

  A) ``new Event('drop')`` + ``dataTransfer`` definido con ``defineProperty``
     (por defecto NO enumerable: así estaba en v5.1433-v5.1435).
  B) igual que A, pero con ``dataTransfer`` enumerable.
  C) ``new DragEvent('drop')`` con un ``DataTransfer`` real y los mismos File.

El puente de WebView2 se sustituye por un capturador para poder leer el JSON
que pywebview habría enviado a Python (sin tocar archivos reales del disco).
Devuelve exit 0 si C entrega ``dataTransfer.files[].name`` y A no.
"""

import json
import sys
import time

import webview
from webview.dom import DOMEventHandler

RECEIVED = []
POSTED = []


def _on_drop(event):
    data_transfer = event.get("dataTransfer") or {}
    names = []
    if isinstance(data_transfer, dict):
        for item in data_transfer.get("files") or []:
            if isinstance(item, dict):
                names.append(item.get("name"))
    RECEIVED.append({"has_data_transfer": "dataTransfer" in event, "names": names})
    print(f"[PROBE] handler: dataTransfer={'dataTransfer' in event} nombres={names}", flush=True)


HTML = """
<!doctype html>
<html><body>
<div id="sink" style="display:none"></div>
<div id="zone" style="width:300px;height:120px;border:1px dashed #888">zona</div>
</body></html>
"""

HOOK_JS = """
(function(){
  window.__probePosted = [];
  var cw = window.chrome && window.chrome.webview;
  if (!cw) { return 'sin chrome.webview'; }
  cw.postMessageWithAdditionalObjects = function(){ return true; };
  cw.postMessage = function(msg){ window.__probePosted.push(msg); return true; };
  return 'puente capturado';
})();
"""

DISPATCH_JS = """
(function(){
  function fileFor(name){
    return new File([new Uint8Array([1,2,3,4])], name, {type: 'image/jpeg'});
  }
  var sink = document.getElementById('sink');

  var a = new Event('drop', {bubbles: true});
  Object.defineProperty(a, 'dataTransfer', {value: {files: [fileFor('a.jpg')]}});
  sink.dispatchEvent(a);

  var b = new Event('drop', {bubbles: true});
  Object.defineProperty(b, 'dataTransfer', {
    value: {files: [fileFor('b.jpg')], types: ['Files']},
    enumerable: true,
    configurable: true
  });
  sink.dispatchEvent(b);

  var dt = new DataTransfer();
  dt.items.add(fileFor('c.jpg'));
  var c = new DragEvent('drop', {bubbles: true, cancelable: true, dataTransfer: dt});
  sink.dispatchEvent(c);

  return [a.type, b.type, c.type].join(',');
})();
"""


def _summarize_posted(raw):
    try:
        func_name, payload, _ = raw if isinstance(raw, (list, tuple)) else json.loads(raw)
    except Exception as exc:
        return {"raw": str(raw)[:200], "error": str(exc)}
    event = (json.loads(payload) or {}).get("event") or {}
    files = ((event.get("dataTransfer") or {}).get("files")) or []
    return {
        "fn": func_name,
        "has_data_transfer": "dataTransfer" in event,
        "files": [
            {k: f.get(k) for k in ("name", "size", "type")} if isinstance(f, dict) else repr(f)
            for f in files
        ],
    }


def _drive(window):
    try:
        time.sleep(3.0)
        element = window.dom.get_element("#sink")
        if element is None:
            print("[PROBE] FAIL: receptor no encontrado", flush=True)
            window.destroy()
            return
        element.events.drop += DOMEventHandler(_on_drop, prevent_default=True, stop_propagation=True)
        time.sleep(0.3)
        print("[PROBE] hook:", window.evaluate_js(HOOK_JS), flush=True)
        print("[PROBE] dispatch:", window.evaluate_js(DISPATCH_JS), flush=True)
        time.sleep(2.0)
        POSTED.extend(json.loads(window.evaluate_js("JSON.stringify(window.__probePosted || [])") or "[]"))
    except Exception as exc:  # pragma: no cover - diagnóstico
        print(f"[PROBE] error: {exc}", flush=True)
    finally:
        try:
            window.destroy()
        except Exception:
            pass


def main():
    window = webview.create_window("probe-native-drop", html=HTML, width=520, height=320)
    webview.start(_drive, window, debug=False)

    print(f"[PROBE] mensajes al puente: {len(POSTED)}", flush=True)
    summaries = [_summarize_posted(raw) for raw in POSTED]
    for index, summary in enumerate(summaries):
        print(f"[PROBE] {index}: {json.dumps(summary, ensure_ascii=False)}", flush=True)

    names_c = [f.get("name") for s in summaries[-1:] for f in s.get("files", []) if isinstance(f, dict)]
    a_without = not summaries[0].get("has_data_transfer") if summaries else False
    ok = "c.jpg" in names_c and a_without
    print(f"[PROBE] A sin dataTransfer={a_without} · C nombres={names_c}", flush=True)
    print("[PROBE] RESULT:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
