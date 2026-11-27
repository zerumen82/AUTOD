"""Drag real files onto both FaceSwap galleries in the WebView2 window.

Requires the current UI to have AUTOAUTO_DIAG_CDP_PORT=9222. This only loads
photos and detects faces; it does not perform a swap or generate an image.
"""

import json
from pathlib import Path
import re
from types import SimpleNamespace
import time
import urllib.request

from diagnostics.probe_native_drop_cdp_20261123 import CDP


SOURCE = Path(r"D:\PROJECTS\uee\FACES\AATL\grok_image_1773140604875.jpg")
TARGET = Path(r"D:\PROJECTS\uee\FACES\AATL\paloma.jpg")
PORT = 9222


def _page():
    pages = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/list", timeout=5).read())
    return next(p["webSocketDebuggerUrl"] for p in pages if p.get("type") == "page" and "127.0.0.1:786" in p.get("url", ""))


def _snapshot(cdp):
    return json.loads(cdp.evaluate("""JSON.stringify((function(){
      function seen(s){var e=document.querySelector(s);return !!e && e.getBoundingClientRect().height>0;}
      function count(s){var e=document.querySelector(s);return e ? e.querySelectorAll('img').length : 0;}
      function label(s){var e=document.querySelector(s);return e ? (e.innerText||'').slice(0,200) : '';}
      return {origin:count('.origin-gallery'),target:count('.target-gallery'),
        candidate:count('.small-face-gallery'),candidatePanel:seen('.dynamic-panel'),
        sourceInfo:label('.origin-gallery'),targetInfo:label('.target-gallery'),
        sourcePage:label('#faceswap_input_page_info'),
        targetPage:label('#faceswap_target_page_info'),
        fileIndicator:label('.file-indicator'),dropEnabled:!!window.__autoautoDropEnabled};
    })())"""))


def _drag(cdp, selector, path):
    rect = json.loads(cdp.evaluate("""JSON.stringify((function(){
      var e=document.querySelector(%s);if(!e)return null;
      e.scrollIntoView({block:'center'});var r=e.getBoundingClientRect();
      return {x:r.x,y:r.y,w:r.width,h:r.height};
    })())""" % json.dumps(selector)))
    assert rect and rect["w"] > 20 and rect["h"] > 20, (selector, rect)
    x, y = round(rect["x"] + rect["w"] / 2), round(rect["y"] + rect["h"] / 2)
    data = {"items": [], "files": [str(path)], "dragOperationsMask": 1}
    for kind in ("dragEnter", "dragOver", "drop"):
        cdp.call("Input.dispatchDragEvent", type=kind, x=x, y=y, data=data)
        time.sleep(0.4)
    print(f"[CDP] {selector}: drop {path.name} at ({x},{y})")


def _until(cdp, key, timeout=30):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        state = _snapshot(cdp)
        if state[key] > 0:
            return state
        time.sleep(1)
    raise AssertionError(f"{key} no visible after drop: {_snapshot(cdp)}")


def _until_target_loaded(cdp, filename, timeout=30):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        state = _snapshot(cdp)
        if state["candidate"] > 0 and filename in state["fileIndicator"]:
            return state
        time.sleep(1)
    raise AssertionError(f"destination not fully loaded: {_snapshot(cdp)}")


def _face_count(text):
    """Caras totales del paginador (el campo de nombres está oculto)."""
    match = re.search(r"de\s+(\d+)\s+caras", text or "")
    return int(match.group(1)) if match else 0


def _until_source_loaded(cdp, filename, timeout=30):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        state = _snapshot(cdp)
        if state["origin"] > 0 and _face_count(state.get("sourcePage")) > 0:
            return state
        time.sleep(1)
    raise AssertionError(f"source gallery/page info not fully loaded: {_snapshot(cdp)}")


def _until_second_target(cdp, filename, timeout=30):
    """Segundo destino activo con sus candidatas y sin añadirse solo (v5.1463)."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        state = _snapshot(cdp)
        if ("(2/2)" in state["fileIndicator"] and filename in state["fileIndicator"]
                and state["target"] == 1 and state["candidate"] > 0):
            return state
        time.sleep(1)
    raise AssertionError(f"second destination not visible: {_snapshot(cdp)}")


def _resent_batch_repaints(paths):
    """Reenviar el mismo lote debe repintar, no dejar la galería vacía.

    v5.1462: `_LOADED_TARGET_PATHS` vive en el proceso servidor. Al repetir los
    archivos ya cargados, la carga no tiene novedades y antes devolvía salidas
    sin tocar: una ventana nueva contra ese servidor se quedaba sin caras.

    Se ejecuta en un proceso Gradio aislado y se llama al evento directamente:
    empujar la cola compartida desde aquí competiría con el timer de la ventana
    viva, que la vacía antes de que responda otro cliente.
    """
    import gradio as gr
    import roop.globals
    import ui.globals as ui_globals
    from ui.tabs.faceswap import events as faceswap_events, state as faceswap_state
    from ui.tabs.faceswap import ui as faceswap_ui

    with gr.Blocks(analytics_enabled=False) as demo:
        components = faceswap_ui.build_faceswap_ui()
        faceswap_events.wire_events(components)
    poll = next(fn.fn for fn in demo.fns.values()
                if getattr(fn.fn, "__name__", "") == "poll_native_dest_drop")

    click = next(fn.fn for fn in demo.fns.values()
                 if getattr(fn.fn, "__name__", "") == "on_face_selection_click")

    def push():
        queue = roop.globals.NATIVE_DROP_QUEUES.setdefault("faceswap_dest", [])
        queue.extend(str(path) for path in paths)

    push()
    result = poll()
    assert not ui_globals.ui_target_thumbs, (
        "loading the destination must not select a face by itself",
        result,
    )
    assert result[4].get("visible") is True, result[4]
    assert getattr(result[5], "value", None), result[5]
    click(SimpleNamespace(index=0), 1)
    stored = len(ui_globals.ui_target_thumbs)
    assert stored == 1, f"clicking a candidate must add exactly one face: {stored}"

    # Fuera de la ventana de 3 s: la segunda carga no tiene archivos nuevos.
    time.sleep(3.2)
    push()
    result = poll()
    gallery = result[6]
    assert gallery, f"resent batch left the destination gallery empty: {result}"
    expected = min(stored, faceswap_state.FACES_PER_PAGE)
    assert len(gallery) == expected, (len(gallery), expected, result)
    return len(gallery)


def _until_target_count(cdp, count, timeout=30):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        state = _snapshot(cdp)
        if state["target"] == count:
            return state
        time.sleep(1)
    raise AssertionError(f"target count {count} missing: {_snapshot(cdp)}")


def _until_file_count(cdp, count, timeout=30):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        state = _snapshot(cdp)
        if f"/{count})" in state["fileIndicator"]:
            return state
        time.sleep(1)
    raise AssertionError(f"destination count {count} missing: {_snapshot(cdp)}")


def main():
    assert SOURCE.is_file() and TARGET.is_file()
    cdp = CDP(_page())
    cdp.call("Runtime.enable")
    clicked = cdp.evaluate("""(function(){
      var b=Array.from(document.querySelectorAll('button[role=tab]'))
        .find(x=>(x.innerText||'').toLowerCase().replace(/\s+/g,'').includes('faceswap'));
      if(b)b.click();return !!b;
    })()""")
    if not clicked:
        print("[CDP] tabs:", cdp.evaluate("""JSON.stringify(Array.from(document.querySelectorAll('[role=tab]'))
          .map(x=>({tag:x.tagName,text:x.innerText})))"""))
    assert clicked, "FaceSwap tab not found"
    time.sleep(1)
    print("[CDP] initial:", _snapshot(cdp))
    try:
        _drag(cdp, ".origin-gallery", SOURCE)
        source = _until_source_loaded(cdp, SOURCE.name)
        assert _face_count(source["sourcePage"]) > 0, source
        print("[CDP] source visible:", source)
        _drag(cdp, ".target-gallery", TARGET)
        target = _until_target_loaded(cdp, TARGET.name)
        assert target["candidatePanel"], target
        assert TARGET.name in target["fileIndicator"], target
        assert "(1/1)" in target["fileIndicator"], target
        assert target["target"] == 0, (
            "loading a destination must not select a face by itself (v5.1463)",
            target,
        )
        print("[CDP] destination candidates visible:", target)
        clicked_face = cdp.evaluate("""(function(){
          var e=document.querySelector('.small-face-gallery img');
          if(e){e.scrollIntoView({block:'center'});e.click();}
          return !!e;
        })()""")
        assert clicked_face, "candidate face not clickable"
        selected = _until(cdp, "target")
        assert selected["target"] == 1, selected
        print("[CDP] selected destination visible:", selected)
        _drag(cdp, ".target-gallery", SOURCE)
        _until_file_count(cdp, 2)
        second = _until_second_target(cdp, SOURCE.name)
        print("[CDP] second destination loaded without auto-selection:", second)
        clicked_second = cdp.evaluate("""(function(){
          var e=document.querySelector('.small-face-gallery img');
          if(e){e.scrollIntoView({block:'center'});e.click();}
          return !!e;
        })()""")
        assert clicked_second, "second destination candidate not clickable"
        second_selected = _until_target_count(cdp, 2)
        assert "🎯 destino: 1/3" in second_selected["fileIndicator"], second_selected
        print("[CDP] second destination selected:", second_selected)
        repainted = _resent_batch_repaints([TARGET, SOURCE])
        print(f"[CDP] resent batch repaints {repainted} destination face(s) from stored state")
        print("PASS: real WebView2 drop loads source/list, both destinations, and the resent batch repaints")
    finally:
        for title in ("Vaciar origen", "Vaciar destino"):
            cdp.evaluate("""(function(){var b=Array.from(document.querySelectorAll('button'))
              .find(x=>(x.innerText||'').includes(%s));if(b)b.click();return !!b;})()""" % json.dumps(title))
        time.sleep(2)


if __name__ == "__main__":
    main()
