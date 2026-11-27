"""Comprueba en la ventana viva que el listado de origen se refresca.

Requiere la UI arrancada con AUTOAUTO_DIAG_CDP_PORT=9222. Solo carga detección
de caras de origen; no lanza ningún swap ni generación.
"""

import json
from pathlib import Path
import time
import urllib.request

from diagnostics.probe_native_drop_cdp_20261123 import CDP


SOURCE = Path(r"D:\PROJECTS\uee\FACES\AATL\grok_image_1773140604875.jpg")
PORT = 9222


def _page():
    pages = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/list", timeout=5).read())
    return next(p["webSocketDebuggerUrl"] for p in pages if p.get("type") == "page" and "127.0.0.1:786" in p.get("url", ""))


SNAPSHOT = """
(function(){
  function text(s){var e=document.querySelector(s);return e ? (e.innerText||'').trim() : null;}
  function count(s){var e=document.querySelector(s);return e ? e.querySelectorAll('img').length : -1;}
  return JSON.stringify({
    list: text('#faceswap_src_file_list'),
    origin: count('.origin-gallery'),
    page: text('#faceswap_input_page_info')
  });
})()
"""


def _snapshot(cdp):
    return json.loads(cdp.evaluate(SNAPSHOT))


def _wait(cdp, name, predicate, timeout=60):
    end = time.monotonic() + timeout
    last = None
    while time.monotonic() < end:
        last = _snapshot(cdp)
        if predicate(last):
            return last
        time.sleep(1)
    raise AssertionError(f"{name} not reached: {last}")


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
    print(f"[CDP] drop {path.name} on {selector} at ({x},{y})")


def _click(cdp, title):
    return cdp.evaluate("""(function(){var b=Array.from(document.querySelectorAll('button'))
      .find(x=>(x.innerText||'').includes(%s));if(b)b.click();return !!b;})()""" % json.dumps(title))


def main():
    assert SOURCE.is_file(), SOURCE
    cdp = CDP(_page())
    cdp.call("Runtime.enable")
    clicked = cdp.evaluate("""(function(){
      var b=Array.from(document.querySelectorAll('button[role=tab]'))
        .find(x=>(x.innerText||'').toLowerCase().replace(/\\s+/g,'').includes('faceswap'));
      if(b)b.click();return !!b;
    })()""")
    assert clicked, "FaceSwap tab not found"
    time.sleep(1)
    initial = _snapshot(cdp)
    print("[CDP] initial:", initial)
    assert initial["list"] is not None, "el listado de origen no existe en el DOM"
    assert "ninguno cargado" in initial["list"], initial

    _drag(cdp, ".origin-gallery", SOURCE)
    loaded = _wait(cdp, "listado tras soltar", lambda s: s["list"] and SOURCE.name in s["list"])
    assert loaded["origin"] > 0, loaded
    assert "1" in loaded["page"], loaded
    print("[CDP] listado tras soltar:", loaded)

    assert _click(cdp, "Vaciar origen"), "botón Vaciar origen no encontrado"
    cleared = _wait(cdp, "listado tras vaciar",
                    lambda s: s["list"] and "ninguno cargado" in s["list"] and s["origin"] == 0)
    print("[CDP] listado tras vaciar:", cleared)
    print("PASS: el listado de origen aparece al soltar la foto y vuelve a vacío al limpiar")


if __name__ == "__main__":
    main()
