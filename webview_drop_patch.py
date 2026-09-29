"""
webview_drop_patch.py - Drag & drop NATIVO de archivos para pywebview (Windows).

Problema: el WebView2 expone los archivos arrastrados desde el Explorador como
File SIN ruta (por seguridad web). Gradio entonces SUBE el archivo entero por
HTTP (copia + hash sha1): 0.3s para 250MB, ~20s para 2GB, minutos para videos
de varios GB = la UI se queda "flasheada" = cuelgue percibido.

Solucion (la ventana es propia/local, el archivo ya esta en el MISMO PC):
  1. AllowExternalDrop=False en el WebView2 (el navegador ya no consume el drop).
  2. AllowDrop=True en el formulario WinForms: captura las RUTAS nativas
     (DataFormats.FileDrop) y las guarda en un API expuesto al JS.
  3. El JS de Face Swap identifica la zona de Origen o Destino y carga las
     rutas en un solo lote, sin subir los archivos al servidor local.

Uso (en el launcher, ANTES de webview.start()):
    import webview_drop_patch
    drop_api = webview_drop_patch.apply_drop_patch()
    window = webview.create_window(..., js_api=drop_api)
"""

import threading
import time


class DropApi:
    def __init__(self):
        self._lock = threading.Lock()
        self._dropped = []
        self._last_drop = None

    def set_dropped(self, paths, x=None, y=None):
        with self._lock:
            if paths:
                paths = [str(p) for p in paths]
                now = time.monotonic()
                if self._last_drop and self._last_drop[0] == paths and now - self._last_drop[1] < 0.25:
                    return
                self._last_drop = (paths, now)
                self._dropped.append({"paths": paths, "x": x, "y": y})

    def get_dropped_files(self):
        """Entrega y consume los drops pendientes, con posición dentro del WebView."""
        with self._lock:
            pending = self._dropped
            self._dropped = []
            return pending


def apply_drop_patch():
    api = DropApi()

    try:
        from webview.platforms import winforms as wf
    except Exception as exc:
        print(f"[DROPPATCH] winforms no disponible: {exc}")
        return api

    if not getattr(wf, 'is_chromium', False):
        print("[DROPPATCH] backend no es EdgeChromium - drop nativo desactivado")
        return api

    BrowserForm = wf.BrowserView.BrowserForm
    orig_form_init = BrowserForm.__init__

    def patched_form_init(self, *args, **kwargs):
        orig_form_init(self, *args, **kwargs)
        try:
            self._pwv_drag_enter = lambda sender, e: _on_drag_enter(e)
            def on_drag_drop(sender, e):
                point = sender.PointToClient(_screen_point(e))
                webview = getattr(self, "webview", None)
                x = int(point.X - webview.Left) if sender is self and webview is not None else int(point.X)
                y = int(point.Y - webview.Top) if sender is self and webview is not None else int(point.Y)
                api.set_dropped(_extract_paths(e), x, y)
            self._pwv_drag_drop = on_drag_drop
            self.AllowDrop = True
            self.DragEnter += self._pwv_drag_enter
            self.DragDrop += self._pwv_drag_drop
            if getattr(self, "webview", None) is not None:
                self.webview.AllowDrop = True
                self.webview.DragEnter += self._pwv_drag_enter
                self.webview.DragDrop += self._pwv_drag_drop
        except Exception as exc:
            print(f"[DROPPATCH] form hook error: {exc}")

    BrowserForm.__init__ = patched_form_init

    try:
        from webview.platforms.edgechromium import EdgeChrome
        orig_edge_init = EdgeChrome.__init__

        def patched_edge_init(self, *args, **kwargs):
            orig_edge_init(self, *args, **kwargs)
            try:
                self.webview.AllowExternalDrop = False
            except Exception as exc:
                print(f"[DROPPATCH] WebView2 drop hook error: {exc}")

        EdgeChrome.__init__ = patched_edge_init
    except Exception as exc:
        print(f"[DROPPATCH] edgechromium hook error: {exc}")

    print("[DROPPATCH] drag & drop nativo ACTIVADO (rutas via window.pywebview.api)")
    return api


def _extract_paths(e):
    try:
        import System.Windows.Forms as WinForms
        if e.Data.GetDataPresent(WinForms.DataFormats.FileDrop):
            data = e.Data.GetData(WinForms.DataFormats.FileDrop)
            if data is not None:
                return [str(x) for x in data]
    except Exception:
        pass
    return []


def _screen_point(e):
    from System.Drawing import Point
    return Point(int(e.X), int(e.Y))


def _on_drag_enter(e):
    try:
        import System.Windows.Forms as WinForms
        if e.Data.GetDataPresent(WinForms.DataFormats.FileDrop):
            e.Effect = WinForms.DragDropEffects.Copy
        else:
            e.Effect = getattr(WinForms.DragDropEffects, 'None')
    except Exception:
        pass
