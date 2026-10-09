
import webview
import os
import threading
import time
import ctypes

_icon_path = "D:\\PROJECTS\\AUTOAUTO\\assets\\icon.ico"

def _set_window_icon_impl(form, path):
    from System import Action
    import ctypes
    user32 = ctypes.windll.user32
    WM_SETICON = 0x0080
    ICON_SMALL = 0
    ICON_BIG = 1
    GCLP_HICON = -14
    GCLP_HICONSM = -34
    def _set():
        try:
            try:
                setAppId = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID
                setAppId.argtypes = [ctypes.c_wchar_p]
                setAppId.restype = ctypes.c_long
                setAppId("AutoAuto")
            except Exception:
                pass
            from System.Drawing import Icon
            ico = Icon(path)
            form.Icon = ico
            hWnd_ptr = ctypes.c_void_p(form.Handle.ToInt64())
            shell32 = ctypes.windll.shell32
            shell32.ExtractIconExW.argtypes = [ctypes.c_wchar_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
            shell32.ExtractIconExW.restype = ctypes.c_uint
            large = ctypes.c_void_p()
            small = ctypes.c_void_p()
            if shell32.ExtractIconExW(path, 0, ctypes.byref(large), ctypes.byref(small), 1) > 0:
                h32 = ctypes.c_void_p(large.value)
                h16 = ctypes.c_void_p(small.value)
                user32.SendMessageW(hWnd_ptr, WM_SETICON, ICON_BIG, h32)
                user32.SetClassLongPtrW(hWnd_ptr, GCLP_HICON, h32)
                user32.SendMessageW(hWnd_ptr, WM_SETICON, ICON_SMALL, h16)
                user32.SetClassLongPtrW(hWnd_ptr, GCLP_HICONSM, h16)
        except Exception:
            pass
    form.Invoke(Action(_set))

def _set_window_icon(uid):
    try:
        from webview.platforms.winforms import BrowserView
    except Exception:
        return
    if not os.path.exists(_icon_path):
        return
    while uid not in BrowserView.instances:
        time.sleep(0.1)
    form = BrowserView.instances[uid]
    _set_window_icon_impl(form, _icon_path)


def open_sd():
    window = webview.create_window(
        "Stable Diffusion",
        "http://127.0.0.1:9871",
        width=1400,
        height=900,
        resizable=True,
        background_color='#1a1a1a'
    )
    threading.Thread(target=_set_window_icon, args=(window.uid,), daemon=True).start()
    webview.start(debug=False)

if __name__ == "__main__":
    open_sd()
