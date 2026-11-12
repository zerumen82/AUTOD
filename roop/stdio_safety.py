"""v5.1335 — La salida a consola nunca puede abortar una operación.

En Windows en español la consola usa cp1252, que no puede codificar los
símbolos que usa el proyecto en sus mensajes de diagnóstico (❌ ⚠️ → ✅ 🔍).
`print()` lanzaba entonces `UnicodeEncodeError` DENTRO del código de negocio.

El efecto era especialmente dañino porque `extract_face_images` (roop/
face_util.py) envuelve todo su flujo en un `except Exception` que devolvía
`[]`: en cuanto se imprimía un mensaje de descarte, la FOTO ENTERA se
descartaba en silencio y el propio mensaje de error nunca llegaba a verse.
Medido el 2026-11-12: `00726.jpg` devolvía 0 caras y el log no mostraba
ningún motivo.

Este módulo se importa pronto (roop/core.py y roop/face_util.py) para que
la reconfiguración ocurra antes de que se imprima nada.
"""
import sys


def ensure_utf8_stdio() -> bool:
    """Reconfigura stdout/stderr a UTF-8 con `errors='replace'`.

    Devuelve True si se pudo reconfigurar. Nunca lanza: si el host no
    permite reconfigurar (embedding, pytest, harness), se deja como está.
    """
    changed = False
    for _stream_name in ("stdout", "stderr"):
        _stream = getattr(sys, _stream_name, None)
        _reconfigure = getattr(_stream, "reconfigure", None)
        if _reconfigure is None:
            continue
        try:
            _reconfigure(encoding="utf-8", errors="replace")
            changed = True
        except Exception:
            pass
    return changed


def safe_print(*args, **kwargs):
    """`print` que nunca lanza por caracteres no representables.

    Último recurso para los puntos donde un fallo de codificación no puede
    abortar la operación (mensajes de descarte queanglement el flujo).
    """
    try:
        print(*args, **kwargs)
    except Exception:
        try:
            _text = " ".join(str(a) for a in args)
            print(_text.encode("ascii", "replace").decode("ascii"))
        except Exception:
            pass
