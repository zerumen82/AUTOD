"""Cierre de sesión: liberar la GPU y cerrar ComfyUI al cerrar la ventana.

Pedido del usuario (2026-10-20): «cuando se cierra la webview desde la X debe
liberar la gpu siempre y cerrar comfuy». Antes de este cambio
``roop.core.run()`` devolvía nada más cerrar ``webview.start()`` sin tocar
nada: ComfyUI seguía vivo, la VRAM seguía ocupada y la propia salida decía
«ComfyUI permanece disponible para otras sesiones».

La política entera vive en ``config/session_shutdown.json``; este módulo no
contiene parámetros mágicos:

* ``stop_comfyui_on_close`` — detiene ComfyUI **siempre**, aunque este proceso
  no lo haya arrancado (ventana secundaria que reutiliza una sesión abierta).
* ``release_local_gpu_on_close`` — descarga los tenedores locales de GPU.
* ``abort_local_batch_on_close`` — si tras la gracia sigue un batch de FaceSwap
  en marcha, se pide parar con la misma bandera de cancelación documentada
  (``roop.globals.processing``) antes de descargar la VRAM.
* ``queue_grace_seconds`` / ``poll_seconds`` — espera a que cola ComfyUI o batch
  local terminen antes de tocar nada.

La ejecución es idempotente: la llamada explícita que se hace al cerrar la
ventana y la que registra ``atexit`` (Ctrl+C, excepción) no pueden duplicarse.
Cada cierre deja un JSON de evidencia en ``logs/`` —misma convención que
``ui_start_*.log`` y ``process_*.log``— con el estado antes y después. La VRAM
se mide con ``nvidia-smi``, la única fuente que ve la VRAM de otros procesos
(``torch.cuda.mem_get_info`` subinforma, medido en
``diagnostics/out/vision_guard_busy_gpu_20261020/``).
"""
from __future__ import annotations

import atexit
import gc
import importlib
import json
import os
import sys
import threading
import time
from typing import Any, Dict, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SETTINGS_PATH = os.path.join(_PROJECT_ROOT, "config", "session_shutdown.json")
_SESSION_LOG_DIR = os.path.join(_PROJECT_ROOT, "logs")

# Módulos que el cierre necesita y que hay que cargar ANTES de registrar el
# hook: dentro de un callback de ``atexit`` Python rechaza ``atexit.register``
# con «can't register atexit after shutdown» y cualquier import perezoso que
# haga un módulo al cargarse estallaría ahí, dejando ComfyUI vivo. Medido en
# diagnostics/out/session_shutdown_20261020/ (informe atexit, antes de este
# arreglo): ``stopped=false`` con «RuntimeError: can't register atexit…».
# Los cuatro son ligeros: os/sys/subprocess/requests/typing y roop.utils.
_WARM_MODULES = (
    "ui.tabs.comfy_launcher",
    "roop.comfy_workflows",
    "roop.img_editor.vision_guard",
    "ui.tabs.faceswap.state",
)

_lock = threading.Lock()
_done = False
_report: Dict[str, Any] = {}


class _LenientStream:
    """Salida que no puede tumbar el cierre por un carácter no codificable.

    ``comfy_launcher.stop()`` imprime ``⚠`` y ``✔`` y el resumen del cierre
    imprimía ``→``: en una consola Windows con codepage cp1252 esos caracteres
    no se codifican y la ``UnicodeEncodeError`` salía de ``stop()``, con el
    resultado de que ComfyUI seguía vivo. Aquí el texto se reemplaza por
    ``?`` en lugar de romper la limpieza.
    """

    def __init__(self, stream):
        self._stream = stream

    def write(self, data):
        try:
            return self._stream.write(data)
        except UnicodeEncodeError:
            encoding = getattr(self._stream, "encoding", None) or "utf-8"
            safe = data.encode(encoding, "replace").decode(encoding)
            return self._stream.write(safe)

    def flush(self):
        try:
            return self._stream.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _install_lenient_streams() -> Dict[str, Any]:
    saved = {}
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None or isinstance(stream, _LenientStream):
            continue
        saved[name] = stream
        setattr(sys, name, _LenientStream(stream))
    return saved


def _restore_streams(saved: Dict[str, Any]) -> None:
    for name, stream in saved.items():
        try:
            setattr(sys, name, stream)
        except Exception:
            pass


def _warm_imports() -> tuple:
    """Importa ya los módulos del cierre. Devuelve (cargados, fallos)."""
    loaded, failed = [], []
    for name in _WARM_MODULES:
        try:
            importlib.import_module(name)
            loaded.append(name)
        except Exception as exc:
            failed.append(f"{name}: {type(exc).__name__}: {exc}")
    return loaded, failed


class SettingsError(ValueError):
    """La configuración de cierre no es utilizable y no debe adivinarse."""


def load_settings() -> Dict[str, Any]:
    """Lee y valida la política de cierre. Lanza ``SettingsError`` si es inútil."""
    if not os.path.isfile(_SETTINGS_PATH):
        raise SettingsError(f"falta el archivo {_SETTINGS_PATH}")
    try:
        with open(_SETTINGS_PATH, encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception as exc:
        raise SettingsError(f"JSON ilegible en {_SETTINGS_PATH}: {exc}") from exc
    if not isinstance(raw, dict):
        raise SettingsError("la configuración de cierre debe ser un objeto JSON")
    for key in ("stop_comfyui_on_close", "release_local_gpu_on_close",
                "abort_local_batch_on_close"):
        if not isinstance(raw.get(key), bool):
            raise SettingsError(f"'{key}' debe ser true o false")
    grace = raw.get("queue_grace_seconds")
    if isinstance(grace, bool) or not isinstance(grace, (int, float)) or grace < 0:
        raise SettingsError("'queue_grace_seconds' debe ser un número >= 0")
    poll = raw.get("poll_seconds")
    if isinstance(poll, bool) or not isinstance(poll, (int, float)) or poll <= 0:
        raise SettingsError("'poll_seconds' debe ser un número > 0")
    return raw


def _nvidia_vram_mb() -> tuple:
    """(usada, total) en MiB según nvidia-smi; (None, None) si no se puede leer."""
    try:
        from roop.img_editor.vision_guard import _nvidia_smi_vram
        return _nvidia_smi_vram()
    except Exception:
        return None, None


def _comfy_state() -> tuple:
    """(¿ComfyUI vivo?, cola en ejecución, cola pendiente)."""
    try:
        from ui.tabs.comfy_launcher import is_comfyui_running
        running = bool(is_comfyui_running())
    except Exception:
        return False, None, None
    if not running:
        return False, 0, 0
    try:
        import requests
        from roop.comfy_workflows import get_comfyui_url
        payload = requests.get(f"{get_comfyui_url()}/queue", timeout=5).json()
        return (True, len(payload.get("queue_running") or []),
                len(payload.get("queue_pending") or []))
    except Exception:
        return True, None, None


def _local_batch_busy() -> bool:
    """¿Hay un batch de FaceSwap en marcha en este proceso?

    ``state.is_processing`` se pone a True al empezar el batch (logic.py) y a
    False en su bloque ``finally``, así que refleja el trabajo real. En cambio
    ``roop.globals.processing`` es la bandera de cancelación y se queda en True
    después de un batch terminado: mentiría sobre el estado.
    """
    try:
        from ui.tabs.faceswap import state as faceswap_state
        return bool(getattr(faceswap_state, "is_processing", False))
    except Exception:
        return False


def observe_state() -> Dict[str, Any]:
    """Estado observable de la sesión: VRAM, ComfyUI y trabajo local."""
    used, total = _nvidia_vram_mb()
    comfy_running, queue_running, queue_pending = _comfy_state()
    return {
        "wall": time.strftime("%Y-%m-%d %H:%M:%S"),
        "vram_used_mb": used,
        "vram_total_mb": total,
        "comfyui_running": comfy_running,
        "comfyui_queue_running": queue_running,
        "comfyui_queue_pending": queue_pending,
        "local_batch_busy": _local_batch_busy(),
    }


def _is_busy(state: Dict[str, Any]) -> bool:
    """Trabajo en marcha que conviene esperar. Cola desconocida no bloquea."""
    if state.get("local_batch_busy"):
        return True
    if state.get("comfyui_running"):
        return bool(state.get("comfyui_queue_running") or state.get("comfyui_queue_pending"))
    return False


def _release_local_gpu() -> Dict[str, Any]:
    """Descarga los tenedores de GPU ya cargados en este proceso.

    Solo se mira ``sys.modules`` y no se importa nada: un módulo que nunca se
    cargó no puede estar reteniendo VRAM, y dentro de ``atexit`` un import
    nuevo fallaría («can't register atexit after shutdown»).
    """
    steps = []
    errors = []

    try:
        manager_module = sys.modules.get("roop.img_editor.img_editor_manager")
        if manager_module is None:
            steps.append("img_editor_manager_no_cargado")
        elif getattr(manager_module, "_manager", None) is None:
            steps.append("img_editor_manager_ausente")
        else:
            manager_module._manager._release_local_gpu_before_comfy()
            steps.append("img_editor_local_released")
    except Exception as exc:
        errors.append(f"img_editor: {type(exc).__name__}: {exc}")

    try:
        face_util = sys.modules.get("roop.face_util")
        if face_util is None:
            steps.append("face_util_no_cargado")
        else:
            face_util.release_face_analyser()
            steps.append("face_analyser_released")
    except Exception as exc:
        errors.append(f"face_analyser: {type(exc).__name__}: {exc}")

    try:
        torch = sys.modules.get("torch")
        if torch is None:
            steps.append("torch_no_cargado")
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            steps.append("torch_cache_emptied")
        else:
            steps.append("torch_cuda_no_disponible")
    except Exception as exc:
        errors.append(f"torch: {type(exc).__name__}: {exc}")

    try:
        gc.collect()
        steps.append("gc_collected")
    except Exception as exc:
        errors.append(f"gc: {type(exc).__name__}: {exc}")

    return {"attempted": True, "steps": steps, "errors": errors}


def _write_report(report: Dict[str, Any]) -> Optional[str]:
    """Vuelca el informe del cierre a logs/session_shutdown_<ts>.json."""
    try:
        os.makedirs(_SESSION_LOG_DIR, exist_ok=True)
        path = os.path.join(
            _SESSION_LOG_DIR,
            f"session_shutdown_{time.strftime('%Y%m%d_%H%M%S')}.json",
        )
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
        return path
    except Exception:
        return None


def shutdown_session(reason: str = "ventana-cerrada",
                     settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Ejecuta el cierre completo de la sesión. Idempotente: la 2ª vez no repite.

    Nunca lanza excepciones hacia el llamante: cerrar la ventana no puede fallar
    por un problema de limpieza.
    """
    global _done, _report
    with _lock:
        if _done:
            return dict(_report)
        _done = True
    started = time.monotonic()
    report: Dict[str, Any] = {
        "reason": reason,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pid": os.getpid(),
        "ok": False,
    }
    # La limpieza no puede caerse porque un carácter no se pueda codificar en
    # la consola ni porque un import nuevo tueste dentro de atexit.
    saved_streams = _install_lenient_streams()
    try:
        warm_loaded, warm_failed = _warm_imports()
        report["warm_imports"] = {"loaded": warm_loaded, "failed": warm_failed}
        if warm_failed:
            print(f"[SessionShutdown] WARNING imports no disponibles: "
                  f"{'; '.join(warm_failed)}", flush=True)
        if settings is None:
            try:
                settings = load_settings()
            except SettingsError as exc:
                report["error"] = str(exc)
                report["note"] = "configuración inválida: no se ejecuta ninguna acción"
                print(f"[SessionShutdown] ERROR {exc}", flush=True)
                return report
        report["settings"] = settings

        before = observe_state()
        report["before"] = before
        print(
            f"[SessionShutdown] Cierre ({reason}): VRAM {before['vram_used_mb']}/"
            f"{before['vram_total_mb']}MB, ComfyUI={'si' if before['comfyui_running'] else 'no'}"
            f" (cola {before['comfyui_queue_running']}/{before['comfyui_queue_pending']}),"
            f" batch local={'si' if before['local_batch_busy'] else 'no'}",
            flush=True,
        )

        # 1) Estado: comprobar siempre y dar un margen antes de tocar nada.
        grace = float(settings["queue_grace_seconds"])
        poll = float(settings["poll_seconds"])
        grace_start = time.monotonic()
        state = observe_state()
        while _is_busy(state) and (time.monotonic() - grace_start) < grace:
            time.sleep(poll)
            state = observe_state()
        waited = round(time.monotonic() - grace_start, 2)
        idle = not _is_busy(state)
        report["grace"] = {"seconds": grace, "waited_seconds": waited,
                           "idle": idle, "state": state}
        if not idle:
            print(f"[SessionShutdown] Trabajo en marcha tras {waited}s de gracia",
                  flush=True)

        # 2) El trabajo local que siga vivo se pide parar con la bandera
        #    documentada antes de soltar la VRAM que está usando.
        report["local_batch_cancelled"] = False
        if not idle and state.get("local_batch_busy") and settings["abort_local_batch_on_close"]:
            try:
                import roop.globals
                roop.globals.processing = False
                report["local_batch_cancelled"] = True
                print("[SessionShutdown] Batch local en marcha: se pide cancelación "
                      "antes de liberar la GPU", flush=True)
            except Exception as exc:
                print(f"[SessionShutdown] No se pudo cancelar el batch local: {exc}", flush=True)

        # 3) ComfyUI: siempre, aunque este proceso no lo haya arrancado.
        report["comfyui"] = {"attempted": bool(settings["stop_comfyui_on_close"])}
        if settings["stop_comfyui_on_close"]:
            try:
                from ui.tabs.comfy_launcher import stop as stop_comfyui
                stopped, message = stop_comfyui()
            except Exception as exc:
                stopped, message = False, f"{type(exc).__name__}: {exc}"
            report["comfyui"].update(stopped=bool(stopped), message=message)
            print(f"[SessionShutdown] ComfyUI: {message}", flush=True)

        # 4) Tenedores locales de GPU.
        report["local_gpu"] = (
            _release_local_gpu() if settings["release_local_gpu_on_close"]
            else {"attempted": False}
        )

        after = observe_state()
        report["after"] = after
        used_before, used_after = before.get("vram_used_mb"), after.get("vram_used_mb")
        report["vram_delta_mb"] = (
            used_after - used_before
            if isinstance(used_before, int) and isinstance(used_after, int) else None
        )
        comfy_ok = report["comfyui"].get("stopped") or not before.get("comfyui_running")
        local_errors = report.get("local_gpu", {}).get("errors") or []
        report["ok"] = bool(comfy_ok) and not local_errors
        report["seconds"] = round(time.monotonic() - started, 2)
        print(
            f"[SessionShutdown] Listo en {report['seconds']}s: "
            f"ComfyUI detenido={report['comfyui'].get('stopped')}, "
            f"VRAM {used_before}->{used_after}MB (delta {report['vram_delta_mb']}MB), "
            f"ok={report['ok']}",
            flush=True,
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[SessionShutdown] ERROR inesperado: {report['error']}", flush=True)
    finally:
        report.setdefault("seconds", round(time.monotonic() - started, 2))
        report["report_path"] = _write_report(report)
        _restore_streams(saved_streams)
        with _lock:
            _report = dict(report)
    return dict(report)


def register_shutdown_hook(reason: str = "atexit") -> None:
    """Cubre las salidas que no pasan por el cierre de la ventana (Ctrl+C).

    Los módulos del cierre se cargan ANTES de registrar: ``atexit.register``
    falla dentro de un callback de ``atexit`` y un import perezoso ahí dejaría
    ComfyUI vivo.
    """
    loaded, failed = _warm_imports()
    if failed:
        print(f"[SessionShutdown] WARNING imports no precargados: "
              f"{'; '.join(failed)}", flush=True)
    elif loaded:
        print(f"[SessionShutdown] Hook registrado con {len(loaded)} módulos "
              f"precargados", flush=True)
    try:
        atexit.register(shutdown_session, reason)
    except Exception as exc:
        print(f"[SessionShutdown] No se pudo registrar atexit: {exc}", flush=True)


