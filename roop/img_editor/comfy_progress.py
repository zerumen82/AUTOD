#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Polling de progreso ComfyUI (/progress + /history) con ETA y fases.

v5.1205 — este ComfyUI NO expone /progress (404), así que el "paso X/Y" real
solo llega por el WebSocket /ws (mensajes executing/progress). WsProgress
ofrece fases por nodo y avance real del sampler; si falta websocket-client o
el socket cae, todo degrada al polling HTTP anterior sin romper nada.
v5.1206 — el /ws solo recibe executing/execution_* si el POST /prompt lleva
client_id (execution.py:483/718): ComfyUI los rutea a self.sockets[client_id]
y con POST sin client_id solo llegaba "progress" en broadcast (por eso en la
edición real hubo "Sampler 1/8" pero nunca fases ni resumen "Fases:").
Usar new_client_id() en el POST y pasarlo también a wait_for_comfy_image.
"""
import json
import threading
import time
import uuid
from typing import Any, Callable, Dict, Optional, Tuple

import requests

try:
    import websocket as _websocket_client
except Exception:  # pragma: no cover - dependencia opcional
    _websocket_client = None


ProgressCallback = Callable[[Dict[str, Any]], None]

_HIST_CACHE: Dict[str, Any] = {"ts": 0.0, "mean": 0.0}


def new_client_id() -> str:
    """Identificador único para el ruteo de mensajes /ws de ComfyUI.

    El MISMO valor debe ir en el POST /prompt (campo client_id) y en la
    conexión /ws?clientId=... : ComfyUI rutea executing/progress/executed a
    self.sockets[client_id]. Sin ese client_id en el POST, execution.py no
    emite executing (fases por nodo) y solo queda progress en broadcast.
    """
    return f"autoauto-{uuid.uuid4().hex[:16]}"


def history_mean_seconds(base_url: str, prefix: str = "Gen", ttl: float = 60.0) -> float:
    """Duración media (mediana) de corridas previas de Generate.

    Este ComfyUI no expone /progress, así que el ETA se deriva de /history:
    se promedian las duraciones execution_start → execution_success de las
    corridas cuyas salidas llevan el filename_prefix de Generate.
    """
    now = time.time()
    if now - float(_HIST_CACHE.get("ts") or 0) < ttl:
        return float(_HIST_CACHE.get("mean") or 0.0)
    mean = 0.0
    try:
        r = requests.get(f"{base_url}/history", timeout=10)
        if r.status_code == 200:
            durs = []
            for entry in r.json().values():
                outputs = entry.get("outputs") or {}
                if not any(
                    str(img.get("filename", "")).startswith(prefix)
                    for out in outputs.values()
                    for img in (out.get("images") or [])
                ):
                    continue
                start = end = None
                for mtype, data in (entry.get("status") or {}).get("messages") or []:
                    try:
                        ts = float((data or {}).get("timestamp") or 0)
                    except (TypeError, ValueError):
                        continue
                    if ts > 1e12:
                        ts /= 1000.0
                    if mtype == "execution_start":
                        start = ts
                    elif mtype == "execution_success" and ts:
                        end = ts
                if start and end and end > start:
                    durs.append(end - start)
            if durs:
                durs.sort()
                mean = durs[len(durs) // 2]
    except Exception:
        mean = 0.0
    # Solo se cachea un valor válido: con la cola recién iniciada no hay
    # corridas previas y un 0 cacheado bloquearía el ETA durante el TTL.
    if mean > 0:
        _HIST_CACHE["ts"] = now
        _HIST_CACHE["mean"] = mean
    return mean


def format_duration(seconds: float) -> str:
    if seconds is None or seconds < 0:
        return "--:--"
    s = int(seconds)
    if s < 60:
        return f"0:{s:02d}"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}:{s:02d}"
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


def check_comfy_progress(base_url: str, prompt_id: str, timeout: float = 10) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(f"{base_url}/progress", timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            if data.get("running_prompt") == prompt_id:
                return {
                    "running": True,
                    "progress": float(data.get("progress", 0) or 0),
                    "eta": float(data.get("eta_remaining", 0) or 0),
                    "step": int(data.get("current_step", 0) or 0),
                    "total": int(data.get("total_steps", 0) or 0),
                    "node": data.get("node") or "",
                }
        h = requests.get(f"{base_url}/history/{prompt_id}", timeout=timeout)
        if h.status_code == 200 and prompt_id in h.json():
            return {"done": True}
        # Sin /progress, /queue dice si la tarea sigue en cola o corriendo.
        q = requests.get(f"{base_url}/queue", timeout=timeout)
        if q.status_code == 200:
            data = q.json()
            if any(item[1] == prompt_id for item in data.get("queue_running") or []):
                return {"running": True, "progress": 0.0, "step": 0, "total": 0, "eta": 0}
            if any(item[1] == prompt_id for item in data.get("queue_pending") or []):
                return {"running": True, "progress": 0.0, "step": 0, "total": 0, "eta": 0, "queued": True}
    except Exception:
        pass
    return None


def _infer_phase(prog: Dict[str, Any], elapsed: float, steps_hint: int) -> str:
    if prog.get("done"):
        return "Completado"
    if prog.get("queued"):
        return "En cola ComfyUI"
    step = int(prog.get("step") or 0)
    total = int(prog.get("total") or 0) or steps_hint
    pct = float(prog.get("progress") or 0) * 100
    if step > 0 and total > 0:
        if step >= total and pct >= 99:
            return "Decodificando VAE"
        return f"Sampler {step}/{total}"
    if elapsed < 8:
        return "En cola ComfyUI"
    if elapsed < 90 and pct < 1:
        return "Cargando modelo"
    if pct >= 99:
        return "Finalizando"
    return "Preparando generación"


def stop_comfy_prompt(base_url: str, prompt_id: str, timeout: float = 45) -> bool:
    """Stop only this prompt and wait until ComfyUI releases its queue slot."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{base_url}/queue", timeout=5)
            response.raise_for_status()
            queue = response.json()
            running = any(item[1] == prompt_id for item in queue.get("queue_running", []))
            pending = any(item[1] == prompt_id for item in queue.get("queue_pending", []))
            if not running and not pending:
                return True
            if pending:
                requests.post(
                    f"{base_url}/queue", json={"delete": [prompt_id]}, timeout=5
                ).raise_for_status()
            if running:
                # This ComfyUI version accepts prompt_id and will not interrupt
                # a different task if the queue changes between GET and POST.
                requests.post(
                    f"{base_url}/interrupt", json={"prompt_id": prompt_id}, timeout=5
                ).raise_for_status()
        except Exception as exc:
            print(f"[ComfyUI] No se pudo detener {prompt_id}: {exc}", flush=True)
        time.sleep(0.5)
    return False


_WS_LOADER_NODES = {
    "UnetLoaderGGUF", "UNETLoaderGGUF", "CLIPLoader", "CLIPLoaderGGUF",
    "DualCLIPLoader", "DualCLIPLoaderGGUF", "VAELoader", "CheckpointLoaderSimple",
}

_WS_FRIENDLY = {
    "TextEncodeQwenImageEditPlus": "Codificando texto+imagen (Qwen VL)",
    "TextEncodeQwenImageEdit": "Codificando texto+imagen (Qwen VL)",
    "CLIPTextEncode": "Codificando texto",
    "VAEEncode": "Codificando foto (VAE)",
    "VAEDecode": "Decodificando VAE",
    "SaveImage": "Guardando imagen",
    "SetLatentNoiseMask": "Aplicando máscara al latente",
    "FluxKontextMultiReferenceLatentMethod": "Preparando referencia visual",
}


class WsProgress:
    """Progreso real de un prompt vía el WebSocket /ws de ComfyUI.

    Expone paso actual del sampler, fase por nodo y duraciones por nodo.
    Nunca lanza: cada mensaje se procesa bajo try/except y la ausencia de
    websocket-client deja el objeto inerte (connected=False).
    """

    def __init__(self, base_url: str, prompt_id: str, node_names=None, client_id=None):
        # v5.1206 — client_id: ComfyUI solo emite "executing"/"executed" si el
        # prompt se POSTeó con client_id (execution.py:483/718) y las rutea a
        # ESE sid exacto (server.py send_json). Sin client_id en el POST, el
        # /ws solo recibía "progress" (broadcast con sid=None) y no llegaba
        # nunca la fase por nodo ni el resumen "Fases:".
        self.base_url = (base_url or "").rstrip("/")
        self.prompt_id = prompt_id
        self.client_id = client_id or f"autoauto-{prompt_id[:12]}"
        # La URL se construye aquí para que los probes puedan verificar el
        # clientId sin abrir socket.
        _ws_base = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        self.ws_url = f"{_ws_base}/ws?clientId={self.client_id}"
        self.node_names = dict(node_names or {})
        self.step = 0
        self.total_steps = 0
        self.current_node_id = None
        self.done = False
        self.error = ""
        self.connected = False
        self.node_ms: Dict[str, Dict[str, Any]] = {}
        self._node_start: Optional[float] = None
        self._step_ts: list = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._app = None  # WebSocketApp activo (para cerrarlo en stop())
        self._lock = threading.Lock()

    # ---------- ciclo de vida ----------

    def start(self) -> bool:
        if _websocket_client is None:
            return False
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="comfy-ws-progress"
        )
        self._thread.start()
        return True

    def stop(self):
        self._stop.set()
        # v5.1206 — cerrar el socket: run_forever no retorna solo porque
        # _stop esté seteado, y sin esto cada edición filtraba un hilo+socket
        # (acumulación en lotes largos; el join de 2s caducaba y el hilo
        # seguía vivo hasta salir del proceso).
        app = getattr(self, "_app", None)
        if app is not None:
            try:
                app.close()
            except Exception:
                pass
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=2.0)

    def _run(self):
        # v5.1206 — clientId (self.ws_url) debe ser EXACTAMENTE el client_id
        # del POST /prompt: ComfyUI rutea executing/progress a
        # self.sockets[client_id] y ese sid nace de este query param.
        ws_url = self.ws_url
        attempts = 0
        while not self._stop.is_set() and not self.done and attempts < 3:
            attempts += 1
            try:
                app = _websocket_client.WebSocketApp(
                    ws_url,
                    on_open=lambda _ws: setattr(self, "connected", True),
                    on_close=lambda _ws, _c, _m: setattr(self, "connected", False),
                    on_message=lambda _ws, msg: self._on_ws_message(msg),
                    on_error=lambda _ws, _e: None,
                )
                self._app = app
                app.run_forever(ping_interval=20, ping_timeout=10)
            except Exception:
                pass
            if self._stop.is_set() or self.done:
                break
            time.sleep(1.5)

    def _on_ws_message(self, message):
        try:
            if not isinstance(message, str):
                return  # previews binarios (cabecera 0x01/0x02)
            self.handle_payload(json.loads(message))
        except Exception:
            return

    # ---------- datos ----------

    def handle_payload(self, payload: Dict[str, Any]) -> None:
        """Procesa un mensaje JSON del /ws (expuesto para probes)."""
        if not isinstance(payload, dict):
            return
        mtype = payload.get("type")
        data = payload.get("data") or {}
        if not isinstance(data, dict):
            return
        if (
            mtype in (
                "executing", "executed", "progress", "execution_start",
                "execution_success", "execution_error", "execution_interrupted",
            )
            and data.get("prompt_id")
            and data["prompt_id"] != self.prompt_id
        ):
            return
        with self._lock:
            if mtype == "executing":
                node = data.get("node")
                if node:
                    self._close_node_locked()
                    self.current_node_id = str(node)
                    self.node_ms[self.current_node_id] = {
                        "name": self.node_names.get(self.current_node_id, ""),
                        "ms": 0.0,
                    }
                    self._node_start = time.time()
            elif mtype == "progress":
                try:
                    self.step = int(data.get("value") or 0)
                    self.total_steps = int(data.get("max") or 0)
                except (TypeError, ValueError):
                    pass
                if self.step > 0:
                    self._step_ts.append(time.time())
                    if len(self._step_ts) > 64:
                        del self._step_ts[:32]
            elif mtype == "executed":
                # v5.1206 — el nodo terminó: cerrar YA su duración (el
                # "executing" siguiente tarda en llegar y execution_success
                # solo cierra el último; con executed cada nodo queda exacto).
                if data.get("node") is not None and str(data.get("node")) == str(self.current_node_id or ""):
                    self._close_node_locked()
            elif mtype == "execution_success":
                self._close_node_locked()
                self.done = True
            elif mtype == "execution_error":
                self._close_node_locked()
                self.done = True
                self.error = str(
                    data.get("exception_message") or data.get("exception_type") or "error"
                )
            elif mtype == "execution_interrupted":
                self._close_node_locked()
                self.done = True
                self.error = "interrupted"

    def _close_node_locked(self):
        if self._node_start is not None:
            entry = self.node_ms.get(self.current_node_id)
            if entry is not None:
                entry["ms"] = entry.get("ms", 0.0) + (time.time() - self._node_start)
            self._node_start = None

    # ---------- lecturas ----------

    def phase_label(self) -> str:
        if self.error:
            return "Error en ComfyUI"
        nid = self.current_node_id
        if not nid:
            return ""
        name = self.node_names.get(nid) or self.node_ms.get(nid, {}).get("name") or ""
        if name == "KSampler":
            if self.total_steps:
                return f"Muestreando paso {self.step}/{self.total_steps}"
            return "Muestreando"
        if name in _WS_LOADER_NODES:
            return "Cargando modelos"
        return _WS_FRIENDLY.get(name) or name or f"Nodo {nid}"

    def seconds_per_step(self) -> float:
        with self._lock:
            if len(self._step_ts) < 2:
                return 0.0
            span = self._step_ts[-1] - self._step_ts[0]
            n = len(self._step_ts) - 1
        return span / n if n > 0 else 0.0

    def summary(self) -> str:
        """Línea de consola con la duración de cada nodo ejecutado."""
        try:
            with self._lock:
                items = [
                    (nid, (info.get("name") or nid), float(info.get("ms") or 0.0))
                    for nid, info in self.node_ms.items()
                ]
            parts = [
                f"{name} {ms:.0f}s"
                for _nid, name, ms in items
                if ms >= 0.05
            ]
            if not parts:
                return ""
            sps = self.seconds_per_step()
            extra = f" | {sps:.1f}s/paso ×{self.total_steps}" if sps > 0 and self.total_steps else ""
            return "Fases: " + " | ".join(parts) + extra
        except Exception:
            return ""


def wait_for_comfy_image(
    base_url: str,
    prompt_id: str,
    *,
    timeout: float = 1800,
    poll_interval: float = 1.0,
    steps_hint: int = 0,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    stall_timeout: float = 0,
    node_names: Optional[Dict[str, str]] = None,
    client_id: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Espera hasta que /history tenga imágenes. Devuelve (img_meta, mensaje).

    v5.1205 — con node_names se abre WsProgress (WebSocket /ws): fase real por
    nodo, paso real del sampler y ETA por s/paso. Sin websocket o si el socket
    cae, sigue igual que antes vía polling HTTP.
    v5.1206 — client_id: debe ser el MISMO valor enviado en el POST /prompt
    para recibir executing/executed (si no, solo llega progress en broadcast).
    """
    t0 = time.time()
    last_cb = 0.0
    last_phase = ""
    last_console_phase = ""
    started_sampling = False
    ws_progress = WsProgress(base_url, prompt_id, node_names, client_id=client_id)
    ws_progress.start()

    try:
        while True:
            if cancel_check and cancel_check():
                stopped = stop_comfy_prompt(base_url, prompt_id)
                return None, (
                    "Cancelado" if stopped else
                    "Cancelado; ComfyUI sigue ocupado por esta tarea"
                )

            elapsed = time.time() - t0
            prog = check_comfy_progress(base_url, prompt_id) or {"running": True, "progress": 0}
            if ws_progress.step:
                # El WebSocket da el paso real; este ComfyUI no expone /progress.
                prog["step"] = ws_progress.step
                if ws_progress.total_steps:
                    prog["total"] = ws_progress.total_steps
                    prog["progress"] = max(
                        float(prog.get("progress") or 0.0),
                        ws_progress.step / float(ws_progress.total_steps),
                    )
            if int(prog.get("step") or 0) > 0:
                started_sampling = True
            if stall_timeout and not started_sampling and not prog.get("done") and elapsed > stall_timeout:
                stopped = stop_comfy_prompt(base_url, prompt_id)
                return None, (
                    f"Sin muestreo ({int(elapsed)}s)" if stopped else
                    f"Sin muestreo ({int(elapsed)}s); ComfyUI sigue ocupado por esta tarea"
                )
            if elapsed > timeout:
                stopped = stop_comfy_prompt(base_url, prompt_id)
                return None, (
                    f"Timeout ({int(timeout)}s)" if stopped else
                    f"Timeout ({int(timeout)}s); ComfyUI sigue ocupado por esta tarea"
                )
            if prog.get("done"):
                prog["progress"] = 1.0
                prog["phase"] = "Completado"
            else:
                # ETA histórico: /progress no existe en este ComfyUI, se estima
                # con la duración media de corridas previas de Generate.
                if not prog.get("queued"):
                    mean = history_mean_seconds(base_url)
                    if mean > 0:
                        prog["eta"] = max(mean - elapsed, 0.0)
                prog["phase"] = _infer_phase(prog, elapsed, steps_hint)
                prog["elapsed"] = elapsed
                # v5.1205 — el WS manda sobre la estimación: fase por nodo y
                # ETA por s/paso real del sampler.
                ws_phase = ws_progress.phase_label()
                if ws_phase:
                    prog["phase"] = ws_phase
                sps = ws_progress.seconds_per_step()
                if sps > 0 and ws_progress.total_steps and ws_progress.step:
                    prog["eta"] = max(ws_progress.total_steps - ws_progress.step, 0) * sps

            now = time.time()
            if progress_callback and (now - last_cb >= 0.8 or prog["phase"] != last_phase):
                progress_callback(prog)
                last_cb = now
                last_phase = prog.get("phase", "")

            phase_now = prog.get("phase") or "Esperando"
            if phase_now != last_console_phase:
                step = int(prog.get("step") or 0)
                total = int(prog.get("total") or 0) or steps_hint
                pct = float(prog.get("progress") or 0) * 100
                print(
                    f"[ComfyUI] {phase_now} | {pct:.0f}% | paso {step}/{total} | "
                    f"transcurrido {format_duration(elapsed)}"
                )
                last_console_phase = phase_now

            if prog.get("done"):
                ws_summary = ws_progress.summary()
                if ws_summary:
                    print(f"[ComfyUI] {ws_summary}", flush=True)
                if ws_progress.error:
                    print(f"[ComfyUI] WebSocket registró: {ws_progress.error}", flush=True)
                try:
                    h = requests.get(f"{base_url}/history/{prompt_id}", timeout=15)
                    if h.status_code == 200 and prompt_id in h.json():
                        hist = h.json()[prompt_id]
                        for node_out in hist.get("outputs", {}).values():
                            if "images" in node_out:
                                return node_out["images"][0], "OK"
                except Exception as e:
                    return None, str(e)
                return None, "ComfyUI terminó sin imagen"

            time.sleep(poll_interval)
    finally:
        ws_progress.stop()


def build_generation_progress_html(
    *,
    pct: float = 0,
    step: int = 0,
    total_steps: int = 0,
    elapsed: str = "0:00",
    eta: str = "--:--",
    phase: str = "Iniciando",
    detail: str = "",
) -> str:
    safe_pct = max(0, min(100, pct))
    bar = "linear-gradient(90deg, #a855f7, #22d3ee)"
    step_txt = f"{step}/{total_steps}" if total_steps > 0 else "—"
    detail_html = (
        f"<div style='color:#94a3b8;font-size:11px;margin-top:8px;text-align:center;'>{detail}</div>"
        if detail
        else ""
    )
    return f"""
    <div style="background:linear-gradient(135deg,#1e293b 0%,#0f172a 100%);padding:14px;border-radius:10px;
        border:1px solid #334155;margin:8px 0;">
        <div style="color:#a855f7;font-size:14px;font-weight:bold;margin-bottom:10px;text-align:center;">
            {phase}
        </div>
        <div style="color:#64748b;font-size:10px;text-align:center;margin-bottom:8px;">
            No está colgado — LongCat/ComfyUI puede tardar 1–5 min la primera vez
        </div>
        <div style="margin-bottom:10px;background:rgba(255,255,255,0.05);border-radius:8px;height:22px;overflow:hidden;">
            <div style="width:{safe_pct:.1f}%;height:100%;background:{bar};transition:width 0.35s ease-out;
                display:flex;align-items:center;justify-content:center;">
                <span style="color:white;font-size:11px;font-weight:bold;">{safe_pct:.0f}%</span>
            </div>
        </div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;text-align:center;">
            <div><div style="color:#64748b;font-size:10px;">PASO</div>
                <div style="color:#10b981;font-size:16px;font-weight:bold;">{step_txt}</div></div>
            <div><div style="color:#64748b;font-size:10px;">TRANSCURRIDO</div>
                <div style="color:#f59e0b;font-size:15px;font-weight:bold;">{elapsed}</div></div>
            <div><div style="color:#64748b;font-size:10px;">RESTANTE</div>
                <div style="color:#22d3ee;font-size:15px;font-weight:bold;">{eta}</div></div>
            <div><div style="color:#64748b;font-size:10px;">FASE</div>
                <div style="color:#c084fc;font-size:12px;font-weight:bold;">{phase}</div></div>
        </div>
        {detail_html}
    </div>
    """
