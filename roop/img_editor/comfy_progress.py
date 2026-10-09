#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Polling de progreso ComfyUI (/progress + /history) con ETA y fases."""
import time
from typing import Any, Callable, Dict, Optional, Tuple

import requests


ProgressCallback = Callable[[Dict[str, Any]], None]

_HIST_CACHE: Dict[str, Any] = {"ts": 0.0, "mean": 0.0}


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
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Espera hasta que /history tenga imágenes. Devuelve (img_meta, mensaje)."""
    t0 = time.time()
    last_cb = 0.0
    last_phase = ""
    last_console_phase = ""
    last_log = 0.0
    started_sampling = False

    while True:
        if cancel_check and cancel_check():
            stopped = stop_comfy_prompt(base_url, prompt_id)
            return None, (
                "Cancelado" if stopped else
                "Cancelado; ComfyUI sigue ocupado por esta tarea"
            )

        elapsed = time.time() - t0
        prog = check_comfy_progress(base_url, prompt_id) or {"running": True, "progress": 0}
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
            last_log = now
            last_console_phase = phase_now

        if prog.get("done"):
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
