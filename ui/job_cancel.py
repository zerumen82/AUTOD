#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cancelación cooperativa de trabajos largos en la UI."""

from __future__ import annotations

import threading
from typing import Optional

import requests

SCOPE_IMG_EDITOR = "img_editor"
SCOPE_GENERATION = "generation"
SCOPE_ANIMATE = "animate"
SCOPE_FACESWAP = "faceswap"

_lock = threading.Lock()
_flags = {k: False for k in (
    SCOPE_IMG_EDITOR, SCOPE_GENERATION, SCOPE_ANIMATE, SCOPE_FACESWAP,
)}


def clear(scope: str) -> None:
    with _lock:
        _flags[scope] = False


def request(scope: str, *, interrupt_comfy: bool = False) -> None:
    with _lock:
        _flags[scope] = True
    if interrupt_comfy:
        interrupt_comfyui()


def is_cancelled(scope: str) -> bool:
    with _lock:
        return bool(_flags.get(scope))


def interrupt_comfyui() -> None:
    try:
        from roop.comfy_workflows import get_comfyui_url
        base = get_comfyui_url()
        requests.post(f"{base}/interrupt", timeout=3)
    except Exception:
        pass


def cancel_status_html(message: str = "Operación cancelada") -> str:
    return (
        f"<div style='text-align:center;color:#f59e0b;padding:10px;font-weight:bold;'>"
        f"⏹ {message}</div>"
    )


def btn_running():
    import gradio as gr
    return gr.update(interactive=False), gr.update(interactive=True)


def btn_idle():
    import gradio as gr
    return gr.update(interactive=True), gr.update(interactive=False)