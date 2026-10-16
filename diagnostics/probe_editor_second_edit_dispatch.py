"""Contratos del dispatch de ediciones consecutivas en el Image Editor.

Regresiones reales cubiertas (incidente 2026-10-15: 1ª edición salió y la 2ª
"no arrancó" sin rastro alguno):
1. Una excepción temprana en on_generate (p.ej. Image.open de una ruta
   temporal borrada) dejaba state.is_generating clavado en True: cada edición
   posterior moría en silencio con "ya hay una transformacion en proceso".
2. Dos ediciones consecutivas deben arrancar siempre: el guard se libera al
   terminar la primera y el rechazo (si lo hay) informa del tiempo
   transcurrido en vez de devolver un "no" mudo.
3. El clic debe dejar rastro inmediato en consola (print de entrada) y la
   barra de progreso de la UI debe recibir las fases del manager sin que un
   fallo en el mapeo pueda romper la edición.

CPU-only: el manager se sustituye por un stub; no toca ComfyUI ni GPU.
"""
import inspect
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402
from PIL import Image  # noqa: E402

from ui.tabs.img_editor import events, state  # noqa: E402

PASS = []


def _check(name, ok, extra=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {extra}" if extra else ""),
          flush=True)
    PASS.append(bool(ok))


class _FakeManager:
    """Sustituye al manager real: devuelve una imagen ya y registra la llamada."""

    def __init__(self):
        self.calls = []
        self.releases = 0
        self._last_instruction_compliance = {}

    def generate_intelligent(self, **kwargs):
        self.calls.append(kwargs)
        cb = kwargs.get("progress_callback")
        if cb:
            cb({"phase": "Codificando", "progress": 0.5, "step": 3, "total": 12})
            cb({"phase": "Mejorando caras", "progress": 0.94, "detail": "CodeFormer"})
            cb({"progress": 2.0})  # fuera de rango: debe acotarse sin reventar
            cb({"progress": "corrupto"})  # tipo inválido: no debe romper
        return Image.new("RGB", (32, 32), (10, 10, 10)), "OK stub", None

    def _release_local_gpu_before_comfy(self):
        self.releases += 1


def _run(fake, img, prompt="haz algo"):
    return events.on_generate(img, prompt, "longcat", 0.0, False, 1, False, "hd")


def main():
    # Los PNG de prueba no deben acabar en el directorio de ediciones real.
    tmp_out = Path(__file__).resolve().parent / "out"
    tmp_out.mkdir(exist_ok=True)
    events.get_img_editor_output_dir = lambda: str(tmp_out)
    fake = _FakeManager()
    events.get_img_editor_manager = lambda: fake
    img = Image.new("RGB", (64, 64), (120, 30, 30))

    # --- 1) Excepción temprana NO clava el guard --------------------------
    r, msg, _ = events.on_generate(
        str(tmp_out / "ruta_temporal_inexistente.png"),
        "hola", "longcat", 0.0, False, 1, False, "hd")
    _check("excepción temprana devuelve Error", r is None and "Error" in str(msg),
           str(msg)[:60])
    _check("guard liberado tras excepción temprana", state.is_generating is False)
    _check("timestamp liberado tras excepción", state.generation_started_at is None)
    _check("stub NO invocado con ruta rota", len(fake.calls) == 0)

    # --- 2) Dos ediciones consecutivas arrancan ---------------------------
    r1, m1, _ = _run(fake, img)
    _check("edición 1 devuelve imagen", r1 is not None, str(m1)[:40])
    _check("guard liberado tras edición 1", state.is_generating is False)
    _check("timestamp liberado tras edición 1", state.generation_started_at is None)

    r2, m2, _ = _run(fake, img, prompt="segunda")
    _check("edición 2 arranca (no rechazada)",
           r2 is not None and "transformacion" not in str(m2), str(m2)[:40])
    _check("ambas llegaron al manager", len(fake.calls) == 2)
    _check("progress_callback conectado en ambas",
           all(c.get("progress_callback") for c in fake.calls))
    _check("release post-petición ejecutado 2 veces", fake.releases >= 2,
           f"releases={fake.releases}")

    # --- 3) Guard rechaza con aviso (no en silencio) ----------------------
    state.is_generating = True
    state.generation_started_at = time.time() - 7
    r3, m3, _ = _run(fake, img, prompt="tercera")
    _check("edición concurrente rechazada",
           r3 is None and "transformacion" in str(m3), str(m3)[:60])
    _check("rechazo informa del tiempo transcurrido", "(lleva" in str(m3),
           str(m3)[:60])
    _check("stub NO invocado en rechazo", len(fake.calls) == 2)
    state.is_generating = False
    state.generation_started_at = None

    # --- 4) Firma expone progress=gr.Progress (Gradio lo inyecta) ---------
    param = inspect.signature(events.on_generate).parameters.get("progress")
    _check("parámetro progress con default gr.Progress",
           param is not None and isinstance(param.default, gr.Progress))

    ok = all(PASS)
    print(f"[PROBE] {'PASS' if ok else 'FAIL'} {sum(PASS)}/{len(PASS)}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
