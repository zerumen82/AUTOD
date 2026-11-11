# -*- coding: utf-8 -*-
"""Verificación real del fix de la tab Animate Image: construir la UI completa.

El crash reportado ocurría al salir del `with gr.Blocks(...)` (get_config_file),
cuando Gradio serializa las dependencias y encuentra una lambda en `inputs`
('function' object has no attribute '_id'). Si este script termina 0, la UI
construye y serializa bien en el mismo punto donde fallaba.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import ui.main as m

demo = m.create_ui()
cfg = demo.get_config_file() if hasattr(demo, "get_config_file") else demo.config
n_deps = len(cfg.get("dependencies", [])) if isinstance(cfg, dict) else -1
print(f"UI CONSTRUIDA OK — {n_deps} dependencias serializadas")
sys.exit(0)
