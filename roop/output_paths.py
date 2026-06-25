#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rutas de salida del proyecto (carpetas en raíz, fuera de output/)."""

import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMG_EDITOR_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "img_editor")
GENERATION_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "generate")
ANIMATE_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "animate")


def get_img_editor_output_dir() -> str:
    os.makedirs(IMG_EDITOR_OUTPUT_DIR, exist_ok=True)
    return IMG_EDITOR_OUTPUT_DIR


def get_generation_output_dir() -> str:
    os.makedirs(GENERATION_OUTPUT_DIR, exist_ok=True)
    return GENERATION_OUTPUT_DIR


def get_animate_output_dir() -> str:
    os.makedirs(ANIMATE_OUTPUT_DIR, exist_ok=True)
    return ANIMATE_OUTPUT_DIR