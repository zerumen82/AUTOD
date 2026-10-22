"""Runtime preflight for the local Moondream vision path.

The installed ``llama_cpp`` Llava projector exposes no argument to select the
device it runs on: ``clip_model_load(path, verbosity)`` leaves the choice to the
ggml backend that was loaded into the process.  When that backend is the CPU
build, the projector runs on CPU and the vision path dies with a native
``STATUS_STACK_BUFFER_OVERRUN`` that cannot be caught as a Python exception
(observed 2026-10-19, evidence in
``diagnostics/out/moondream_cpu_control_20261019``).  The text model alone is
safe on CPU.

This module is the single place that decides whether a vision run may start:

* the ggml backend must expose GPU offload (``llama_supports_gpu_offload``);
* the GPU must have enough free VRAM for the projector, which is the only
  part that runs on the GPU (the text model is loaded with ``n_gpu_layers=0``
  and stays in system RAM).

It never loads a model and never runs inference, so it is cheap enough to call
before every vision request.  All thresholds live in the project configuration
instead of being hardcoded here.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

_ROOT = Path(__file__).resolve().parents[2]
_SETTINGS_PATH = _ROOT / "config" / "editor_selection.json"


class VisionUnavailableError(RuntimeError):
    """Raised when the vision path must not run because it would crash.

    The CPU projector path aborts the process with a native error, so this is
    raised before any model is loaded and must reach the caller as a message.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@dataclass
class VisionPreflight:
    """Outcome of a vision preflight check."""

    ok: bool
    reason: str = ""
    gpu_offload: bool = False
    max_devices: int = 0
    free_vram_mb: Optional[int] = None
    total_vram_mb: Optional[int] = None
    required_vram_mb: int = 0
    details: Dict[str, str] = field(default_factory=dict)

    def as_message(self) -> str:
        """Human readable explanation for the UI."""
        if self.ok:
            return "Moondream: proyector en GPU listo"
        return f"Moondream no se puede ejecutar: {self.reason}"


def _load_settings() -> dict:
    try:
        return json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _resolve_path(value: str) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return path if path.is_absolute() else _ROOT / path


def _file_size_mb(path: Path) -> int:
    try:
        return int(path.stat().st_size / (1024 * 1024))
    except OSError:
        return 0


def _nvidia_smi_vram() -> tuple[Optional[int], Optional[int]]:
    """Total and used VRAM in MiB according to nvidia-smi.

    nvidia-smi is the only source that accounts for VRAM held by *other*
    processes.  ``torch.cuda.mem_get_info`` under-reports it: measured on this
    machine, a 6400 MiB external reservation was only visible as 3738 MiB
    (diagnostics/out/vision_guard_busy_gpu_20261020/vram_probe_6.0.json), so
    using it alone would let the guard approve a run that cannot fit.
    """
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        if completed.returncode != 0:
            return None, None
        line = completed.stdout.strip().splitlines()[0]
        used, total = (int(value.strip()) for value in line.split(","))
        return used, total
    except Exception:
        return None, None


def _vram_state() -> tuple[Optional[int], Optional[int]]:
    """Free and total VRAM in MiB, or (None, None) when it cannot be read."""
    used, total = _nvidia_smi_vram()
    if used is not None and total is not None:
        return max(total - used, 0), total
    try:
        import torch

        if not torch.cuda.is_available():
            return None, None
        free, total = torch.cuda.mem_get_info(0)
        return int(free // (1024 * 1024)), int(total // (1024 * 1024))
    except Exception:
        return None, None


def _backend_state() -> tuple[bool, int]:
    """Whether ggml exposes GPU offload and how many devices it can use.

    ``llama_max_devices`` reports the maximum the build supports, not how many
    are present, so it is only used to prove the build knows about GPU
    offloading at all.
    """
    try:
        from llama_cpp import llama_cpp as llama

        return bool(llama.llama_supports_gpu_offload()), int(llama.llama_max_devices())
    except Exception:
        return False, 0


def required_vram_mb(settings: dict) -> int:
    """VRAM the vision path needs, from the configured models and margin.

    Only the projector occupies VRAM: the text model is loaded with
    ``n_gpu_layers=0`` and stays in system RAM.  The projector buffer measured
    by the backend was 867.61 MB for the installed f16 mmproj.
    """
    inventory = settings.get("inventory") or {}
    guard = settings.get("vision_guard") or {}
    projector = _resolve_path(inventory.get("projector_path", ""))
    margin = int(guard.get("vram_margin_mb", 0))
    return _file_size_mb(projector) + margin


def check_vision_preflight(settings: Optional[dict] = None) -> VisionPreflight:
    """Decide whether Moondream may run with the projector on the GPU.

    The check is conservative: when the GPU state cannot be read, or when the
    backend does not expose offload, the vision path is refused instead of
    being allowed to crash the process natively.
    """
    settings = settings if settings is not None else _load_settings()

    gpu_offload, max_devices = _backend_state()
    if not gpu_offload:
        return VisionPreflight(
            ok=False,
            reason=("la compilación de llama_cpp no expone descarga a GPU "
                    "(el proyector se ejecutaría en CPU, ruta que aborta el proceso)"),
            gpu_offload=False, max_devices=max_devices,
        )
    if max_devices < 1:
        return VisionPreflight(
            ok=False,
            reason="no se ha detectado ninguna GPU utilizable por llama_cpp",
            gpu_offload=True, max_devices=max_devices,
        )

    required = required_vram_mb(settings)
    free, total = _vram_state()
    if free is None:
        # The backend claims offload but the GPU cannot be queried.  Refusing
        # here is the safe choice: the CPU fallback is a native crash.
        return VisionPreflight(
            ok=False,
            reason="no se pudo consultar la VRAM disponible",
            gpu_offload=True, max_devices=max_devices, required_vram_mb=required,
        )
    if free < required:
        return VisionPreflight(
            ok=False,
            reason=(f"VRAM insuficiente: {free} MiB libres y se necesitan "
                    f"{required} MiB para el proyector"),
            gpu_offload=True, max_devices=max_devices, free_vram_mb=free,
            total_vram_mb=total, required_vram_mb=required,
        )

    return VisionPreflight(
        ok=True,
        gpu_offload=True, max_devices=max_devices, free_vram_mb=free,
        total_vram_mb=total, required_vram_mb=required,
    )
