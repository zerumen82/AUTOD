"""Measure real LongCat compliance for a local appearance edit.

    python probe_editor_bald_compliance.py capture [label] [photo] [es] [en]
    python probe_editor_bald_compliance.py run <variant> [label]
    python probe_editor_bald_compliance.py all <label> <variant...>
    python probe_editor_bald_compliance.py real [label] [photo] [es] [en]

`capture` runs the real dispatch over a photo with a stub generator and keeps
the exact window, mask, covered reference and prompt the manager sends. `run`
replays those arguments against the real model and reports how much of the
measured subject the model actually changed. `real` runs the whole manager
against the real model and scores the photo it returns. Set EDITOR_REAL_SEED
to pin the generation seed so two runs are comparable.
"""
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
from roop.img_editor.img_editor_manager import ImgEditorManager


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagnostics/out"
DEFAULT_PHOTO = ROOT / (
    ".autodeep_temp/8706a3e259add0e9fc35384eed4bed7b1d536912735341f5d42fd7e8a3bea72a/"
    "17167729_004_7849.jpg"
)
PROMPT_ES = "el hombre debe ser calvo"
PROMPT_EN = "The man must be bald"
FULL = "LongCat-Image-Edit-Q4_K_S.gguf"
TURBO = "LongCat-Image-Edit-Turbo-Q4_K_S.gguf"

VARIANTS = {
    "turbo": {"version": TURBO, "reference": True, "denoise": None},
    "turbo_no_ref": {"version": TURBO, "reference": False, "denoise": None},
    "turbo_no_ref_d10": {"version": TURBO, "reference": False, "denoise": 1.0},
    "turbo_no_ref_d10_s28": {"version": TURBO, "reference": False, "denoise": 1.0, "steps": 28},
    "turbo_no_ref_d090": {"version": TURBO, "reference": False, "denoise": 0.90},
    "turbo_no_ref_d075": {"version": TURBO, "reference": False, "denoise": 0.75},
    "turbo_no_ref_d085": {"version": TURBO, "reference": False, "denoise": 0.85},
    "turbo_no_ref_d060": {"version": TURBO, "reference": False, "denoise": 0.60},
    "turbo_d10": {"version": TURBO, "reference": True, "denoise": 1.0},
    "full": {"version": FULL, "reference": True, "denoise": None},
    "full_no_ref": {"version": FULL, "reference": False, "denoise": None},
}


def case_dir(label):
    return OUT / label


class StubClient:
    def __init__(self):
        self.kwargs = None

    def load(self, version):
        return True, "OK (stub)"

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(image=kwargs["image"]), "OK (stub)"


def capture(label, photo, prompt_es, prompt_en):
    target = case_dir(label)
    target.mkdir(parents=True, exist_ok=True)
    source = Image.open(photo).convert("RGB")
    print(f"[CAPTURE] caso={label} foto={Path(photo).name} {source.size} prompt={prompt_es!r}")
    stub = StubClient()
    manager = ImgEditorManager()
    with (
        patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
        patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
        patch("roop.img_editor.prompt_translator.translate_prompt", return_value=prompt_en),
        patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client", return_value=stub),
        patch.object(manager, "_release_local_gpu_before_comfy"),
    ):
        output, message, _ = manager.generate_intelligent(
            image=source, prompt=prompt_es, engine="longcat",
        )
    assert output is not None, message
    kwargs = stub.kwargs
    assert kwargs, "el manager no llamó al modelo"
    keep = ("prompt", "negative_prompt", "num_inference_steps", "guidance_scale",
            "denoise", "seed", "keep_source_pixels", "preserve_frame",
            "prompt_authority", "no_kontext", "step_cap")
    meta = {key: kwargs.get(key) for key in keep}
    meta["reference"] = kwargs.get("reference_image") is not None
    meta["message"] = message
    meta["prompt_es"] = prompt_es
    meta["prompt_en"] = prompt_en
    meta["photo"] = str(photo)
    (target / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    kwargs["image"].save(target / "window.png")
    kwargs["mask_image"].convert("L").save(target / "mask.png")
    if kwargs.get("reference_image") is not None:
        kwargs["reference_image"].save(target / "reference.png")
    mask = np.asarray(kwargs["mask_image"].convert("L")) >= 96
    print(f"[CAPTURE] ventana {kwargs['image'].size}, máscara {mask.mean():.2%}, "
          f"reference={meta['reference']}")
    print(f"[CAPTURE] steps={meta['num_inference_steps']} denoise={meta['denoise']} "
          f"cfg={meta['guidance_scale']} keep_source={meta['keep_source_pixels']} "
          f"no_kontext={meta['no_kontext']}")
    print(f"[CAPTURE] mensaje del manager: {message}")


def _score(window, result, mask, reference=None):
    before = np.asarray(window.convert("RGB"), np.int16)
    after = np.asarray(result.convert("RGB").resize(window.size), np.int16)
    delta = np.abs(before - after).max(axis=2)
    lum_before = before[mask].mean()
    lum_after = after[mask].mean()
    print(f"[RUN] cambio dentro del sujeto: {(delta[mask] > 10).mean():.1%} px >10, "
          f"px >40 {(delta[mask] > 40).mean():.1%}")
    print(f"[RUN] luminancia media dentro del sujeto: {lum_before:.1f} -> {lum_after:.1f} "
          f"(Δ{lum_after - lum_before:+.1f})")
    if reference is not None:
        ref = np.asarray(reference.convert("RGB").resize(window.size), np.int16)
        print(f"[RUN] MAE contra la referencia cubierta dentro del sujeto: "
              f"{np.abs(after - ref)[mask].mean():.1f}")


def run(name, label, stop_server=True):
    spec = VARIANTS[name]
    target = case_dir(label)
    window = Image.open(target / "window.png").convert("RGB")
    mask = np.asarray(Image.open(target / "mask.png").convert("L")) >= 96
    meta = json.loads((target / "meta.json").read_text())
    ref_path = target / "reference.png"
    reference = None
    if spec["reference"] and ref_path.exists():
        reference = Image.open(ref_path).convert("RGB")
    denoise = spec["denoise"] if spec["denoise"] is not None else meta["denoise"]
    steps = spec.get("steps") or meta["num_inference_steps"]
    step_cap = spec.get("steps") or meta.get("step_cap")
    print(f"[RUN] caso={label} variante={name} modelo={spec['version']} "
          f"ref={reference is not None} denoise={denoise} steps={steps}")

    from ui.tabs import comfy_launcher
    process = None
    if not comfy_launcher.is_comfyui_running():
        ok, msg, _ = comfy_launcher.start(directly_run=True)
        assert ok, msg
        process = comfy_launcher.comfy_process
    try:
        client = get_flux_edit_comfy_client()
        ok, msg = client.load(flux_version=spec["version"])
        assert ok, msg
        result, msg = client.generate(
            image=window,
            mask_image=Image.open(target / "mask.png").convert("L"),
            reference_image=reference,
            prompt=meta["prompt"],
            negative_prompt=meta.get("negative_prompt") or "",
            num_inference_steps=steps,
            guidance_scale=meta["guidance_scale"],
            denoise=denoise,
            seed=meta.get("seed"),
            keep_source_pixels=bool(meta["keep_source_pixels"]),
            preserve_frame=bool(meta["preserve_frame"]),
            prompt_authority=bool(meta["prompt_authority"]),
            no_kontext=bool(meta["no_kontext"]),
            step_cap=step_cap,
        )
        assert result is not None, msg
        out = result.image if hasattr(result, "image") else result
        out.save(target / f"result_{name}.png")
        ref_score = Image.open(ref_path).convert("RGB") if ref_path.exists() else None
        _score(window, out, mask, ref_score)
        print(f"[RUN] status={msg} -> {target / f'result_{name}.png'}")
    finally:
        if stop_server and process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except Exception:
                process.kill()


def run_all(label, names):
    from ui.tabs import comfy_launcher
    ok, msg, _ = comfy_launcher.start(directly_run=True)
    assert ok, msg
    try:
        for name in names:
            run(name, label, stop_server=False)
    finally:
        process = comfy_launcher.comfy_process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except Exception:
                process.kill()


def real(label, photo, prompt_es, prompt_en, manager=None):
    """Full manager + real model: the photo the user would receive."""
    from ui.tabs import comfy_launcher

    target = case_dir(label)
    target.mkdir(parents=True, exist_ok=True)
    source = Image.open(photo).convert("RGB")
    source.save(target / "source_real.png")
    started = None
    if not comfy_launcher.is_comfyui_running():
        ok, msg, _ = comfy_launcher.start(directly_run=True)
        assert ok, msg
        started = comfy_launcher.comfy_process
    manager = manager if manager is not None else ImgEditorManager()
    seed = os.environ.get("EDITOR_REAL_SEED")
    client = get_flux_edit_comfy_client()
    generate = client.generate
    calls = 0

    def record_generation(**kwargs):
        """Record actual model inputs/output without replacing inference."""
        nonlocal calls
        calls += 1
        call_dir = target / f"call_{calls}_{time.time_ns()}"
        call_dir.mkdir()
        for key, name in (("image", "window"), ("mask_image", "mask"),
                          ("reference_image", "reference")):
            if kwargs.get(key) is not None:
                kwargs[key].save(call_dir / f"{name}.png")
        recorded = ("prompt", "negative_prompt", "num_inference_steps",
                    "guidance_scale", "denoise", "seed", "keep_source_pixels",
                    "preserve_frame", "prompt_authority", "no_kontext", "step_cap")
        (call_dir / "meta.json").write_text(json.dumps(
            {key: kwargs.get(key) for key in recorded}, indent=2, default=str))
        result, message = generate(**kwargs)
        if result is not None:
            sample = result.image if hasattr(result, "image") else result
            sample.save(call_dir / "sample.png")
        print(f"[REAL] inferencia real {calls}: {message} -> {call_dir}", flush=True)
        return result, message

    client.generate = record_generation
    try:
        output, message, _ = manager.generate_intelligent(
            image=source, prompt=prompt_es, engine="longcat",
            seed=int(seed) if seed else None,
        )
    finally:
        client.generate = generate
        process = started or comfy_launcher.comfy_process
        if started is not None and process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except Exception:
                process.kill()
    assert output is not None, message
    out = output.convert("RGB")
    out.save(target / "result_real.png")
    before = np.asarray(source, np.int16)
    after = np.asarray(out.resize(source.size), np.int16)
    delta = np.abs(before - after).max(axis=2)
    changed = delta > 10
    print(f"[REAL] caso={label} prompt={prompt_es!r} tamaño={out.size}")
    print(f"[REAL] mensaje del manager: {message}")
    print(f"[REAL] píxeles cambiados en la foto: {changed.mean():.2%}")
    if changed.any():
        ys, xs = np.where(changed)
        print(f"[REAL] bbox del cambio: ({xs.min()},{ys.min()})-({xs.max()},{ys.max()}) "
              f"= {(xs.max()-xs.min()+1)/source.width:.1%} x {(ys.max()-ys.min()+1)/source.height:.1%} del encuadre")
        box = np.asarray(before, np.int16)
        print(f"[REAL] luminancia dentro del cambio: "
              f"{box[changed].mean(axis=0).mean():.1f} -> "
              f"{np.asarray(after, np.int16)[changed].mean(axis=0).mean():.1f}")
    strip = Image.new("RGB", (source.width * 2 + 12, source.height), (24, 24, 24))
    strip.paste(source, (0, 0))
    strip.paste(out, (source.width + 12, 0))
    strip.save(target / "compare_real.png")
    print(f"[REAL] guardado {target / 'result_real.png'} y compare_real.png")


if __name__ == "__main__":
    args = sys.argv[1:]
    mode = args[0] if args else "capture"
    if mode == "capture":
        capture(
            args[1] if len(args) > 1 else "bald_capture",
            args[2] if len(args) > 2 else DEFAULT_PHOTO,
            args[3] if len(args) > 3 else PROMPT_ES,
            args[4] if len(args) > 4 else PROMPT_EN,
        )
    elif mode == "run":
        run(args[1], args[2] if len(args) > 2 else "bald_capture")
    elif mode == "all":
        run_all(args[1], args[2:] or list(VARIANTS))
    elif mode == "real":
        real(
            args[1] if len(args) > 1 else "bald_capture",
            args[2] if len(args) > 2 else DEFAULT_PHOTO,
            args[3] if len(args) > 3 else PROMPT_ES,
            args[4] if len(args) > 4 else PROMPT_EN,
        )
    else:
        raise SystemExit(f"modo desconocido: {mode}")
