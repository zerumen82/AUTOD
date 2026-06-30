#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test structural-global routing + optional ComfyUI e2e."""

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PIL import Image

from roop.img_editor.prompt_translator import translate_prompt
from roop.img_editor.img_editor_manager import get_img_editor_manager
from roop.img_editor.nlp.semantic_analyzer import LightLocalIntentAnalyzer


STRUCTURAL_PROMPTS = [
    ("añade dos personas al fondo", True, None),
    ("quita a la gente de la derecha", True, None),
    ("elimina a la persona del fondo", True, None),
    ("add another person on the left", True, None),
    ("remove the person in the background", True, None),
    ("cambiale la ropa a rojo", True, None),  # global por mag+clothes, no structural wrapper
    ("que este bailando", False, None),
]


def analyze_prompt(prompt: str) -> dict:
    mgr = get_img_editor_manager()
    translated = translate_prompt(prompt)
    nlp = LightLocalIntentAnalyzer()
    axes = nlp.get_axis_scores(translated)
    mag = nlp.get_magnitude(translated)
    target = nlp.detect_target(translated)
    structural = float(axes.get("structural", 0.0))
    high_structural = nlp.is_structural_dominant(translated)
    bias = nlp.get_structural_bias(translated) if hasattr(nlp, "get_structural_bias") else "neutral"
    force_global = high_structural or (
        mag > 0.6 and target in ("subject", "clothes", "face")
    )
    prompt_enhanced = mgr._compose_generation_prompt(
        translated,
        engine="imagine",
        magnitude=mag,
        high_structural=high_structural,
        structural_bias=bias,
    )
    version = (
        "LongCat-Image-Edit-Q4_K_S.gguf"
        if mag >= 0.62
        else "LongCat-Image-Edit-Turbo-Q4_K_S.gguf"
    )
    return {
        "prompt": prompt,
        "translated": translated,
        "structural": structural,
        "mag": mag,
        "target": target,
        "high_structural": high_structural,
        "force_global": force_global,
        "version": version,
        "prompt_enhanced": prompt_enhanced,
        "structural_bias": bias,
    }


def run_analysis_tests() -> bool:
    print("=" * 70)
    print("PHASE 1 — Análisis structural → global (sin ComfyUI)")
    print("=" * 70)
    ok = True
    for prompt, expect_global, expect_target in STRUCTURAL_PROMPTS:
        r = analyze_prompt(prompt)
        global_ok = r["force_global"] == expect_global
        target_ok = expect_target is None or r["target"] == expect_target
        status = "OK" if global_ok and target_ok else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"\n[{status}] {prompt!r}")
        print(
            f"  tr={r['translated']!r} struct={r['structural']:.2f} "
            f"mag={r['mag']:.2f} target={r['target']} bias={r['structural_bias']} "
            f"global={r['force_global']} model={r['version']}"
        )
        pe = r["prompt_enhanced"].lower()
        if r["high_structural"]:
            bias = r["structural_bias"]
            if bias == "remove":
                assert "completely remove" in pe, "missing remove structural wrapper"
            elif bias == "add":
                assert "seamlessly add" in pe, "missing add structural wrapper"
            else:
                assert "scene change" in pe, "missing neutral structural wrapper"
        elif "scene change" in pe or "completely remove" in pe or "seamlessly add" in pe:
            raise AssertionError("structural wrapper applied without dominant structural signal")
    return ok


def run_e2e_test(image_path: str, prompt: str, out_dir: str) -> bool:
    import requests
    from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client

    print("\n" + "=" * 70)
    print("PHASE 2 — Generación e2e (ComfyUI + LongCat)")
    print("=" * 70)

    client = get_flux_edit_comfy_client()
    if not client.is_available():
        print("SKIP: ComfyUI no disponible")
        return False

    img = Image.open(image_path).convert("RGB")
    mgr = get_img_editor_manager()
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    print(f"Prompt: {prompt!r}")
    print(f"Image: {image_path} ({img.size[0]}x{img.size[1]})")

    result, msg, mask = mgr.generate_intelligent(
        image=img,
        prompt=prompt,
        use_rewriter=False,
        use_semantic=True,
        engine="imagine",
        enhance_faces=False,
        auto_upscale=False,
        quality_mode=False,
    )
    elapsed = time.time() - t0

    if result is None:
        print(f"FAIL e2e ({elapsed:.0f}s): {msg}")
        return False

    out_path = os.path.join(out_dir, "structural_e2e_result.png")
    result.save(out_path)
    print(f"OK e2e ({elapsed:.0f}s): {msg}")
    print(f"Saved: {out_path}")
    return True


def main():
    analysis_ok = run_analysis_tests()
    print("\n" + "-" * 70)
    print(f"Phase 1 result: {'PASS' if analysis_ok else 'FAIL'}")

    # Try start ComfyUI if down
    try:
        from ui.tabs.comfy_launcher import start as start_comfy, is_comfyui_running

        if not is_comfyui_running():
            print("\nIniciando ComfyUI...")
            ok, msg, port = start_comfy(port=8188, directly_run=True)
            print(f"ComfyUI start: ok={ok} msg={msg} port={port}")
            for i in range(120):
                if is_comfyui_running():
                    print(f"ComfyUI listo tras {i + 1}s")
                    break
                time.sleep(1)
            else:
                print("ComfyUI no respondió en 120s — e2e omitido")
    except Exception as e:
        print(f"No se pudo iniciar ComfyUI: {e}")

    test_img = os.path.join(ROOT, "testdata", "test_person.jpg")
    if os.path.isfile(test_img):
        e2e_ok = run_e2e_test(
            test_img,
            "quita a la persona de la foto",
            os.path.join(ROOT, "debug_test"),
        )
        print(f"Phase 2 result: {'PASS' if e2e_ok else 'SKIP/FAIL'}")
    else:
        print(f"SKIP e2e: no image at {test_img}")

    sys.exit(0 if analysis_ok else 1)


if __name__ == "__main__":
    main()