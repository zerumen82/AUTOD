"""Matriz de coste y calidad de LTX-2 Q4 en 8 GB (v5.1328).

Ejecuta tres trabajos con el mismo código que usa Animate
(`roop.animate.ltx2_audio.generate`) y mide por trabajo: tiempo, pico de VRAM,
tamaño del MP4 y pista de audio.

  1. 960x544, 49 fotogramas  → coste de la versión corta (~2 s).
  2. 1280x704, 97 fotogramas → 720p real (704 por la regla de múltiplo de 32).
  3. 960x544, 97 fotogramas  → retrato con diálogo y ambiente.

Uso:
    venv/Scripts/python.exe diagnostics/ltx2_matrix_20261110.py \
        --image generate/gen_1791137652.png --portrait <retrato.png>
"""
import argparse
import json
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))

from roop.animate import ltx2_audio as ltx  # noqa: E402
from real_animate_ltx2_audio_20261110 import (  # noqa: E402
    ResourceSampler, _comfy_pid, probe_media,
)

OUT_DIR = os.path.join(ROOT, "diagnostics", "out", "ltx2_matrix_20261110")

PROMPT_SCENE = (
    "A red sports car idles at a city intersection at night. Its engine revs once, "
    "the exhaust note echoing between the buildings; a distant tram bell rings and "
    "tires hiss on the asphalt. The camera pushes in slowly toward the rear of the car "
    "while the brake lights glow brighter and reflections slide across the glossy red "
    "paint. Streetlights flicker, a taxi passes from left to right, and the car edges "
    "forward a few centimetres."
)

PROMPT_PORTRAIT = (
    "The woman looks straight into the camera, smiles and says: \"Hola, ya estoy aquí.\" "
    "Her hair moves slightly with the breeze and she blinks naturally. The camera stays "
    "at eye level with a slow, gentle push-in. Room tone is quiet, with soft wind and a "
    "distant bird."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--portrait", default="", help="retrato real; sin él se omiten los trabajos de persona")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--only", default="", help="nombres de trabajo separados por coma")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    jobs = [
        ("49f_540p", args.image, PROMPT_SCENE, 960, 544, 49),
        ("97f_720p", args.image, PROMPT_SCENE, 1280, 704, 97),
    ]
    if args.portrait:
        jobs.append(("97f_540p_retrato", args.portrait, PROMPT_PORTRAIT, 960, 544, 97))
    if args.only:
        wanted = {name.strip() for name in args.only.split(",")}
        jobs = [job for job in jobs if job[0] in wanted]

    from ui.tabs.comfy_launcher import is_comfyui_running, start as start_comfyui
    started = False
    if not is_comfyui_running():
        print("[Matriz] ComfyUI cerrado; arrancando")
        ok, msg, _port = start_comfyui(port=8188, directly_run=True)
        started = True
        print(f"[Matriz] Arranque: {ok} {msg}")
        if not ok:
            return 3

    results = []
    try:
        for name, image, prompt, width, height, frames in jobs:
            print(f"\n[Matriz] === {name}: {width}x{height}, {frames} frames ===", flush=True)
            sampler = ResourceSampler(_comfy_pid())
            sampler.start()
            started_at = time.time()
            path, message = ltx.generate(
                image, prompt, os.path.join(OUT_DIR, f"{name}.mp4"),
                width=width, height=height, frames=frames, fps=25,
                seed=1234, timeout=args.timeout, progress=lambda m: print(f"  {m}", flush=True),
            )
            elapsed = time.time() - started_at
            sampler.finish()
            sampler.join(timeout=5)
            media = probe_media(path) if path else {}
            row = {
                "job": name, "ok": bool(path), "message": message,
                "width": width, "height": height, "frames": ltx.fit_frames(frames),
                "seconds": round(elapsed, 1),
                "peak_gpu_mb": sampler.peak_gpu,
                "peak_rss_gb": round(sampler.peak_rss / 1e9, 1),
                "bytes": media.get("bytes", 0),
                "duration_s": media.get("duration_s", 0),
                "has_audio": media.get("has_audio", False),
                "audio": media.get("audio", []),
            }
            print(f"[Matriz] {name}: {row['seconds']}s, pico {row['peak_gpu_mb']} MB, "
                  f"{row['bytes'] / 1e6:.2f} MB, audio={row['has_audio']}", flush=True)
            results.append(row)
            if path:
                dest = os.path.join(OUT_DIR, os.path.basename(path))
                # `generate` ya escribe en OUT_DIR: copiar sobre sí mismo falla
                # con SameFileError (v5.1328). Solo copiar si la ruta difiere.
                if os.path.abspath(path) != os.path.abspath(dest):
                    shutil.copy2(path, dest)
    finally:
        # Conservar trabajos anteriores: cada ejecución sustituye solo su fila,
        # no borra las demás (antes la corrida 720p perdió la fila de 49f).
        matrix_path = os.path.join(OUT_DIR, "matrix.json")
        merged = {}
        if os.path.exists(matrix_path):
            try:
                with open(matrix_path, "r", encoding="utf-8") as fh:
                    for row in json.load(fh):
                        if isinstance(row, dict) and row.get("job"):
                            merged[row["job"]] = row
            except Exception:
                merged = {}
        for row in results:
            if row.get("job"):
                merged[row["job"]] = row
        with open(matrix_path, "w", encoding="utf-8") as fh:
            json.dump(list(merged.values()), fh, indent=2)
        if started:
            try:
                from ui.tabs.comfy_launcher import stop as stop_comfyui
                stop_comfyui()
            except Exception as e:
                print(f"[Matriz] No se pudo cerrar ComfyUI: {e}")

    print("\n[Matriz] resumen:", json.dumps(results, indent=2))
    return 0 if all(r["ok"] and r["has_audio"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
