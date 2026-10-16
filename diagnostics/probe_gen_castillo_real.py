# -*- coding: utf-8 -*-
"""probe_gen_castillo_real.py — Render REAL del prompt que no se respetó.

Reproduce el caso reportado («CASTILLO CON UNICORNIOS, ESTILO DIBUJO» salió
mujer desnuda) con el código nuevo y compara A/B con la MISMA semilla:

  A) inject_nsfw=False (checkbox «Contenido NSFW» desmarcado) → debe salir el
     castillo en estilo dibujo.
  B) inject_nsfw=True  (comportamiento actual, NSFW marcado) → mide si el
     prefijo adulto sigue secuestrando el prompt.

Arranca ComfyUI si no está vivo (log en logs/comfyui_manual.log, proceso
desacoplado de este script) y guarda en diagnostics/out/gen_castillo_style/.

Uso:  python diagnostics/probe_gen_castillo_real.py [--only-off|--only-on]
                       [--engine <motor>]

El motor por defecto es el de la UI (generate_ui.default_engine en
config/generation_prompt.json; hoy flux_dev_q4). El par histórico
castillo_nsfw_off/on.png de este directorio usa flux_dev_abliterated: era el
motor por defecto cuando se escribió el probe (2026-10-04); se mantiene
nombrado aparte para no pisar esa evidencia. Las salidas nuevas llevan el
motor en el nombre (castillo_<motor>_nsfw_off/on.png).
"""
import os
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

OUT_DIR = os.path.join("diagnostics", "out", "gen_castillo_style")
PROMPT = "CASTILLO CON UNICORNIOS, ESTILO DIBUJO"
SEED = 123456
ENGINE = "flux_dev_abliterated"  # lo resuelve main() (--engine o config)
MODS = {
    "image_type": "photoreal",
    "lighting": "natural",
    "framing": "auto",
    "skin_finish": "detailed",
    "color_grade": "auto",
    "use_rewriter": False,
}


def comfy_up() -> bool:
    import requests
    from roop.comfy_workflows import get_comfyui_url
    try:
        return requests.get(f"{get_comfyui_url()}/system_stats", timeout=3).status_code == 200
    except Exception:
        return False


def start_comfy() -> bool:
    if comfy_up():
        print("[Comfy] ya está corriendo")
        return True
    from ui.tabs import comfy_launcher as CL
    exe = CL.get_comfyui_executable()
    if not exe:
        print(f"[Comfy] ERROR: no se encontró main.py en {CL.COMFYUI_DIR}")
        return False
    python_exe = CL.COMFYUI_PYTHON if os.path.exists(CL.COMFYUI_PYTHON) else sys.executable
    cmd = [python_exe, "-u", exe, "--port", str(CL.OUR_PORT), "--normalvram"]
    try:
        os.makedirs(CL.COMFYUI_TEMP_ROOT, exist_ok=True)
        cmd += ["--temp-directory", CL.COMFYUI_TEMP_ROOT]
    except OSError as e:
        print(f"[Comfy] aviso: temp aislado no disponible ({e})")
    cmd.append("--disable-async-offload")

    env = os.environ.copy()
    env.update({
        "COMFYUI_DISABLE_SAFETY_CHECKER": "true",
        "SAFETY_CHECKER_DISABLED": "true",
        "DIFFUSERS_SAFETY_CHECKER_DISABLED": "true",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        "COMFYUI_STARTUP_TIMEOUT": "300",
        "PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync",
        "CUDA_MODULE_LOADING": "LAZY",
        "COMFYUI_AUTO_CACHE": "false",
    })
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.abspath(os.path.join("logs", "comfyui_manual.log"))
    log = open(log_path, "wb")
    print(f"[Comfy] arrancando: {' '.join(cmd)}")
    print(f"[Comfy] cwd={CL.COMFYUI_DIR} log={log_path}")
    subprocess.Popen(cmd, cwd=CL.COMFYUI_DIR, env=env, stdout=log, stderr=subprocess.STDOUT)
    log.close()
    for i in range(150):
        time.sleep(2)
        if comfy_up():
            print(f"[Comfy] listo en {2 * (i + 1)}s")
            return True
    print("[Comfy] TIMEOUT (300s) esperando a ComfyUI")
    return False


def run_one(tag: str, inject_nsfw: bool) -> str:
    from roop.img_editor.flux_gen_comfy_client import get_flux_gen_client
    client = get_flux_gen_client()
    ok, msg = client.load(ENGINE)
    print(f"[{tag}] motor={ENGINE} load: {ok} {msg}")
    res, gmsg = client.generate_ai(
        prompt=PROMPT,
        width=1152,
        height=768,
        prompt_modifiers=dict(MODS),
        enhance_quality=False,
        seed=SEED,
        inject_nsfw=inject_nsfw,
        enhance_tier="hd",
    )
    if res is None:
        print(f"[{tag}] FALLO: {gmsg}")
        return ""
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{tag}.png")
    res.image.save(path)
    print(f"[{tag}] guardado: {path} ({gmsg})")
    return path


def _resolve_engine() -> str:
    args = sys.argv[1:]
    if "--engine" in args:
        index = args.index("--engine")
        if index + 1 < len(args):
            return args[index + 1]
    from roop.img_editor.flux_gen_comfy_client import get_default_generation_engine
    engine = get_default_generation_engine() or "flux_dev_abliterated"
    return engine


def main() -> int:
    global ENGINE
    ENGINE = _resolve_engine()
    only_off = "--only-off" in sys.argv
    only_on = "--only-on" in sys.argv
    print("=" * 70)
    print(f"PROBE REAL — Generate: prompt del castillo (A/B, motor={ENGINE})")
    print("=" * 70)
    if not start_comfy():
        return 1

    failed = False
    if not only_on:
        if not run_one(f"castillo_{ENGINE}_nsfw_off", inject_nsfw=False):
            failed = True
    if not only_off:
        if not run_one(f"castillo_{ENGINE}_nsfw_on", inject_nsfw=True):
            failed = True

    print("\n" + "=" * 70)
    print(f"Salidas en {OUT_DIR}")
    print("=" * 70)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
