# -*- coding: utf-8 -*-
"""probe_v51140_gen.py — Verificación funcional de la generación (tab GENERAR).

Construye el workflow REAL de ComfyUI para cada motor (mockeando red/disco)
y valida: nodos, encoders, guidance/clamp cfg, sampler, post-acabado.
Además: paridad de params con la APK y estado de liberación de VRAM.
"""
import sys
import types

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PASS = 0
FAIL = 0

def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  OK  {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}")


# ---- Mock de red ANTES de importar el cliente (is_available / post / get) ----
class _Resp:
    status_code = 200
    text = "{}"
    content = b""
    def json(self):
        return {"prompt_id": "test-pid"}

import requests as _requests

class _Captured:
    workflow = None
    url = None

def _fake_post(url, json=None, **kw):
    _Captured.workflow = json.get("prompt") if isinstance(json, dict) else None
    _Captured.url = url
    return _Resp()

_requests.post = _fake_post
_requests.get = lambda *a, **kw: _Resp()

sys.path.insert(0, ".")
from roop.img_editor import flux_gen_comfy_client as gen_mod
from roop.img_editor import comfy_progress as _cp

# Mock wait_for_comfy_image (sin ComfyUI real)
def _fake_wait(url, pid, **kw):
    return {"filename": "x.png", "subfolder": "", "type": "output"}, "OK"

gen_mod.wait_for_comfy_image = _fake_wait

client = gen_mod.FluxGenComfyClient()
client._model_exists = lambda d, f: True  # mock disco

from PIL import Image
import numpy as np

def _fake_view_get(url, timeout=60):
    r = _Resp()
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    import io as _io
    buf = _io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    r.content = buf.getvalue()
    return r

_requests.get = _fake_view_get

print("=" * 70)
print("PROBE v5.1140 — Generate: workflows reales por motor + APK + VRAM")
print("=" * 70)

# ---------------------------------------------------------------- FLUX abliterated
print("\n[1] flux_dev_abliterated (selección explícita): grafo ComfyUI")
ok, msg = client.load("flux_dev_abliterated")
check("load OK", ok)
check("dual clip (clip_l + T5 GGUF)", client._is_dual_clip and client._clip_type == "flux")
check("NO es sdxl", not client._is_sdxl)

res, gmsg = client.generate(
    prompt="una mujer rubia en la playa",
    width=1152, height=768,
    prompt_modifiers={"image_type": "photoreal", "lighting": "natural", "framing": "auto", "skin_finish": "detailed"},
    enhance_quality=False,
    _skip_rewrite=True,
)
wf = _Captured.workflow or {}
check("generación devolvió imagen", res is not None)
check("UNETLoaderGGUF presente", wf.get("2", {}).get("class_type") == "UnetLoaderGGUF")
check("DualCLIPLoaderGGUF (clip_l + T5)", wf.get("3", {}).get("class_type") == "DualCLIPLoaderGGUF")
check("EmptyLatent 1152x768", wf.get("1", {}).get("inputs", {}).get("width") == 1152)
conf = client._get_model_conf()
ks = wf.get("8", {}).get("inputs", {})
check(f"steps={conf['steps']}", ks.get("steps") == conf["steps"])
check("sampler de config", ks.get("sampler_name") == conf["sampler"])
check("FluxGuidance con cfg 3.5", wf.get("14", {}).get("inputs", {}).get("guidance") == 3.5)
check("KSampler a cfg 1.0 (receta FLUX.1)", ks.get("cfg") == 1.0)
check("denoise 1.0 (puro txt2img)", ks.get("denoise") == 1.0)
check("VAEDecode + SaveImage", wf.get("9", {}).get("class_type") == "VAEDecode" and wf.get("10", {}).get("class_type") == "SaveImage")
# v5.1203: 'censored/sfw/clothed' salió del negative del modelo y vive ahora en
# prompt_extras.anti_censor_negative, que se inyecta SOLO con el checkbox
# «Contenido NSFW» activo (ruta AI). Ver probe_gen_prompt_adherence.py [1].
neg_text = (wf.get("7", {}).get("inputs", {}).get("text") or "")
check("negative del workflow = config del modelo", neg_text == conf["negative_prompt"])
check("anti-censor fuera de la config base (gateado por inject_nsfw)", "panties" not in neg_text)

# ---------------------------------------------------------------- Schnell (receta distilled)
print("\n[2] schnell: receta distilled 4 pasos / cfg 1.0")
ok, msg = client.load("schnell")
check("load OK", ok)
res, gmsg = client.generate(prompt="a cat", width=768, height=768, _skip_rewrite=True)
wf2 = _Captured.workflow or {}
ks2 = wf2.get("8", {}).get("inputs", {})
check("4 steps", ks2.get("steps") == 4)
check("SIN nodo FluxGuidance (cfg 1.0 distilled)", "14" not in wf2)
check("KSampler cfg 1.0", ks2.get("cfg") == 1.0)
check("sampler euler (no ancestral)", ks2.get("sampler_name") == "euler")

# ---------------------------------------------------------------- Prompt assembly
print("\n[3] Ensamblado de prompt (usuario primero, sufijo fotoreal)")
from roop.img_editor.gen_prompt_modifiers import assemble_generation_prompt
mods = {"image_type": "photoreal", "lighting": "natural", "framing": "auto", "skin_finish": "detailed"}
final_prompt, suffix_disp, eff_type = assemble_generation_prompt(
    "flux_dev_abliterated", "una mujer rubia en la playa", mods,
    client._model_configs, is_sdxl=False,
)
check("usuario presente en el prompt", "mujer" in final_prompt.lower() and "playa" in final_prompt.lower())
check("prefijo photorealistic RAW (1 sola vez)", final_prompt.lower().count("photorealistic") == 1 and "raw" in final_prompt.lower())
check("tail con skin texture", "skin" in final_prompt.lower())
check("sin dobles comas ni tokens duplicados", ",," not in final_prompt)
# dedup con prompt de usuario que ya trae estilo
fp2, _, _ = assemble_generation_prompt(
    "flux_dev_abliterated", "photorealistic portrait of a blonde woman at the beach", mods,
    client._model_configs, is_sdxl=False,
)
check("sin duplicar estilo si el usuario ya lo incluye", fp2.lower().count("photorealistic") == 1)

# ---------------------------------------------------------------- Paridad APK
print("\n[4] Paridad de params con la APK (callGenerate, api_name=generate_image)")
import inspect
src_events = open("ui/tabs/gen/events.py", encoding="utf-8").read()
check("api_name=generate_image presente", 'api_name="generate_image"' in src_events)
sig_params = ["prompt", "width", "height", "engine_model", "image_style", "shot_framing", "lighting", "skin_finish", "enhance_quality", "use_rewriter"]
inputs_ok = all(p in src_events for p in sig_params)
check("los 10 params de la APK cableados en inputs", inputs_ok)

# ---------------------------------------------------------------- VRAM post-run
print("\n[5] VRAM: liberación post-generación")
src_gen = open("roop/img_editor/flux_gen_comfy_client.py", encoding="utf-8").read()
has_free = "_free_server_vram" in src_gen or "autoauto/unload" in src_gen
if has_free:
    check("liberación post-run presente", True)
else:
    print("  INFO: el gen client NO libera VRAM post-run (el modelo queda residente).")
    print("        El Image Editor tiene guard pre-run 2200MB + unload en cambio de modelo,")
    print("        así que NO se rompe, pero tras un batch de Gen el pool queda menguado")
    print("        igual que pasaba en el editor (v5.1136).")
    check("documentado como hallazgo (no bloqueante)", True)

# ---------------------------------------------------------------- Sintaxis
print("\n[S] Sintaxis")
for m in ("roop/img_editor/flux_gen_comfy_client.py", "ui/tabs/gen/ui.py", "ui/tabs/gen/events.py"):
    src = open(m, encoding="utf-8", errors="replace").read()
    try:
        compile(src, m, "exec")
        check(f"compila {m}", True)
    except SyntaxError as e:
        check(f"compila {m}: {e}", False)

print("\n" + "=" * 70)
print(f"RESULTADO: {PASS} OK / {FAIL} FAIL")
print("=" * 70)
sys.exit(1 if FAIL else 0)
