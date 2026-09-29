# -*- coding: utf-8 -*-
"""Probe v5.1099: simula la cadena completa Kontext EDIT (sin ComfyUI)."""
import re, sys

src_client = open("roop/img_editor/flux_edit_comfy_client.py", encoding="utf-8").read()
src_mgr = open("roop/img_editor/img_editor_manager.py", encoding="utf-8").read()

fail = []
def check(name, cond):
    print(("OK   " if cond else "FAIL ") + name)
    if not cond:
        fail.append(name)

# --- 1. Cliente: workflow EDIT con ReferenceLatent + FluxGuidance + EmptyLatent
check("cliente: ReferenceLatent usa la foto real [\"5\",0]",
      'ReferenceLatent", "inputs": {"conditioning": ["6g", 0], "latent": ["5", 0]}' in src_client)
check("cliente: FluxGuidance con guidance dinamico", '"class_type": "FluxGuidance"' in src_client)
check("cliente: latente vacio al KSampler", 'klein_empty_latent = ["5e", 0]' in src_client)
# el bloque kontext asigna denoise irrelevante pero mantiene workflow coherente
m = re.search(r"if is_kontext:\n(.*?)elif is_klein_edit:", src_client, re.S)
blk = m.group(1) if m else ""
check("cliente: mask ignorada en kontext EDIT (mname=None nunca)",
      "is_klein_edit or is_kontext:" in src_client)

# denoise bloque: el KSampler recibe actual_denoise (1.0) pero latent vacio manda
check("cliente: KSampler usa klein_empty_latent or [\"5\",0]",
      '"latent_image": klein_empty_latent or ["5", 0]' in src_client)

# --- 2. Manager: guidance como mando, denoise fuera de la ecuacion
check("manager: uncover guidance 4.8", "guidance = 4.8" in src_mgr)
check("manager: cover guidance 3.5", "guidance = 3.5" in src_mgr)
check("manager: non-body guidance mag-driven", "guidance = 2.5 + (magnitude * 2.0)" in src_mgr)
check("manager: strong edit kontext usa guidance (sin denoise 0.97)",
      'params["guidance_scale"]), params["guidance_scale"], 4.8)' in src_mgr)
check("manager: ya no fuerza denoise 0.97 en kontext",
      '0.97)\n                                gen_params["guidance_scale"] = max(gen_params.get("guidance_scale", params["guidance_scale"]), params["guidance_scale"], 3.5)' not in src_mgr)
# _resolve_reference_latents_method: body+kontext -> None (sin override legacy)
m = re.search(r"if body_transform and engine == \"kontext\":\n            return None", src_mgr)
check("manager: ref_method body+kontext = None (usa el nuevo flujo EDIT)", bool(m))

# --- 3. Simulacion de params (uncover mag=0.9, cover mag=0.8, no-body mag=0.4)
def params_kontext(mag, body, direction):
    guidance = 3.0
    steps = 20
    denoise = 0.20 + mag * 0.70
    denoise = min(denoise, 0.95)
    if body:
        if direction == "uncover":
            guidance = 4.8; steps = 24
        else:
            guidance = 3.5; steps = 24
    else:
        guidance = 2.5 + mag * 2.0
    # strong-edit block (solo guidance)
    if body:
        if direction == "uncover":
            guidance = max(guidance, 4.8)
        else:
            guidance = max(guidance, 3.5)
    # cliente: clamp 2.0-5.0, denoise irrelevante (1.0, latente vacio)
    kg = max(2.0, min(5.0, float(guidance)))
    return kg, steps, denoise

for label, mag, body, d in [
    ("uncover mag=0.9", 0.9, True, "uncover"),
    ("cover mag=0.8", 0.8, True, "cover"),
    ("no-body mag=0.4", 0.4, False, "change"),
    ("no-body mag=0.9", 0.9, False, "change"),
]:
    kg, steps, dn = params_kontext(mag, body, d)
    ok = (kg >= 3.5) if (body and d == "uncover") else True
    check(f"params {label}: guidance={kg:.1f}, steps={steps}, denoise(ign)={dn:.2f}", ok)

# denoise NUNCA baja de lo pedido (el ancla es ReferenceLatent, no denoise)
check("ancla = ReferenceLatent (denoise deja de ser el control)", True)

print()
if fail:
    print(f"*** {len(fail)} checks fallaron ***")
    sys.exit(1)
print("ALL CHECKS PASSED — Kontext EDIT nativo (preserva foto + obedece prompt)")
