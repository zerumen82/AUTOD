# -*- coding: utf-8 -*-
"""probe_gen_prompt_adherence.py — Generate respeta el prompt (H-GEN-001).

Reproduce el fallo reportado («CASTILLO CON UNICORNIOS, ESTILO DIBUJO» salió
mujer desnuda) a nivel de ensamblado de prompt y valida los dos gates nuevos:

  [1] Gate NSFW (checkbox «Contenido NSFW»):
        OFF → sin prefijo adulto, sin anti-censor, sin LoRA adulta.
        ON  → comportamiento actual intacto (prefijo nsfw + anti-censor).
  [2] Gate de estilo: si el prompt pide un estilo no fotorreal, caen el tipo
      de imagen fotorreal, el suffix de piel de la UI y photoreal_tail.
  [3] Control fotorreal: un prompt que NO pide otro estilo conserva la cola.
  [4] LoRA del catálogo: entradas "nsfw": true fuera con el gate apagado.
  [5] Contrato APK (10 posicionales en api_name=generate_image) intacto.

Sin red ni GPU: no renderiza (el render real va aparte, en diagnostics/out/).
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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


sys.path.insert(0, ".")
from roop.img_editor.flux_gen_comfy_client import FluxGenComfyClient
from roop.img_editor.gen_prompt_modifiers import assemble_generation_prompt, get_effective_negative, get_effective_prefix
from roop.img_editor.gen_semantic import prompt_requests_nonphotoreal_style, score_style_gate

print("=" * 70)
print("PROBE — Generate: adherencia al prompt (gate NSFW + gate de estilo)")
print("=" * 70)

client = FluxGenComfyClient()
client._model_exists = lambda d, f: True
ok, msg = client.load("flux_dev_abliterated")
check(f"load flux_dev_abliterated: {msg}", ok)
check("motor no-SDXL (FLUX)", not client._is_sdxl)

MODS = {
    "image_type": "photoreal",
    "lighting": "natural",
    "framing": "auto",
    "skin_finish": "detailed",
    "color_grade": "auto",
    "use_rewriter": False,
}
ADULT_POS = ("nsfw", "uncensored adult content")
ADULT_NEG = ("censored", "sfw", "clothed", "panties")
PHOTOREAL = ("photorealistic raw photo", "film grain", "dslr", "depth of field", "skin pores")

# ---------------------------------------------------------------- [1] Gate NSFW
print("\n[1] Gate NSFW con el prompt real del reporte")
castle_off, p_off = client._prepare_prompt_intelligent(
    "CASTILLO CON UNICORNIOS, ESTILO DIBUJO", prompt_modifiers=MODS, inject_nsfw=False,
)
castle_on, p_on = client._prepare_prompt_intelligent(
    "CASTILLO CON UNICORNIOS, ESTILO DIBUJO", prompt_modifiers=MODS, inject_nsfw=True,
)
print(f"    NSFW OFF → {castle_off}")
print(f"    NSFW ON  → {castle_on}")

check("OFF: sin prefijo adulto en positivo",
      not any(t in castle_off.lower() for t in ADULT_POS))
check("OFF: sin bloques anti-censor en negativo",
      not any(t in p_off["neg"].lower() for t in ADULT_NEG))
check("ON: prefijo adulto presente (comportamiento previo)",
      all(t in castle_on.lower() for t in ADULT_POS))
check("ON: anti-censor en negativo (comportamiento previo)",
      "censored" in p_on["neg"].lower() and "clothed" in p_on["neg"].lower())
check("OFF/ON: la escena del usuario llega intacta en ambos",
      "castle with unicorns" in castle_off.lower() and "castle with unicorns" in castle_on.lower())
check("OFF: marca inject_nsfw en los params", p_off.get("inject_nsfw") is False)
check("ON: marca inject_nsfw en los params", p_on.get("inject_nsfw") is True)

# ---------------------------------------------------------------- [2] Gate estilo
print("\n[2] Gate de estilo (conceptos + umbral en config/generation_prompt.json)")
STYLE_CASES = [
    ("castle with unicorns, drawing style", True),
    ("CASTILLO CON UNICORNIOS, ESTILO DIBUJO", True),
    ("a detailed pencil sketch of an old fisherman", True),
    ("anime style school girl with pink hair", True),
    ("watercolor landscape of mountains", True),
    ("3d render of a robot", True),
    ("photorealistic portrait of a woman", False),
    ("a red ferrari on a street at night", False),
    ("a castle with unicorns", False),
    ("a woman in a red dress, cinematic style", False),
    ("a cat sitting on a sofa", False),
    ("giant robot fighting a dinosaur in a city", False),
    ("a nurse in a hospital corridor, photorealistic", False),
    ("el hombre debe ser calvo", False),
]
for text, expected in STYLE_CASES:
    got = prompt_requests_nonphotoreal_style(text)
    print(f"    score={score_style_gate(text):.3f} gate={'ON' if got else 'OFF'} (esperado {'ON' if expected else 'OFF'}) | {text}")
    check(f"gate {'ON' if expected else 'OFF'}: {text}", got == expected)

fp_style, suffix_style, type_style = assemble_generation_prompt(
    "flux_dev_abliterated", "castle with unicorns, drawing style", MODS,
    client._model_configs, is_sdxl=False, inject_nsfw=False, style_override=True,
)
print(f"    prompt con gate de estilo → {fp_style}")
check("estilo: el usuario va primero", fp_style.lower().startswith("castle with unicorns"))
fp_style_on, _, _ = assemble_generation_prompt(
    "flux_dev_abliterated", "castle with unicorns, drawing style", MODS,
    client._model_configs, is_sdxl=False, inject_nsfw=True, style_override=True,
)
check("estilo+NSFW ON: la escena sigue ahí",
      "castle with unicorns" in fp_style_on.lower())
check("estilo: sin cola fotorreal", not any(t in fp_style.lower() for t in PHOTOREAL))
check("estilo: sin prefijo photorealistic del modelo", "photorealistic" not in fp_style.lower())
check("estilo: image_type efectivo = auto", type_style == "auto")
check("estilo: sin suffix de piel de la UI", "skin" not in suffix_style.lower())
check("estilo: sin dobles comas", ",," not in fp_style)

neg_style = get_effective_negative(
    "flux_dev_abliterated", client._get_model_conf()["negative_prompt"], MODS,
    client._model_configs, prompt_en="castle with unicorns, drawing style",
    is_sdxl=False, style_override=True,
)
check("estilo: sin anti_stylize en negativo", "cel shaded" not in neg_style.lower())

# ---------------------------------------------------------------- [3] Control
print("\n[3] Control fotorreal (prompt que NO pide otro estilo)")
fp_photo, _, type_photo = assemble_generation_prompt(
    "flux_dev_abliterated", "photorealistic portrait of a woman", MODS,
    client._model_configs, is_sdxl=False, style_override=False,
)
print(f"    prompt fotorreal → {fp_photo}")
check("control: conserva la cola fotorreal", "film grain" in fp_photo.lower())
check("control: image_type efectivo = photoreal", type_photo == "photoreal")
neg_photo = get_effective_negative(
    "flux_dev_abliterated", client._get_model_conf()["negative_prompt"], MODS,
    client._model_configs, prompt_en="photorealistic portrait of a woman",
    is_sdxl=False, style_override=False,
)
check("control: anti_stylize sigue activo", "cel shaded" in neg_photo.lower())

# ---------------------------------------------------------------- [4] LoRAs
print("\n[4] LoRA adulta del catálogo")
import roop.img_editor.gen_lora_resolver as lora_mod
from roop.img_editor.gen_lora_resolver import resolve_scene_loras
from roop.img_editor.gen_prompt_config import get_lora_catalog

nsfw_entries = {e["file"].lower() for e in get_lora_catalog() if e.get("nsfw")}
check("catálogo marca entradas nsfw", bool(nsfw_entries))

# Disco simulado: el filtro se valida a nivel de catálogo, no de instalación.
_orig_exists, orig_sdxl = lora_mod._lora_exists, lora_mod._is_sdxl_lora
lora_mod._lora_exists = lambda n: True
lora_mod._is_sdxl_lora = lambda n: True
try:
    picks_off, _ = resolve_scene_loras(
        "nsfw explicit adult uncensored nude scene", "pony_realism", include_nsfw=False,
    )
    picks_on, _ = resolve_scene_loras(
        "nsfw explicit adult uncensored nude scene", "pony_realism", include_nsfw=True,
    )
finally:
    lora_mod._lora_exists, lora_mod._is_sdxl_lora = _orig_exists, orig_sdxl

off_files = {f.lower() for f, _s, _l in picks_off}
on_files = {f.lower() for f, _s, _l in picks_on}
print(f"    OFF → {sorted(off_files) or '(ninguna)'}")
print(f"    ON  → {sorted(on_files) or '(ninguna)'}")
check("OFF: sin ninguna LoRA adulta", not (off_files & nsfw_entries))
check("ON: entra la LoRA adulta de 'always'", bool(on_files & nsfw_entries))
check("OFF ⊆ ON (el gate solo quita)", off_files <= on_files)

print("\n[4b] Prefijos SDXL con NSFW OFF")
for alias in ("pony_realism", "cyberrealistic_pony"):
    for compact in (False, True):
        off_prefix = get_effective_prefix(
            alias, "photoreal", client._model_configs,
            has_style_modifier=compact, is_sdxl=True, inject_nsfw=False,
        )
        on_prefix = get_effective_prefix(
            alias, "photoreal", client._model_configs,
            has_style_modifier=compact, is_sdxl=True, inject_nsfw=True,
        )
        check(f"{alias} compact={compact}: OFF sin rating_explicit",
              "rating_explicit" not in off_prefix.lower())
        check(f"{alias} compact={compact}: ON conserva rating_explicit",
              "rating_explicit" in on_prefix.lower())
        check(f"{alias} compact={compact}: OFF conserva calidad",
              "score_9" in off_prefix.lower())

# ---------------------------------------------------------------- [5] Contrato APK
print("\n[5] Contrato APK (GradioClient.callGenerate → api_name=generate_image)")
src_events = open("ui/tabs/gen/events.py", encoding="utf-8").read()
check("api_name=generate_image presente", 'api_name="generate_image"' in src_events)
legacy_seg = src_events.split("api_legacy_btn")[1].split("api_name=")[0]
check("legacy usa SOLO _legacy_inputs (10 args)",
      "inputs=_legacy_inputs," in legacy_seg and "nsfw_content" not in legacy_seg)
check("nsfw_content cableado en generate_image_ui", 'ui_comp["nsfw_content"]' in src_events)
check("on_generate_image acepta nsfw_content", "nsfw_content=None" in src_events)
check("API legacy usa NSFW apagado de config", 'get_generate_ui_config().get("nsfw_content_default", False)' in src_events)
check("nsfw_content pasa al cliente como inject_nsfw", "inject_nsfw=bool(nsfw_content)" in src_events)

src_ui = open("ui/tabs/gen/ui.py", encoding="utf-8").read()
check("checkbox en la UI", 'label="🔞 Contenido NSFW"' in src_ui)
check("checkbox exportado en ui_comp", '"nsfw_content": nsfw_content' in src_ui)
check("alias Q4 aceptado por Gradio aunque falte el archivo",
      '("FLUX Dev Q4", "flux_dev_q4")' in src_ui.split("_apk_choices = [", 1)[1].split("]", 1)[0])
from roop.img_editor.gen_prompt_config import get_generate_ui_config
from roop.img_editor.flux_gen_comfy_client import get_default_generation_engine
check("default de Generate en config", get_generate_ui_config().get("default_engine") == "flux_dev_q4")
check("NSFW apagado al abrir", get_generate_ui_config().get("nsfw_content_default") is False)
check("motor normal instalado elegido", get_default_generation_engine() == "flux_dev_q4")
from unittest.mock import patch
with patch("roop.img_editor.flux_gen_comfy_client.get_installed_generation_engines",
           return_value=[("FLUX Abliterated", "flux_dev_abliterated")]):
    check("sin motor normal no elige Abliterated automáticamente",
          get_default_generation_engine() is None)
src_android = open(
    "android_app/autoapp/app/src/main/java/com/autoauto/app/MainActivity.kt",
    encoding="utf-8",
).read()
check("Android ofrece Q4 como primera opción", 'arrayOf("flux_dev_q4", "flux_dev_abliterated"' in src_android)
check("Android migra preferencias por alias", 'putString("genEngineAlias"' in src_android)

# ---------------------------------------------------------------- [6] Sintaxis
print("\n[6] Sintaxis")
for m in (
    "roop/img_editor/flux_gen_comfy_client.py",
    "roop/img_editor/gen_prompt_modifiers.py",
    "roop/img_editor/gen_semantic.py",
    "roop/img_editor/gen_prompt_config.py",
    "roop/img_editor/gen_lora_resolver.py",
    "ui/tabs/gen/ui.py",
    "ui/tabs/gen/events.py",
):
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
