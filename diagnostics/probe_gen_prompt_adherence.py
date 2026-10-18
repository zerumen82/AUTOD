# -*- coding: utf-8 -*-
"""probe_gen_prompt_adherence.py — Generate respeta el prompt (H-GEN-001).

Reproduce el fallo reportado («CASTILLO CON UNICORNIOS, ESTILO DIBUJO» salió
mujer desnuda) a nivel de ensamblado de prompt y valida los dos gates nuevos:

  [1] Obediencia del prompt (v5.1210): el checkbox «Contenido NSFW» ya NO
        decide solo. El prefijo adulto, el anti-censor y las LoRAs adultas
        automáticas solo entran si el PROMPT mide intención adulta
        (adult_gate en config/generation_prompt.json). Un prompt neutro con
        el checkbox ON recibe cero sesgo adulto — el prompt manda siempre.
        ON + prompt adulto → el sesgo adulto sigue intacto.
  [2] Gate de estilo: si el prompt pide un estilo no fotorreal, caen el tipo
      de imagen fotorreal, el suffix de piel de la UI y photoreal_tail.
  [3] Control fotorreal: un prompt que NO pide otro estilo conserva la cola.
  [4] LoRAs: auto con gate adulto medido; la elección del usuario en el
      dropdown manda por encima de los gates (su elección es intención
      explícita) y usa la fuerza del catálogo.
  [5] Contrato APK (10 posicionales en api_name=generate_image) intacto.
  [7] UI con LoRA visible: dropdown lora_choice cableado a generate_image_ui
      y ausente del contrato legacy.

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
check("ON+neutro: sin prefijo adulto en positivo (el prompt manda)",
      not any(t in castle_on.lower() for t in ADULT_POS))
check("ON+neutro: sin bloques anti-censor en negativo",
      not any(t in p_on["neg"].lower() for t in ADULT_NEG))
check("OFF/ON: la escena del usuario llega intacta en ambos",
      "castle with unicorns" in castle_off.lower() and "castle with unicorns" in castle_on.lower())
check("OFF: marca inject_nsfw en los params", p_off.get("inject_nsfw") is False)
check("ON: marca inject_nsfw en los params", p_on.get("inject_nsfw") is True)

# ------------------------------------------- [1b] Prompt adulto + NSFW ON
print("\n[1b] Prompt que SÍ mide intención adulta (el sesgo sigue intacto)")
adult_on, p_adult = client._prepare_prompt_intelligent(
    "nude woman lying on a bed, photorealistic", prompt_modifiers=MODS, inject_nsfw=True,
)
print(f"    NSFW ON + adulto → {adult_on}")
check("ON+adulto: prefijo adulto presente", all(t in adult_on.lower() for t in ADULT_POS))
check("ON+adulto: anti-censor en negativo",
      "censored" in p_adult["neg"].lower() and "clothed" in p_adult["neg"].lower())

# ------------------------------------------- [1c] Gate medido (umbral)
print("\n[1c] adult_gate medido (config/generation_prompt.json → gen_semantic)")
from roop.img_editor.gen_semantic import prompt_requests_adult_content, score_adult_gate
GATE_CASES = [
    ("castle with unicorns, drawing style", False),
    ("photorealistic portrait of a woman", False),
    ("woman in a bikini by the pool", False),
    ("family having barbecue in the garden", False),
    ("woman sleeping in bed, photorealistic", False),
    ("hot sexy woman in a leather jacket, studio light", False),
    ("nude woman lying on a bed", True),
    ("explicit sex scene between a man and a woman", True),
    ("topless sunbather on the sand", True),
    ("nsfw photo of a woman from behind", True),
    ("desnuda de pie frente a la ventana", True),
    ("mujer desnuda en el sofá", True),
]
for text, expected in GATE_CASES:
    score = score_adult_gate(text)
    got = prompt_requests_adult_content(text)
    print(f"    score={score:.3f} gate={'ON' if got else 'OFF'} (esperado {'ON' if expected else 'OFF'}) | {text}")
    check(f"adult_gate {'ON' if expected else 'OFF'}: {text}", got == expected)

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

print("\n[4b] Prefijos SDXL: el contenido adulto exige intención adulta medida")
for alias in ("pony_realism", "cyberrealistic_pony"):
    for compact in (False, True):
        off_prefix = get_effective_prefix(
            alias, "photoreal", client._model_configs,
            has_style_modifier=compact, is_sdxl=True, inject_nsfw=False,
        )
        on_adult_prefix = get_effective_prefix(
            alias, "photoreal", client._model_configs,
            has_style_modifier=compact, is_sdxl=True, inject_nsfw=True,
            adult_intent=True,
        )
        on_neutral_prefix = get_effective_prefix(
            alias, "photoreal", client._model_configs,
            has_style_modifier=compact, is_sdxl=True, inject_nsfw=True,
            adult_intent=False,
        )
        default_neutral_prefix = get_effective_prefix(
            alias, "photoreal", client._model_configs,
            has_style_modifier=compact, is_sdxl=True, inject_nsfw=True,
        )
        check(f"{alias} compact={compact}: OFF sin rating_explicit",
              "rating_explicit" not in off_prefix.lower())
        check(f"{alias} compact={compact}: ON+adulto conserva rating_explicit",
              "rating_explicit" in on_adult_prefix.lower())
        check(f"{alias} compact={compact}: ON+neutro sin rating_explicit",
              "rating_explicit" not in on_neutral_prefix.lower())
        check(f"{alias} compact={compact}: sin adult_intent explícito obedece al prompt",
              "rating_explicit" not in default_neutral_prefix.lower())
        check(f"{alias} compact={compact}: OFF conserva calidad",
              "score_9" in off_prefix.lower())
        check(f"{alias} compact={compact}: ON+neutro conserva calidad",
              "score_9" in on_neutral_prefix.lower())

# ------------------------------------------- [4c] LoRAs: gate + elección
print("\n[4c] _resolve_lora_stack: auto con gate adulto, la elección del usuario manda")
_orig_exists4, _orig_sdxl4 = lora_mod._lora_exists, lora_mod._is_sdxl_lora
lora_mod._lora_exists = lambda n: True
lora_mod._is_sdxl_lora = lambda n: True
_was_sdxl = client._is_sdxl
client._is_sdxl = True
pony_conf = client._model_configs["pony_realism"]
try:
    auto_neutral = client._resolve_lora_stack(
        "Auto", None, pony_conf,
        prompt_en="castle with unicorns, drawing style",
        inject_nsfw=True, adult_intent=False,
    )[0]
    auto_default = client._resolve_lora_stack(
        "Auto", None, pony_conf,
        prompt_en="castle with unicorns, drawing style",
        inject_nsfw=True,
    )[0]
    auto_adult = client._resolve_lora_stack(
        "Auto", None, pony_conf,
        prompt_en="nsfw explicit adult uncensored nude scene",
        inject_nsfw=True, adult_intent=True,
    )[0]
    manual = client._resolve_lora_stack(
        "nsfw-xl-2.1.safetensors", None, pony_conf,
        prompt_en="castle with unicorns, drawing style",
        inject_nsfw=False, adult_intent=False,
    )[0]
finally:
    lora_mod._lora_exists, lora_mod._is_sdxl_lora = _orig_exists4, _orig_sdxl4
    client._is_sdxl = _was_sdxl

nsfw_auto_neutral = {f.lower() for f, _s in auto_neutral} & nsfw_entries
nsfw_auto_default = {f.lower() for f, _s in auto_default} & nsfw_entries
nsfw_auto_adult = {f.lower() for f, _s in auto_adult} & nsfw_entries
print(f"    auto+neutro → {sorted(auto_neutral) or '(ninguna)'}")
print(f"    auto+adulto → {sorted(auto_adult) or '(ninguna)'}")
print(f"    manual      → {manual}")
check("auto+neutro+NSFW ON: sin LoRAs adultas (gate medido)",
      not nsfw_auto_neutral)
check("auto sin adult_intent explícito: el prompt decide (neutro → sin adultas)",
      not nsfw_auto_default)
check("auto+adulto+NSFW ON: entra la LoRA adulta", bool(nsfw_auto_adult))
check("usuario elige LoRA adulta con NSFW OFF: se aplica (su elección manda)",
      bool(manual) and manual[0][0].lower() == "nsfw-xl-2.1.safetensors")
check("elección manual usa la fuerza del catálogo (0.50)",
      bool(manual) and abs(float(manual[0][1]) - 0.50) < 1e-6)

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
      '("FLUX Dev Q4 (seguro, por defecto)", "flux_dev_q4")' in src_ui.split("_apk_choices = [", 1)[1].split("]", 1)[0])
from roop.img_editor.gen_prompt_config import get_generate_ui_config
from roop.img_editor.flux_gen_comfy_client import get_default_generation_engine
check("default de Generate en config", get_generate_ui_config().get("default_engine") == "flux_dev_q4")
check("NSFW apagado al abrir", get_generate_ui_config().get("nsfw_content_default") is False)
check("motor por defecto instalado elegido", get_default_generation_engine() == "flux_dev_q4")
from unittest.mock import patch
with patch("roop.img_editor.flux_gen_comfy_client.get_installed_generation_engines",
           return_value=[("PonyRealism", "pony_realism")]):
    check("sin motor normal instalado no se elige solo un modelo adulto",
          get_default_generation_engine() is None)
src_android = open(
    "android_app/autoapp/app/src/main/java/com/autoauto/app/MainActivity.kt",
    encoding="utf-8",
).read()
check("Android ofrece Q4 como primera opción", 'arrayOf("flux_dev_q4", "flux_dev_abliterated"' in src_android)
check("Android migra preferencias por alias", 'putString("genEngineAlias"' in src_android)

# ---------------------------------------------------------------- [7] LoRA visible
print("\n[7] LoRA visible en la UI y elegido por el usuario")
check("ui.py declara lora_choice en ui_comp", '"lora_choice": lora_choice' in src_ui)
check("ui.py construye el dropdown con el catálogo instalado compatible", "get_available_lora_catalog" in src_ui)
check("events cablea lora_choice en generate_image_ui",
      'ui_comp["lora_choice"]' in src_events.split("api_name=\"generate_image_ui\"")[0])
check("events pasa la elección al cliente como lora_name", "lora_name=" in src_events)
check("legacy NO incluye lora_choice (contrato de 10 intacto)",
      "lora_choice" not in legacy_seg)

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
