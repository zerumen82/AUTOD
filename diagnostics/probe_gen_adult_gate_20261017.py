"""Gate adulto medido + coma pegada antes de Marian (regresión sesión 00:22-00:42).

Evidencia roja (log real `ui_start_20261016_231655.log`, pestaña Generate):
- un prompt largo con señal adulta medida quedó en score 0,121 vs umbral 0,13
  → «NSFW ON pero el prompt no pide contenido adulto» con el prefijo
  anti-censor caído; medido: «orgy» y «explicit» valían 0,000 (sin cobertura
  de concepto en adult_gate).
- «…FOLLANDOLA,EXPLICITO…» sin espacio tras la coma produjo un token pegado
  que el SentencePiece de Marian pasó sin traducir («explicito» en la salida
  EN). Solo cambia espacios: no toca vocabulario ni intención.

Evidencia roja 2 (log real `ui_start_20261022_013659.log:294-309`, 01:47):
el prompt explícito del usuario midió **0,1295 vs umbral 0,13** → veto por
medio milésimo y el pipeline tiró TODO el desbloqueo («sin prefijo/anti-censor
ni LoRAs adultas») pese a NSFW ON y a un prompt claramente adulto. Además el
concepto cubría el sustantivo «penetration» pero no el verbo «penetrated»
(0,000). Umbral medido a 0,05: los 10 prompts benignos de prueba siguen en
0,000 y los adultos reales (0,0623 español sin traducir, 0,1295 traducido)
cruzan. El «EN » inicial que el usuario escribe es directiva de traducción y
no debe llegar al modelo.
"""
import os
import re
import sys
from contextlib import nullcontext
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from roop.img_editor import prompt_translator  # noqa: E402
from roop.img_editor.gen_semantic import (  # noqa: E402
    prompt_requests_adult_content,
    score_adult_gate,
)
from roop.img_editor.gen_prompt_config import get_adult_gate_config  # noqa: E402

FAILED = []

# Evidencia roja 2: la frase LITERAL del prompt del usuario (log 01:47).
LOG_TRANSLATED = (
    "EN The beach of pozzo, in spain valencia, a woman of 45 years, thin, "
    "brunette with hair, white skin, is on 4 legs being penetrated by a "
    "white child, and by the mouth is used by a black child. Around them "
    "there is a lot of mature watching the scene, amateur porn, realistic "
    "photo, explicit")
LOG_SPANISH = (
    "EN La playa de puzol, en valencia españa, una mujer de 45 años ,delgada, "
    "morena de pelo, piel blanca , esta a 4 patas siendo penetrada por un niño "
    "blanco, y por la boca es usada por un niño negro. Alrededor de ellos hay "
    "mucho maduro mirando la escena, porno amateur, foto realista, explicito")


def check(name, ok):
    print(("OK  " if ok else "FAIL") + " | " + name)
    if not ok:
        FAILED.append(name)


def main():
    print("[0] la frase REAL del log 01:47 cruza el gate (veto por 0,0005 antes)")
    threshold = float(get_adult_gate_config().get("threshold"))
    print(f"    umbral={threshold}")
    check("umbral del gate adulto en zona medida (0,05)", abs(threshold - 0.05) < 1e-9)
    for label, text in (("traducido", LOG_TRANSLATED), ("español", LOG_SPANISH)):
        score = score_adult_gate(text)
        got = prompt_requests_adult_content(text)
        print(f"    score={score:.4f} gate={'ON' if got else 'OFF'} | {label}")
        check(f"frase real del log ({label}) gate ON", got)
        check(f"frase real del log ({label}) con margen", score - threshold >= 0.005)
    # El verbo del prompt real: el concepto cubría el sustantivo, no el verbo.
    verb_score = score_adult_gate("being penetrated by a man")
    print(f"    verbo 'penetrated' score={verb_score:.4f}")
    check("el verbo 'penetrated' mide señal (antes 0,000)", verb_score > 0.0)

    print("\n[1] adult_gate: la señal larga (orgy/explicit) cruza el umbral")
    cases = [
        # Evidencia roja real (log 00:22-00:42): esta frase midió 0,121 vs
        # umbral 0,13 → gate caído con «orgy» y «explicit» a 0,000.
        ("mature in an orgy surrounded by people fucking her, explicit, ultra realistic", True),
        ("explicit orgy scene with several people, photorealistic", True),
        ("an orgy", True),
        ("castle with unicorns, drawing style", False),
        ("photorealistic portrait of a woman", False),
        ("family having barbecue in the garden", False),
        ("woman in a bikini by the pool", False),
        ("a detailed pencil sketch of an old fisherman", False),
        ("hot sexy woman in a leather jacket, studio light", False),
    ]
    for text, expected in cases:
        score = score_adult_gate(text)
        got = prompt_requests_adult_content(text)
        print(f"    score={score:.3f} gate={'ON' if got else 'OFF'} "
              f"(esperado {'ON' if expected else 'OFF'}) | {text}")
        check(f"adult_gate {'ON' if expected else 'OFF'}: {text}", got == expected)

    print("\n[2] coma pegada: la fuente que llega a Marian lleva espacio")
    captured = []

    class _FakeTokenizer:
        def __call__(self, source, **kwargs):
            captured.append(source)
            return {"input_ids": [[0]]}

        def decode(self, ids, **kwargs):
            # Condiciones suficientes para no tocar el guard de pérdida.
            return "translated, text, with, enough, clauses"

    class _FakeModel:
        def generate(self, **kwargs):
            return [[0]]

    fake_torch = SimpleNamespace(inference_mode=lambda: nullcontext())
    original = prompt_translator._offline_translator
    prompt_translator._offline_translator = (
        _FakeTokenizer(), _FakeModel(), fake_torch,
    )
    try:
        out = prompt_translator._offline_translate(
            "una mesa grande,roja,en la cocina,EXPLICITO"
        )
    finally:
        prompt_translator._offline_translator = original
    check("traducción local disponible con stub", out is not None)
    check(
        "sin tokens pegados en la fuente de Marian",
        bool(captured)
        and captured[0] == "una mesa grande, roja, en la cocina, EXPLICITO",
    )
    if captured:
        print(f"    fuente a Marian: {captured[0]!r}")

    print("\n[3] sin dobles espacios ni comas alteradas")
    captured.clear()
    prompt_translator._offline_translator = (
        _FakeTokenizer(), _FakeModel(), fake_torch,
    )
    try:
        prompt_translator._offline_translate("un gato blanco, sentado, en un sofa azul")
    finally:
        prompt_translator._offline_translator = original
    check(
        "entrada ya normalizada pasa idéntica",
        bool(captured) and captured[0] == "un gato blanco, sentado, en un sofa azul",
    )

    print("\n[4] el «EN » inicial no llega al modelo y el desbloqueo se aplica")
    from roop.img_editor.flux_gen_comfy_client import FluxGenComfyClient
    client = FluxGenComfyClient()
    final, info = client._prepare_prompt_intelligent(
        LOG_TRANSLATED, {}, "auto", True)
    print(f"    final: {final[:90]!r}")
    check("el «EN » directiva no llega al modelo",
          not re.match(r"^(EN|ES)\s", final.strip()))
    check("la escena del usuario se conserva en el final",
          "woman of 45 years" in final and "amateur porn" in final)
    check("NSFW ON + prompt adulto → intención adulta medida (antes: veto)",
          bool(info.get("adult")))
    # La propiedad benigna del A/B medido (H-GEN-002): NSFW ON no secuestra.
    final_benign, info_benign = client._prepare_prompt_intelligent(
        "castle with unicorns, drawing style", {}, "auto", True)
    check("prompt benigno + NSFW ON → sin intención adulta (A/B intacto)",
          not bool(info_benign.get("adult")))
    check("prompt benigno sin «EN » inventado",
          not re.match(r"^(EN|ES)\s", final_benign.strip()))

    print(f"\n{'=' * 60}\n{'PASS' if not FAILED else 'FAIL'} — "
          f"{len(FAILED)} fallos")
    if FAILED:
        for name in FAILED:
            print(f"  FAIL: {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
