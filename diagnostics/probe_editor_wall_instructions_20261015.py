"""Diagnóstico sin generación de las instrucciones de pared del 15/10/2026.

El usuario pidió «pintar las paredes de gris» y «quitar los cuadros» y el
resultado no aplicó el cambio. Este probe reproduce el enrutado de producción
sobre la ENTRADA REAL de la edición 4 (fast_1792075811.png, la foto encadenada
tras la calvicie): hechos de autoridad, traducción real ES→EN, queries que
recibe CLIPSeg con su cobertura medida, máscara final, mensaje y los kwargs
exactos que recibiría LongCat. Un cliente falsificado graba el despacho; no se
genera ninguna imagen.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from roop.img_editor.img_editor_manager import ImgEditorManager


class RecorderClient:
    def __init__(self):
        self.calls = []

    def load(self, version):
        self.calls.append(("load", version))
        return True, "OK"

    def generate(self, **kwargs):
        self.calls.append(("generate", kwargs))
        return SimpleNamespace(image=Image.new("RGB", (64, 40), "red")), "OK (stub)"


def _mask_stats(mask):
    if mask is None:
        return None
    arr = np.asarray(mask.convert("L"))
    ys, xs = np.where(arr > 96)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None
    return {"size": list(mask.size), "coverage": round(float((arr > 96).mean()), 4), "bbox": bbox}


_DISPATCH_SEQ = [0]


def _dump_generate(kwargs):
    interesting = {}
    for key, value in kwargs.items():
        if key in ("image",):
            interesting[key] = list(value.size) if hasattr(value, "size") else str(type(value))
        elif key in ("mask_image", "mask", "spatial_reference"):
            interesting[key] = _mask_stats(value) if isinstance(value, Image.Image) else value
        elif isinstance(value, (str, int, float, bool, type(None))):
            interesting[key] = value
        else:
            interesting[key] = str(type(value))
    print("    [generate kwargs] " + json.dumps(interesting, ensure_ascii=False))
    mask = kwargs.get("mask_image")
    base = kwargs.get("image")
    if isinstance(mask, Image.Image) and isinstance(base, Image.Image):
        # Calco de la máscara que recibiría el modelo sobre la foto de
        # producción: permite ver, sin generar, qué queda dentro y fuera.
        _DISPATCH_SEQ[0] += 1
        stats = _mask_stats(mask)
        mask_l = mask.convert("L").resize(base.size, Image.Resampling.NEAREST)
        overlay = Image.composite(
            Image.new("RGB", base.size, (255, 0, 0)),
            base.convert("RGB"),
            mask_l,
        )
        path = f"diagnostics/out/dispatch_mask_{_DISPATCH_SEQ[0]}.jpg"
        overlay.save(path, quality=92)
        print(f"    [máscara despachada] {path} {json.dumps(stats, ensure_ascii=False)}")


def main():
    source = Image.open("ui/tob/ComfyUI/input/fast_1792075811.png").convert("RGB")
    manager = ImgEditorManager()
    from roop.img_editor.prompt_translator import translate_prompt

    masker = manager._get_clipseg_masker()
    assert masker is not None, "CLIPSeg no disponible"

    cases = (
        "pinta las paredes de gris",
        "quita los cuadros de la pared",
        "quita los cuadros",
        # Formulaciones de referencia que SÍ miden la ruta de región (ver
        # MEMORY «paredes azules»): muestran el despacho completo con máscara.
        "las paredes son de color gris",
        "eliminar los cuadros de la pared",
    )
    REGION_CASES = {
        "pinta las paredes de gris",
        "quita los cuadros de la pared",
        "quita los cuadros",
        "las paredes son de color gris",
        "eliminar los cuadros de la pared",
    }

    for index, raw in enumerate(cases, 1):
        en = translate_prompt(raw)
        print(f"\n=== CASO {index}: {raw!r}")
        print(f"    traducción real ES→EN: {en!r}")
        facts = manager._prompt_authority_facts(raw, en)
        keys = ("pose", "body", "local_appearance", "local_region",
                "surface_recolor", "region_ambiguous")
        print(f"    facts: {json.dumps({k: bool(facts.get(k)) for k in keys})}")

        queries = list(dict.fromkeys(
            manager._subject_queries(raw, en)
            + manager._visual_request_queries([raw, en])
        ))
        print(f"    queries al segmentador: {queries}")
        for query in queries:
            try:
                measured = masker.generate_mask(source, query)
            except Exception as exc:
                print(f"    CLIPSeg {query!r}: ERROR {exc}")
                continue
            if measured is None:
                print(f"    CLIPSeg {query!r}: None")
                continue
            arr = np.asarray(measured.convert("L"))
            coverage = float((arr > 96).mean())
            print(f"    CLIPSeg {query!r}: cobertura={coverage:.2%}")
            overlay = source.copy()
            red = Image.new("RGB", source.size, (255, 0, 0))
            soft = Image.fromarray((arr > 96).astype(np.uint8) * 140)
            overlay = Image.composite(Image.blend(overlay, red, 0.55), overlay, soft)
            overlay.thumbnail((1100, 1100), Image.Resampling.LANCZOS)
            safe = "".join(ch if ch.isalnum() else "_" for ch in query)[:40]
            overlay.save(f"diagnostics/out/wall_routing_{index}_{safe}.jpg", quality=85)

        client = RecorderClient()
        with (patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
              patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
                    return_value=client),
              patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
              patch.object(manager, "_release_local_gpu_before_comfy")):
            output, message, mask = manager.generate_intelligent(
                image=source, prompt=raw, engine="longcat", use_rewriter=True,
                auto_upscale=False, enhance_faces=False,
            )
        print(f"    salida={'OK' if output is not None else 'None'} mensaje={message!r}")
        print(f"    máscara devuelta: {_mask_stats(mask)}")
        generates = [args for name, args in client.calls if name == "generate"]
        print(f"    llamadas cliente: {[name for name, _ in client.calls]} "
              f"(generaciones={len(generates)})")
        for args in generates:
            _dump_generate(args)
        if output is not None and output.size != source.size:
            raise AssertionError((output.size, source.size))

        # Contrato v5.1209: las formulaciones que miden una región deben
        # despachar una generación ENMASCARADA y anclada. Sin F1/F2 la ruta
        # ambigua y el recolor por copula abortaban con "no se sustituye" pese
        # a que CLIPSeg medía la zona (11,4 % y 13,1 % en esta foto); sin F4
        # "pinta las paredes de gris" (imperativo) iba a ruta GLOBAL sin
        # ancla (no_kontext=true, denoise 1,0) porque el concepto de recolor
        # no contenía el encuadre de "pintar" (0,149 ES / 0,000 EN).
        if raw in REGION_CASES:
            assert generates, (raw, message, "la región medida no llegó a generación")
            first = generates[0]
            assert first["mask_image"] is not None, (raw, "generación sin máscara")
            assert first.get("no_kontext") is False, (raw, "región sin anclar")
            assert "no se sustituye" not in message, (raw, message)
            print(f"    contrato región OK: {raw!r}")

    # v5.1211 — Fix A (regresión del defecto de edit_1792086034): en MODO
    # CALIDAD la ruta compuesta medida también debe despachar la cláusula de
    # pared y aplicar el acabado TOP después. Antes `not quality_hybrid`
    # mandaba la instrucción compuesta al paso de sujeto, que descarta la
    # cláusula de región («'the walls' … no coincide con una cara; se
    # descarta»): el prompt de calidad quedaba solo con cláusulas de sujeto
    # (edit_1792086034: paredes intactas con el MISMO texto que
    # edit_1792086138 sí las pintó en modo normal).
    raw = "la mujer debe ir sin zapatos, sin medias, las paredes deben ser grises"
    en = translate_prompt(raw)
    print(f"\n=== CASO 6 (modo calidad, compuesto): {raw!r}")
    print(f"    traducción real ES→EN: {en!r}")
    client = RecorderClient()
    finish_calls = []

    def _finish(output, image_, **kwargs):
        finish_calls.append(kwargs)
        return output, "acabado stub", {"stub": True}

    with (patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
          patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
                return_value=client),
          patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
          patch("roop.img_editor.top_tier_finish.run_quality_enhance_after_edit",
                _finish),
          patch.object(manager, "_release_local_gpu_before_comfy")):
        output, message, mask = manager.generate_intelligent(
            image=source, prompt=raw, engine="longcat", use_rewriter=True,
            auto_upscale=False, enhance_faces=False, quality_mode=True,
        )
    generates = [args for name, args in client.calls if name == "generate"]
    print(f"    salida={'OK' if output is not None else 'None'} mensaje={message!r}")
    for args in generates:
        _dump_generate(args)
    assert output is not None, message
    assert generates, (raw, message, "el compuesto de calidad no llegó a generación")
    assert all(a.get("mask_image") is not None for a in generates), \
        (raw, "generación sin máscara en modo calidad")
    wall_prompts = [a for a in generates
                    if "wall" in str(a.get("prompt", "")).lower()]
    assert wall_prompts, \
        (raw, message, "la cláusula de pared no llegó a la instrucción en modo calidad")
    assert finish_calls, (raw, "el acabado TOP no se aplicó tras la ruta compuesta de calidad")
    print(f"    contrato calidad OK: instrucción con pared + acabado aplicado "
          f"({len(generates)} generación(es))")

    print("\nPASS: enrutado reproducido sin generación y contrato de región cumplido")


if __name__ == "__main__":
    main()
