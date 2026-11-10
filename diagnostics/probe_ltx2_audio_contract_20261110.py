"""Probe CPU del contrato del workflow LTX-2 Q4 + audio nativo (sin GPU).

Comprueba, sobre el grafo generado por `roop/animate/ltx2_audio.py`:
  - nodos esperados y que no falte ninguno del pipeline AV,
  - todas las entradas enlazadas apuntan a nodos existentes,
  - resolución múltiplo de 32 y fotogramas múltiplo de 8 + 1,
  - el audio nativo llega a CreateVideo (no es un MP4 mudo),
  - 8 pasos destilados con CFG=1 y sampler euler,
  - el prompt viaja al CLIP y la imagen al IMG2VID,
  - los ficheros de modelo declarados tienen el tamaño publicado (si están).

Uso: venv/Scripts/python.exe diagnostics/probe_ltx2_audio_contract_20261110.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.animate import ltx2_audio as ltx  # noqa: E402

REQUIRED_NODES = {
    "UnetLoaderGGUF", "DualCLIPLoaderGGUF", "VAELoader", "LTXVAudioVAELoader",
    "LoadImage", "ImageScale", "LTXVPreprocess", "CLIPTextEncode",
    "LTXVConditioning", "EmptyLTXVLatentVideo", "LTXVImgToVideoInplace",
    "LTXVEmptyLatentAudio", "LTXVConcatAVLatent", "CFGGuider",
    "ManualSigmas", "SamplerCustomAdvanced", "LTXVSeparateAVLatent",
    "VAEDecode", "LTXVAudioVAEDecode", "CreateVideo", "SaveVideo",
}

failures = []


def check(name, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not condition:
        failures.append(f"{name}: {detail}")


def nodes_by_class(workflow):
    found = {}
    for node_id, node in workflow.items():
        found.setdefault(node["class_type"], []).append((node_id, node))
    return found


def main():
    prompt = "She turns to the camera and smiles. Wind moves her hair; birds sing."
    wf = ltx.build_workflow("test_input.png", prompt, seed=7)
    by_class = nodes_by_class(wf)

    missing = REQUIRED_NODES - set(by_class)
    check("nodos_requeridos", not missing, ",".join(sorted(missing)) or "todos")

    # Enlaces: cada input tipo [id, slot] debe apuntar a un nodo existente.
    dangling = []
    for node_id, node in wf.items():
        for name, value in node["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                if value[0] not in wf:
                    dangling.append(f"{node_id}.{name}->{value[0]}")
    check("enlaces_internos", not dangling, ",".join(dangling) or "sin enlaces rotos")

    video_latent = wf["11"]["inputs"]
    check("resolucion_multiplo_32",
          video_latent["width"] % 32 == 0 and video_latent["height"] % 32 == 0,
          f"{video_latent['width']}x{video_latent['height']}")
    frames = video_latent["length"]
    check("fotogramas_8mas1", frames % 8 == 1, str(frames))
    check("audio_empareja_fotogramas",
          wf["13"]["inputs"]["frames_number"] == frames
          and wf["13"]["inputs"]["frame_rate"] == wf["10"]["inputs"]["frame_rate"],
          f"{wf['13']['inputs']}")

    audio_ids = [nid for nid, node in wf.items()
                 if node["class_type"] == "LTXVAudioVAEDecode"]
    create_video = by_class["CreateVideo"][0][1]["inputs"]
    check("audio_nativo_en_mp4",
          bool(audio_ids) and create_video.get("audio") == [audio_ids[0], 0],
          str(create_video.get("audio")))

    sigmas = [float(s) for s in wf["17"]["inputs"]["sigmas"].split(",")]
    check("pasos_destilados", len(sigmas) - 1 == ltx.DISTILLED_STEPS == 8, str(len(sigmas) - 1))
    check("cfg_destilado", wf["15"]["inputs"]["cfg"] == 1.0, str(wf["15"]["inputs"]["cfg"]))
    check("sampler_euler", wf["16"]["inputs"]["sampler_name"] == "euler")

    clip_text = wf["8"]["inputs"]["text"]
    check("prompt_al_clip", clip_text == prompt, clip_text[:40])
    check("imagen_al_img2vid",
          wf["7"]["inputs"]["image"] == ["6", 0] and wf["12"]["inputs"]["image"] == ["7", 0])
    check("sin_upscale_x2", "LTXVLatentUpsampler" not in by_class)
    check("vae_audio_en_checkpoints",
          os.path.basename(ltx._path("checkpoints", ltx.AUDIO_VAE_NAME)) == ltx.AUDIO_VAE_NAME)

    # Ajustes de tamaño: 960x544 ya es válido; 1000x700 y 100 frames se corrigen.
    check("fit_size_redondea", ltx.fit_size(1000, 700) == (992, 704), str(ltx.fit_size(1000, 700)))
    check("fit_frames_redondea", ltx.fit_frames(100) == 97, str(ltx.fit_frames(100)))

    status = ltx.model_status()
    present = [s for s in status if s["bytes"]]
    bad = [s for s in present if not s["ok"]]
    check("tamanos_de_modelos", not bad,
          f"{len(present)}/{len(status)} presentes" + (f"; mal: {[s['name'] for s in bad]}" if bad else ""))
    if len(present) < len(status):
        print("INFO  modelos aún en descarga: " + ", ".join(s["name"] for s in status if not s["bytes"]))

    print(f"\nRESULTADO: {'PASS' if not failures else 'FAIL'} ({len(failures)} fallos)")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
