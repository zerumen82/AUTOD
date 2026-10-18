#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descarga aprobada (2026-10-18, "descárgalos") de checkpoints sin censura para GENERAR.

Organizacion elegida por el usuario: PREFIJOS de categoria y todo plano.
Motivo: `_list_generation_checkpoint_files()` en roop/img_editor/flux_gen_comfy_client.py
usa os.listdir sin recursion, asi que las subcarpetas no saldrian en Generate.

    REAL_  -> fotorrealista / foto NSFW (checkpoints SDXL)
    ILUS_  -> ilustracion / anime        (checkpoints SDXL)
    DIT_   -> DiT GGUF                   (diffusion_models)
    VAE_   -> VAE auxiliares             (vae)

Por que no hf_hub_download: con Xet (hf_xet instalado) se quedaba en 0 bytes y
sin Xet tampoco arrancaba en ficheros de 6,5 GB. Medido en esta maquina:
1 conexion 0,5 MB/s, 8 conexiones 3,7 MB/s -> chunks en paralelo con Range.

Contrato de seguridad del repo: se escribe en `<final>.part` y solo se renombra
al nombre final cuando el tamano coincide con el declarado por el Hub. El estado
de chunks ya bajados vive en `<final>.part.json`, asi que se puede reanudar.
No instala paquetes y no toca produccion.
"""
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

LOG_NAME = "download_uncensored_models_20261018.log"
CHUNK = 32 * 1024 * 1024
WORKERS = 8
RETRIES = 4

# (repositorio, archivo en el repo, subcarpeta destino, nombre final, bytes esperados)
BATCH = [
    ("KamCastle/lustify4", "lustifySDXLNSFWSFW_v40.safetensors",
     "checkpoints", "REAL_lustify4_v40.safetensors", 6938042770),
    ("fancyfeast/bigaspv2-5", "bigaspv25-20250716.safetensors",
     "checkpoints", "REAL_bigasp_v25.safetensors", 6938040682),
    ("Aero666/Biglove.Xl4", "bigLove_xl4.safetensors",
     "checkpoints", "REAL_biglove_xl4.safetensors", 6938042722),
    ("LyliaEngine/waiIllustriousSDXL_v170", "waiIllustriousSDXL_v170.safetensors",
     "checkpoints", "ILUS_wai_illustrious_v17.safetensors", 6938040682),
    ("Laxhar/noobai-XL-Vpred-1.0", "NoobAI-XL-Vpred-v1.0.safetensors",
     "checkpoints", "ILUS_noobai_xl_vpred_1.safetensors", 7105350110),
    ("OnomaAIResearch/Illustrious-XL-v2.0", "Illustrious-XL-v2.0.safetensors",
     "checkpoints", "ILUS_illustrious_xl_v2.safetensors", 6938040674),
    ("silveroxides/Chroma1-HD-GGUF", "Chroma1-HD-Q4_K_M.gguf",
     "diffusion_models", "DIT_chroma1-hd_q4_k_m.gguf", 5566533792),
    ("madebyollin/sdxl-vae-fp16-fix", "sdxl_vae.safetensors",
     "vae", "VAE_sdxl_fp16_fix.safetensors", 334641162),
]


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


ROOT = get_project_root()
MODELS = os.path.join(ROOT, "ui", "tob", "ComfyUI", "models")
LOG_PATH = os.path.join(ROOT, "logs", LOG_NAME)
LOCK = threading.Lock()


def log(message):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def human(num):
    return f"{num / 1024 ** 3:.2f} GB"


def resolve_url(repo, filename):
    base = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    response = requests.get(base, allow_redirects=True, stream=True, timeout=60)
    response.close()
    return response.url


def load_state(state_path, expected):
    try:
        with open(state_path, "r", encoding="utf-8") as stream:
            state = json.load(stream)
        if state.get("expected") == expected:
            return set(state.get("done", []))
    except (OSError, ValueError):
        pass
    return set()


def save_state(state_path, expected, done):
    tmp = state_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as stream:
        json.dump({"expected": expected, "done": sorted(done)}, stream)
    os.replace(tmp, state_path)


def fetch_chunk(url, index, part_path, expected, state_path, done):
    start = index * CHUNK
    end = min(start + CHUNK, expected) - 1
    last_error = None
    for attempt in range(1, RETRIES + 1):
        try:
            response = requests.get(
                url, headers={"Range": f"bytes={start}-{end}"}, timeout=(30, 180)
            )
            if response.status_code not in (200, 206):
                raise RuntimeError(f"HTTP {response.status_code}")
            data = response.content
            if len(data) != end - start + 1:
                raise RuntimeError(f"bytes {len(data)} != {end - start + 1}")
            # Escritura y estado bajo el mismo cerrojo: si el proceso muere,
            # lo que quede marcado en disco esta completo de verdad.
            with LOCK:
                with open(part_path, "r+b") as stream:
                    stream.seek(start)
                    stream.write(data)
                done.add(index)
                save_state(state_path, expected, done)
            return index
        except Exception as error:  # red, corte, tamano corto
            last_error = error
            time.sleep(min(2 ** attempt, 12))
    raise RuntimeError(f"chunk {index}: {type(last_error).__name__}: {last_error}")


def download(repo, filename, dest, expected):
    url = resolve_url(repo, filename)
    part = dest + ".part"
    state_path = part + ".json"

    if os.path.isfile(part) and os.path.getsize(part) != expected:
        os.remove(part)

    if not os.path.isfile(part):
        with open(part, "wb") as stream:
            stream.truncate(expected)

    done = load_state(state_path, expected)
    total_chunks = (expected + CHUNK - 1) // CHUNK
    pending = [i for i in range(total_chunks) if i not in done]
    log(f"  chunks: {len(pending)}/{total_chunks} pendientes ({len(done)} reanudados)")

    started = time.time()
    bytes_done = len(done) * CHUNK
    stop = threading.Event()

    def reporter():
        last = bytes_done
        while not stop.wait(30):
            now = len(load_state(state_path, expected)) * CHUNK
            speed = (now - last) / 30 / 1024 ** 2
            log(f"  progreso {human(min(now, expected))}/{human(expected)} · {speed:.2f} MB/s")
            last = now

    thread = threading.Thread(target=reporter, daemon=True)
    thread.start()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [
            pool.submit(fetch_chunk, url, i, part, expected, state_path, done)
            for i in pending
        ]
        for future in futures:
            future.result()

    stop.set()
    real = os.path.getsize(part)
    if real != expected:
        raise RuntimeError(f"tamano final {real} != {expected}")
    os.replace(part, dest)
    if os.path.isfile(state_path):
        os.remove(state_path)
    return real, time.time() - started


def main():
    log("=== Inicio descarga checkpoints sin censura (autorizada 2026-10-18) ===")
    ok, skipped, failed = [], [], []

    for repo, filename, folder, final_name, expected in BATCH:
        dest_dir = os.path.join(MODELS, folder)
        dest = os.path.join(dest_dir, final_name)
        os.makedirs(dest_dir, exist_ok=True)

        if os.path.isfile(dest) and os.path.getsize(dest) == expected:
            log(f"YA PRESENTE {final_name} ({human(expected)}) - sin cambios")
            skipped.append(final_name)
            continue

        log(f"DESCARGA {repo}/{filename} -> {folder}/{final_name} ({human(expected)})")
        try:
            size, elapsed = download(repo, filename, dest, expected)
        except Exception as error:
            log(f"FALLO {final_name}: {type(error).__name__}: {error}")
            failed.append(final_name)
            continue
        speed = size / max(elapsed, 1) / 1024 ** 2
        log(f"OK {final_name} {human(size)} en {elapsed / 60:.1f} min ({speed:.2f} MB/s medio)")
        ok.append(final_name)

    log(f"=== Fin: {len(ok)} descargados, {len(skipped)} ya presentes, {len(failed)} fallos ===")
    if failed:
        log("Fallos: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
