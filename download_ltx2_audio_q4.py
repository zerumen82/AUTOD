"""Descarga los ficheros de LTX-2 destilado Q4 para la prueba de audio nativo en Animate.

Ruta 8 GB: transformer GGUF Q4_K_M + Gemma 3 12B GGUF Q4 + embeddings connector +
VAE de vídeo + VAE de audio. Reanudable: si se corta, relanzar el script continúa.

Uso:
    venv/Scripts/python.exe download_ltx2_audio_q4.py [--check]
"""
import os
import sys

from huggingface_hub import hf_hub_download

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(ROOT, "ui", "tob", "ComfyUI", "models")

# (repo, fichero en el repo, carpeta destino de ComfyUI)
FILES = [
    ("Kijai/LTXV2_comfy", "diffusion_models/ltx-2-19b-distilled_Q4_K_M.gguf", "diffusion_models"),
    ("Kijai/LTXV2_comfy", "text_encoders/ltx-2-19b-embeddings_connector_distill_bf16.safetensors", "text_encoders"),
    ("unsloth/gemma-3-12b-it-GGUF", "gemma-3-12b-it-Q4_K_M.gguf", "text_encoders"),
    ("Kijai/LTXV2_comfy", "VAE/LTX2_video_vae_bf16.safetensors", "vae"),
    # LTXVAudioVAELoader solo lista la carpeta checkpoints
    ("Kijai/LTXV2_comfy", "VAE/LTX2_audio_vae_bf16.safetensors", "checkpoints"),
]

EXPECTED = {
    "ltx-2-19b-distilled_Q4_K_M.gguf": 12651789536,
    "ltx-2-19b-embeddings_connector_distill_bf16.safetensors": 2862983784,
    "gemma-3-12b-it-Q4_K_M.gguf": 7300778336,
    "LTX2_video_vae_bf16.safetensors": 2445008522,
    "LTX2_audio_vae_bf16.safetensors": 217740136,
}


def target_path(repo_file, folder):
    return os.path.join(MODELS, folder, os.path.basename(repo_file))


def check():
    missing = []
    for repo, repo_file, folder in FILES:
        path = target_path(repo_file, folder)
        size = os.path.getsize(path) if os.path.exists(path) else 0
        expected = EXPECTED[os.path.basename(repo_file)]
        state = "OK" if size == expected else ("PARCIAL" if size else "FALTA")
        print(f"[{state}] {os.path.relpath(path, ROOT)} {size / 1e9:.2f} / {expected / 1e9:.2f} GB")
        if size != expected:
            missing.append(path)
    return missing


def collect_staged(folder):
    """hf_hub_download conserva la ruta del repo dentro de local_dir.

    Deja el fichero en `<carpeta>/<subruta del repo>/<nombre>`; aquí se sube a
    `<carpeta>/<nombre>` para que ComfyUI lo liste. No borra parciales.
    """
    moved = []
    dest_dir = os.path.join(MODELS, folder)
    for root, _dirs, files in os.walk(dest_dir):
        if os.path.abspath(root) == os.path.abspath(dest_dir) or ".cache" in root:
            continue
        for name in files:
            src = os.path.join(root, name)
            if name in EXPECTED and os.path.getsize(src) == EXPECTED[name]:
                target = os.path.join(dest_dir, name)
                if not os.path.exists(target):
                    os.replace(src, target)
                    moved.append(name)
    return moved


def main():
    if "--check" in sys.argv:
        return 0 if not check() else 1

    for folder in {entry[2] for entry in FILES}:
        for name in collect_staged(folder):
            print(f"[moved] {name} -> {folder}")

    for repo, repo_file, folder in FILES:
        dest_dir = os.path.join(MODELS, folder)
        os.makedirs(dest_dir, exist_ok=True)
        basename = os.path.basename(repo_file)
        path = target_path(repo_file, folder)
        if os.path.exists(path) and os.path.getsize(path) == EXPECTED[basename]:
            print(f"[skip] {basename} ya completo")
            continue
        print(f"[get] {repo}/{repo_file} -> {os.path.relpath(dest_dir, ROOT)}", flush=True)
        downloaded = hf_hub_download(
            repo_id=repo,
            filename=repo_file,
            local_dir=dest_dir,
            force_download=False,
        )
        if os.path.abspath(downloaded) != os.path.abspath(path):
            os.replace(downloaded, path)
    return 0 if not check() else 1


if __name__ == "__main__":
    sys.exit(main())
