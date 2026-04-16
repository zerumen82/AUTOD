"""
Descargar LTX Video 0.9.5 para ComfyUI
======================================

LTX Video 0.9.5 es un modelo de generacion de video optimizado para GPUs con 8GB VRAM.
NOTA: 0.9.5 tiene arquitectura VAE compatible con ComfyUI (0.9.1 no es compatible).

Requisitos:
- RTX 3060 Ti 8GB o similar
- ~10GB de espacio en disco
- Conexion a internet

Uso:
    python tools/download_ltx_video.py
"""

import os
import sys
import urllib.request
import json
from pathlib import Path


def get_project_root():
    """Obtiene la raiz del proyecto"""
    script_dir = Path(__file__).parent
    return script_dir.parent


def download_file(url: str, dest_path: str, description: str = ""):
    """Descarga un archivo con barra de progreso"""
    print(f"\n[DESCARGA] {description}")
    print(f"   URL: {url}")
    print(f"   Destino: {dest_path}")
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    # Descargar con progreso
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, (downloaded / total_size) * 100) if total_size > 0 else 0
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024) if total_size > 0 else 0
        print(f"\r   Progreso: {percent:.1f}% ({mb_downloaded:.1f}MB / {mb_total:.1f}MB)", end="", flush=True)
    
    try:
        urllib.request.urlretrieve(url, dest_path, progress_hook)
        print("\n   [OK] Descarga completada!")
        return True
    except Exception as e:
        print(f"\n   [ERROR] {e}")
        return False


def download_ltx_video():
    """Descarga LTX Video 0.9.5 desde HuggingFace (arquitectura compatible con ComfyUI)"""
    
    project_root = get_project_root()
    diffusion_models_path = project_root / "ui" / "tob" / "ComfyUI" / "models" / "diffusion_models"
    ltx_path = diffusion_models_path / "ltx-video-0.9.5"
    
    print("=" * 60)
    print("[LTX VIDEO] LTX Video 0.9.5 - Descargador")
    print("=" * 60)
    print(f"\n[INFO] Directorio de destino: {ltx_path}")
    print("\n[INFO] LTX Video 0.9.5 tiene arquitectura compatible con ComfyUI")
    print("       (0.9.1 tiene una arquitectura de VAE incompatible)")
    
    # Crear directorio
    ltx_path.mkdir(parents=True, exist_ok=True)
    
    # Archivos a descargar desde HuggingFace
    # LTX Video 0.9.5 usa el repositorio: Lightricks/LTX-Video-0.9.5
    # Estructura diffusers: transformer/, vae/, text_encoder/, tokenizer/
    # NOTA: El VAE de 0.9.5 tiene arquitectura incompatible con ComfyUI
    # Usamos el VAE del repositorio original LTX-Video que es compatible
    files_to_download = [
        {
            "url": "https://huggingface.co/Lightricks/LTX-Video-0.9.5/resolve/main/transformer/diffusion_pytorch_model.safetensors",
            "dest": str(ltx_path / "transformer.safetensors"),
            "description": "Transformer LTX Video 0.9.5 (~9.5GB)"
        },
        {
            "url": "https://huggingface.co/Lightricks/LTX-Video/resolve/main/vae/diffusion_pytorch_model.safetensors",
            "dest": str(ltx_path / "vae.safetensors"),
            "description": "VAE compatible LTX Video (~1.6GB)"
        }
    ]
    
    # Verificar si ya existe
    model_exists = (ltx_path / "transformer.safetensors").exists()
    vae_exists = (ltx_path / "vae.safetensors").exists()
    
    if model_exists and vae_exists:
        print("\n[OK] LTX Video 0.9.5 ya esta instalado!")
        print(f"     Modelo: {ltx_path / 'transformer.safetensors'}")
        print(f"     VAE: {ltx_path / 'vae.safetensors'}")
        return True
    
    print("\n[DESCARGA] Archivos a descargar:")
    for f in files_to_download:
        exists = "[OK] (ya existe)" if os.path.exists(f["dest"]) else "[--]"
        print(f"   {exists} {f['description']}")
    
    # Descargar archivos
    success = True
    for file_info in files_to_download:
        if os.path.exists(file_info["dest"]):
            print(f"\n[SKIP] Saltando (ya existe): {file_info['description']}")
            continue
            
        if not download_file(file_info["url"], file_info["dest"], file_info["description"]):
            success = False
            break
    
    if success:
        # Copiar VAE a la carpeta vae/ para que VAELoader pueda encontrarlo
        vae_path = ltx_path.parent.parent / "vae"
        vae_path.mkdir(parents=True, exist_ok=True)
        vae_dest = vae_path / "ltx-video_vae.safetensors"
        
        if not vae_dest.exists() and (ltx_path / "vae.safetensors").exists():
            import shutil
            shutil.copy(str(ltx_path / "vae.safetensors"), str(vae_dest))
            print(f"\n[OK] VAE copiado a: {vae_dest}")
        
        print("\n" + "=" * 60)
        print("[OK] LTX Video 0.9.5 instalado correctamente!")
        print("=" * 60)
        print("\n[INFO] Para usar LTX Video:")
        print("   1. Abre ComfyUI")
        print("   2. Selecciona el modelo 'ltx-video-0.9.5/transformer.safetensors'")
        print("   3. Selecciona el VAE 'ltx-video-0.9.5_vae.safetensors'")
        print("   4. Usa el workflow LTX Video")
        print("\n[WARN] Nota: LTX Video requiere nodos especializados:")
        print("   - LTXVImgToVideo")
        print("   - LTXVConditioning")
        print("   - LTXVScheduler")
        print("\n   Si no tienes estos nodos, instala la extension ComfyUI-LTXVideo")
        return True
    else:
        print("\n[ERROR] Error durante la descarga. Intenta manualmente:")
        print("   1. Ve a: https://huggingface.co/Lightricks/LTX-Video-0.9.5")
        print("   2. Descarga los archivos manualmente")
        print(f"   3. Colocalos en: {ltx_path}")
        return False


def check_comfyui_nodes():
    """Verifica si los nodos de LTX Video estan disponibles en ComfyUI"""
    print("\n[CHECK] Verificando nodos de ComfyUI...")
    
    try:
        import requests
        import os
        # Detectar puerto de ComfyUI dinamicamente
        comfy_url = os.environ.get('COMFYUI_URL', 'http://127.0.0.1:8188')
        for port in ['8188', '8189', '8190', '8888']:
            try:
                test_url = f"http://127.0.0.1:{port}"
                response = requests.get(f"{test_url}/system_stats", timeout=1)
                if response.status_code == 200:
                    comfy_url = test_url
                    break
            except:
                continue
        
        response = requests.get(f"{comfy_url}/object_info", timeout=5)
        if response.status_code == 200:
            nodes = response.json().keys()
            
            required_nodes = [
                "LTXVImgToVideo",
                "LTXVConditioning", 
                "LTXVScheduler",
                "LTXVPreprocess"
            ]
            
            print("\n[INFO] Estado de nodos LTX:")
            for node in required_nodes:
                status = "[OK] Disponible" if node in nodes else "[X] No encontrado"
                print(f"   {status}: {node}")
            
            missing = [n for n in required_nodes if n not in nodes]
            if missing:
                print("\n[WARN] Faltan nodos. Instala la extension:")
                print("   cd ui/tob/ComfyUI/custom_nodes")
                print("   git clone https://github.com/Lightricks/ComfyUI-LTXVideo")
                return False
            return True
        else:
            print("   [WARN] ComfyUI no esta ejecutandose")
            return False
    except Exception as e:
        print(f"   [WARN] No se pudo verificar: {e}")
        return False


def main():
    print("\n[LTX VIDEO] Instalador de LTX Video 0.9.5")
    print("=" * 40)
    
    # Descargar modelo
    download_ltx_video()
    
    # Verificar nodos
    check_comfyui_nodes()
    
    print("\n[OK] Proceso completado!")


if __name__ == "__main__":
    main()
