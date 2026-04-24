#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descargador de LoRAs compatibles con FLUX.2-klein-4B
Usa huggingface-cli (requiere login) o descarga directa pública
"""

import os
import sys
import subprocess
import requests
from pathlib import Path

# Configuración
LORA_DIR = r"D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models\loras"
BASE_URL = "https://huggingface.co"

# LoRAs compatibles conocidos (públicos o con access token)
COMPATIBLE_LORAS = {
    # (repo_id, filename, tamaño_MB, requiere_login)
    "DeverStyle/Flux.2-Klein-Loras": [
        ("alba_baptista_vrtlalbabaptista_flux2_klein_4b.safetensors", 158, False),
        ("retoque_vertical_vrtlvertical_flux2_klein_4b.safetensors", 142, False),
    ],
    "fal/flux-2-klein-4B-zoom-lora": [
        ("flux-2-klein-4b-zoom-lora.safetensors", 45, False),  # Pequeño, público
    ],
    # Agregar más según necesidad
}

def check_huggingface_login():
    """Verifica si el usuario está logueado en HuggingFace"""
    try:
        result = subprocess.run(
            ["huggingface-cli", "whoami"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, None
    except:
        return False, None

def download_with_huggingface_cli(repo_id, filename, dest_dir):
    """Descarga usando huggingface-cli (requiere login)"""
    try:
        cmd = [
            "huggingface-cli", "download",
            repo_id, filename,
            "--local-dir", dest_dir,
            "--local-dir-use-symlinks", "false"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            return True, f"Descargado: {filename}"
        return False, result.stderr[:200]
    except Exception as e:
        return False, str(e)

def download_with_requests(repo_id, filename, dest_dir):
    """Descarga directa desde HuggingFace (solo archivos públicos)"""
    try:
        url = f"{BASE_URL}/{repo_id}/resolve/main/{filename}"
        dest_path = os.path.join(dest_dir, filename)
        
        print(f"[DOWNLOAD] Descargando {filename} desde {repo_id}...")
        r = requests.get(url, stream=True, timeout=60)
        if r.status_code == 200:
            total = int(r.headers.get('content-length', 0))
            with open(dest_path, 'wb') as f:
                downloaded = 0
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            percent = (downloaded / total) * 100
                            print(f"\r[DOWNLOAD] {percent:.1f}%", end="", flush=True)
            print(f"\n[DONE] {filename} guardado en {dest_path}")
            return True, "OK"
        else:
            return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 60)
    print("DESCARGADOR DE LORAS COMPATIBLES - FLUX.2-klein-4B")
    print("=" * 60)
    
    # Crear directorio si no existe
    os.makedirs(LORA_DIR, exist_ok=True)
    print(f"[INFO] Directorio LoRAs: {LORA_DIR}")
    
    # Verificar login
    logged_in, user = check_huggingface_login()
    if logged_in:
        print(f"[INFO] HuggingFace: Logueado como {user}")
    else:
        print("[WARN] No hay login en HuggingFace. Solo se descargarán LoRAs públicos.")
        print("[INFO] Para acceso privado: huggingface-cli login")
    
    # Listar LoRAs disponibles
    print("\nSelecciona un LoRA para descargar:")
    available = []
    idx = 1
    for repo, files in COMPATIBLE_LORAS.items():
        for filename, size_mb, needs_login in files:
            if needs_login and not logged_in:
                print(f"  {idx}. {filename} ({size_mb}MB) - [REQUIERE LOGIN]")
            else:
                print(f"  {idx}. {filename} ({size_mb}MB) - PÚBLICO")
            available.append((repo, filename, size_mb, needs_login))
            idx += 1
    
    if not available:
        print("\n[ERROR] No hay LoRAs disponibles. Agrega entradas a COMPATIBLE_LORAS.")
        return
    
    try:
        choice = input("\nOpción (número, o 'q' para salir): ").strip()
        if choice.lower() == 'q':
            return
        
        choice_idx = int(choice) - 1
        if choice_idx < 0 or choice_idx >= len(available):
            print("[ERROR] Opción inválida")
            return
        
        repo_id, filename, size_mb, needs_login = available[choice_idx]
        
        if needs_login and not logged_in:
            print(f"[ERROR] {filename} requiere login en HuggingFace")
            print("   Ejecuta: huggingface-cli login")
            return
        
        # Confirmar
        confirm = input(f"Descargar {filename} ({size_mb}MB) a {LORA_DIR}? (s/n): ").lower()
        if confirm != 's':
            print("Cancelado.")
            return
        
        #Descargar
        print(f"\n[DOWNLOAD] Iniciando descarga...")
        if logged_in:
            success, msg = download_with_huggingface_cli(repo_id, filename, LORA_DIR)
        else:
            success, msg = download_with_requests(repo_id, filename, LORA_DIR)
        
        if success:
            print(f"\n[OK] {msg}")
            print(f"[INFO] Ahora el LoRA está disponible en:")
            print(f"  {os.path.join(LORA_DIR, filename)}")
            print("\n[INFO] Para usarlo:")
            print("  1. Reinicia la aplicación")
            print("  2. En Image Editor, selecciona motor 'flux_klein'")
            print("  3. El sistema detectará automáticamente el LoRA")
        else:
            print(f"\n[ERROR] {msg}")
            
    except KeyboardInterrupt:
        print("\nCancelado.")
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    main()
