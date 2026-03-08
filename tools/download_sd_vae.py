"""
Script para descargar e instalar VAE de alta calidad para Stable Diffusion
===========================================================================

Este script descarga el VAE 'vae-ft-mse-840000.safetensors' que mejora significativamente
la calidad de las imagenes generadas.

El VAE (Variational Autoencoder) es responsable de decodificar los latentes
en imagenes visibles. Un VAE de alta calidad produce:
- Colores mas precisos
- Mas detalles finos
- Menos artefactos
- Texturas mas suaves
"""

import os
import urllib.request
import ssl

# Configuracion
VAE_URL = "https://huggingface.co/stabilityai/sd-vae/resolve/main/vae-ft-mse-840000.safetensors"
VAE_OUTPUT_DIR = "ui/tob/stable-diffusion-webui/models/VAE"
VAE_FILENAME = "vae-ft-mse-840000.safetensors"
VAE_OUTPUT_PATH = os.path.join(VAE_OUTPUT_DIR, VAE_FILENAME)

def download_vae():
    """Descarga el VAE de alta calidad"""
    
    # Crear directorio si no existe
    os.makedirs(VAE_OUTPUT_DIR, exist_ok=True)
    
    # Verificar si ya existe
    if os.path.exists(VAE_OUTPUT_PATH):
        file_size = os.path.getsize(VAE_OUTPUT_PATH)
        print(f"[OK] VAE ya existe: {VAE_OUTPUT_PATH}")
        print(f"  Tamano: {file_size / (1024*1024):.2f} MB")
        return True
    
    print(f"[DOWNLOAD] Descargando VAE de alta calidad...")
    print(f"   URL: {VAE_URL}")
    print(f"   Destino: {VAE_OUTPUT_PATH}")
    print()
    
    # Crear contexto SSL que no verifica certificados (para evitar errores)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    try:
        # Descargar con urllib
        req = urllib.request.Request(
            VAE_URL,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )
        
        with urllib.request.urlopen(req, context=ssl_context, timeout=300) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            print(f"   Tamano total: {total_size / (1024*1024):.2f} MB")
            print()
            
            downloaded = 0
            block_size = 8192
            
            with open(VAE_OUTPUT_PATH, 'wb') as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    
                    downloaded += len(buffer)
                    f.write(buffer)
                    
                    # Mostrar progreso
                    percent = min(100, (downloaded / total_size) * 100) if total_size > 0 else 0
                    print(f"\r   Progreso: {percent:.1f}% ({downloaded / (1024*1024):.2f} MB)", end="")
        
        print()  # Nueva linea al final
        print(f"\n[OK] VAE descargado exitosamente!")
        print(f"   Ruta: {VAE_OUTPUT_PATH}")
        
        # Verificar tamano
        file_size = os.path.getsize(VAE_OUTPUT_PATH)
        print(f"   Tamano: {file_size / (1024*1024):.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Error descargando VAE: {e}")
        # Limpiar archivo parcial si existe
        if os.path.exists(VAE_OUTPUT_PATH):
            os.remove(VAE_OUTPUT_PATH)
        return False

def update_sd_config():
    """Actualiza el config.json de SD para usar el nuevo VAE"""
    config_path = "ui/tob/stable-diffusion-webui/config.json"
    
    if not os.path.exists(config_path):
        print("[ERROR] No se encontro config.json")
        return False
    
    import json
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Actualizar configuracion del VAE
    old_vae = config.get('sd_vae', 'Automatic')
    config['sd_vae'] = VAE_FILENAME
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    print(f"\n[OK] Configuracion actualizada: {old_vae} -> {VAE_FILENAME}")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Downloader de VAE de Alta Calidad para Stable Diffusion")
    print("=" * 60)
    print()
    
    success = download_vae()
    
    if success:
        print()
        update_sd_config()
        print()
        print("[OK] Instalacion completada!")
        print()
        print("NOTA: Reinicia Stable Diffusion para que use el nuevo VAE")
    else:
        print()
        print("[ERROR] Fallo la descarga. Intenta manualmente:")
        print(f"   1. Descarga: {VAE_URL}")
        print(f"   2. Guarda en: {VAE_OUTPUT_PATH}")
