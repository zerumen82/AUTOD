#!/usr/bin/env python3
"""
install_enhancer_models.py
Instalación automática de modelos Face Enhancement (2025)

Descarga e instala automáticamente CodeFormer (mejor modelo 2025)
sin interacción del usuario.
"""

import os
import sys
import urllib.request
from pathlib import Path

# Modelo recomendado por defecto
DEFAULT_MODEL = {
    "name": "CodeFormer",
    "url": "https://huggingface.co/facefusion/models-3.0.0/resolve/main/codeformer.onnx",
    "filename": "CodeFormerv0.1.onnx",
    "subfolder": "CodeFormer",
    "size_mb": 386,
    "description": "Mejor enhancer 2025 - Máxima calidad y preservación de identidad"
}


def get_models_folder():
    """Obtiene la carpeta de modelos relativa al script"""
    script_dir = Path(__file__).parent
    models_dir = script_dir / "roop" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def download_file(url, dest_path):
    """Descarga un archivo con barra de progreso"""
    print(f"\n📥 Descargando: {os.path.basename(dest_path)}")
    print(f"   URL: {url}")
    
    try:
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # Configurar request con user-agent y headers para HuggingFace
        req = urllib.request.Request(
            url,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': '*/*',
            }
        )
        
        # Intentar con redirect automático
        import ssl
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        
        # Descargar
        downloaded = 0
        block_size = 8192
        
        with urllib.request.urlopen(req, timeout=300, context=context) as response:
            # Seguir redirects manualmente si es necesario
            final_url = response.geturl()
            if final_url != url:
                print(f"   Redirect: {final_url}")
            
            total_size = int(response.info().get('Content-Length', 0))
            if total_size == 0:
                # Intentar obtener tamaño con HEAD request
                head_req = urllib.request.Request(url, method='HEAD')
                try:
                    with urllib.request.urlopen(head_req, timeout=30, context=context) as head_resp:
                        total_size = int(head_resp.info().get('Content-Length', 0))
                except:
                    pass
            
            if total_size > 0:
                print(f"   Tamaño: {total_size / (1024*1024):.1f} MB")
            
            with open(dest_path, 'wb') as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Barra de progreso
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        bar_length = 40
                        filled = int(bar_length * downloaded / total_size)
                        bar = '█' * filled + '░' * (bar_length - filled)
                        mb_downloaded = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        print(f"\r   [{bar}] {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='')
                    else:
                        mb_downloaded = downloaded / (1024 * 1024)
                        print(f"\r   Descargados: {mb_downloaded:.1f} MB", end='')
        
        print()  # Nueva línea después del progreso
        
        # Verificar tamaño
        actual_size = os.path.getsize(dest_path)
        if actual_size < 1000000:  # Menos de 1MB probablemente es error
            print(f"\n   ❌ Archivo demasiado pequeño: {actual_size} bytes")
            os.remove(dest_path)
            return False
        
        print(f"   ✅ Descarga completada: {dest_path}")
        print(f"   Tamaño final: {actual_size / (1024*1024):.1f} MB")
        return True
        
    except urllib.error.HTTPError as e:
        print(f"\n   ❌ Error HTTP {e.code}: {e.reason}")
        if e.code == 401:
            print("\n💡 El modelo requiere autenticación.")
            print("   Descarga manual desde HuggingFace:")
            print(f"   {url}")
            print("\n   O usa git-lfs:")
            print("   git lfs install")
            print("   git clone https://huggingface.co/sczhou/CodeFormer")
        return False
    except urllib.error.URLError as e:
        print(f"\n   ❌ Error de conexión: {e.reason}")
        return False
    except Exception as e:
        print(f"\n   ❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Instalación automática de CodeFormer"""
    print("=" * 60)
    print("   INSTALACIÓN AUTOMÁTICA - CODEFORMER (2025)")
    print("=" * 60)
    
    models_dir = get_models_folder()
    dest_dir = models_dir / DEFAULT_MODEL["subfolder"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / DEFAULT_MODEL["filename"]
    
    # Verificar si ya existe
    if dest_path.exists():
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"\n✅ CodeFormer ya está instalado:")
        print(f"   {dest_path} ({size_mb:.1f} MB)")
        print("\n💡 Si deseas reinstalar, elimina el archivo y ejecuta nuevamente.")
        return 0
    
    print(f"\n📦 MODELO: {DEFAULT_MODEL['name']}")
    print(f"   {DEFAULT_MODEL['description']}")
    print(f"   Tamaño: {DEFAULT_MODEL['size_mb']} MB")
    print(f"   Destino: {dest_path}")
    print("\n" + "=" * 60)
    print("🚀 INICIANDO DESCARGA...")
    print("=" * 60)
    
    # Descargar
    success = download_file(DEFAULT_MODEL["url"], str(dest_path))
    
    if success:
        print("\n" + "=" * 60)
        print("✅ ¡INSTALACIÓN COMPLETADA!")
        print("=" * 60)
        print(f"\n📁 Modelo instalado en:")
        print(f"   {dest_path}")
        print("\n💡 Para usar RestoreFormer++ como alternativa, ejecuta:")
        print(f"   python {os.path.basename(__file__)} --restoreformer")
    else:
        print("\n" + "=" * 60)
        print("❌ FALLÓ LA INSTALACIÓN")
        print("=" * 60)
        print("\nPosibles causas:")
        print("   1. Sin conexión a internet")
        print("   2. Firewall bloqueando la descarga")
        print("   3. Servidor temporalmente no disponible")
        print("\nIntenta nuevamente o descarga manualmente desde:")
        print(f"   {DEFAULT_MODEL['url']}")
        return 1
    
    return 0


if __name__ == "__main__":
    # Soporte para argumento --restoreformer
    if len(sys.argv) > 1 and sys.argv[1] == "--restoreformer":
        print("💡 Para descargar RestoreFormer++, usa el script interactivo:")
        print(f"   python download_enhancer_models.py")
        sys.exit(1)
    
    sys.exit(main())
