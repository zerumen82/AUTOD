#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descarga el modelo FLUX.1-Fill-dev NF4 completo desde HuggingFace
Requiere: pip install huggingface-hub
"""

import os
import sys

MODEL_DIR = r"D:\PROJECTS\models"
FLUX_NF4_DIR = os.path.join(MODEL_DIR, "FLUX.1-fill-dev-NF4")

# Token HF (reemplazar con el tuyo)
HF_TOKEN = ""  # <-- PON TU TOKEN AQUI

print("=" * 60)
print("DESCARGA DE FLUX.1-Fill-dev NF4")
print("=" * 60)
print()
print("El modelo NF4 actual está INCOMPLETO.")
print("Este script descargará los archivos faltantes desde HuggingFace.")
print()
print("Requisitos:")
print("  - Conexión a internet")
print("  - ~12 GB de espacio en disco")
print("  - Token de HuggingFace (gratis)")
print()

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("\n❌ Error: huggingface-hub no está instalado.")
    print("   Ejecuta: pip install huggingface-hub")
    input("\nPresiona Enter para salir...")
    sys.exit(1)

# Si no hay token, pedirlo
if not HF_TOKEN:
    print("\n" + "=" * 60)
    print("Ingresa tu token de HuggingFace:")
    print("  - Obtén tu token en: https://huggingface.co/settings/tokens")
    print("=" * 60)
    HF_TOKEN = input("Token HF: ").strip()

if not HF_TOKEN:
    print("\n❌ Error: Se requiere token de HuggingFace.")
    input("\nPresiona Enter para salir...")
    sys.exit(1)

print("\n" + "=" * 60)
print("Iniciando descarga...")
print("=" * 60)

try:
    # Descargar modelo completo
    print("\nDescargando FLUX.1-Fill-dev NF4...")
    print("Ubicación: " + FLUX_NF4_DIR)
    print()
    
    downloaded_path = snapshot_download(
        repo_id="black-forest-labs/FLUX.1-Fill-dev",
        local_dir=FLUX_NF4_DIR,
        local_dir_use_symlinks=False,
        token=HF_TOKEN,
        ignore_patterns=["*.pt", "*.pth"],  # Ignorar formatos antiguos
    )
    
    print("\n" + "=" * 60)
    print("✅ DESCARGA COMPLETADA")
    print("=" * 60)
    print(f"\nModelo descargado en: {downloaded_path}")
    print("\nArchivos descargados:")
    
    # Listar archivos descargados
    for root, dirs, files in os.walk(FLUX_NF4_DIR):
        level = root.replace(FLUX_NF4_DIR, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files[:10]:  # Mostrar solo primeros 10 archivos
            print(f'{subindent}{file}')
    
    print("\n" + "=" * 60)
    print("¡Listo! Reinicia AUTOAUTO para usar el nuevo modelo.")
    print("=" * 60)
    
except Exception as e:
    print("\n" + "=" * 60)
    print(f"❌ ERROR DURANTE LA DESCARGA: {e}")
    print("=" * 60)
    print("\nPosibles causas:")
    print("  1. Token de HuggingFace inválido")
    print("  2. Sin conexión a internet")
    print("  3. Espacio en disco insuficiente")
    print("\nIntenta nuevamente o usa el modelo QUAN que ya está completo.")

input("\nPresiona Enter para salir...")
