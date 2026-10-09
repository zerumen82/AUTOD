#!/usr/bin/env python3
import sys
import os

filepath = r"D:\PROJECTS\AUTOAUTO\roop\img_editor\top_tier_finish.py"

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# Find and replace the _edge_mask_from_image function
old_code = '''def _edge_mask_from_image(img: Image.Image, edge_thresh: float = 28.0, dilate_px: int = 3) -> Optional[np.ndarray]:
    \"\"\"v5.1134 — mapa de BORDES del sujeto (gradiente Sobel sobre luminancia).
    Devuelve máscara float 0..1 (1 = borde fuerte, con dilatación) o None si cv2 no está.
    \"\"\"\n    try:\n        import cv2 as _cv2\n        g = np.array(img.convert(\"L\"), dtype=np.float32)\n        gx = _cv2.Sobel(g, _cv2.CV_32F, 1, 0, ksize=3)\n        gy = _cv2.Sobel(g, _cv2.CV_32F, 0, 1, ksize=3)\n        mag = np.sqrt(gx * gx + gy * gy)\n        m = (mag > edge_thresh).astype(np.uint8) * 255\n        if dilate_px > 0:\n            k = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))\n            m = _cv2.dilate(m, k)\n        m = _cv2.GaussianBlur(m, (0, 0), 2.0)\n        return (m.astype(np.float32) / 255.0)\n    except Exception:\n        return None'''

new_code = '''def _edge_mask_from_image(img: Image.Image, edge_thresh: float = 28.0, dilate_px: int = 2) -> Optional[np.ndarray]:
    \"\"\"v5.1134 — mapa de BORDES del sujeto (gradiente Sobel sobre luminancia).
    Devuelve máscara float 0..1 (1 = borde fuerte, con dilatación) o None si cv2 no está.
    Se reduce dilate_px de 3 a 2 para evitar expansión excesiva de máscara en pelo
    que provoca franja gris translúcida y posible segunda cabeza (memory.md).
    \"\"\"\n    try:\n        import cv2 as _cv2\n        g = np.array(img.convert(\"L\"), dtype=np.float32)\n        gx = _cv2.Sobel(g, _cv2.CV_32F, 1, 0, ksize=3)\n        gy = _cv2.Sobel(g, _cv2.CV_32F, 0, 1, ksize=3)\n        mag = np.sqrt(gx * gx + gy * gy)\n        m = (mag > edge_thresh).astype(np.uint8) * 255\n        if dilate_px > 0:\n            # Usar kernel ligeramente más pequeño y dilatación más suave para hair regions\n            k = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))\n            m = _cv2.dilate(m, k, iterations=1)\n        m = _cv2.GaussianBlur(m, (0, 0), 2.0)\n        return (m.astype(np.float32) / 255.0)\n    except Exception:\n        return None'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print("Reemplazo exitoso en _edge_mask_from_image")
else:
    print("No se encontró el código antiguo en _edge_mask_from_image")
    # Try to find it
    if "def _edge_mask_from_image" in content:
        print("Función encontrada pero texto no coincide exactamente")
    else:
        print("Función no encontrada")
"