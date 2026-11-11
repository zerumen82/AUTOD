# -*- coding: utf-8 -*-
"""Reproducir el 'name np is not defined' de la extracción de caras con traceback completo."""
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ui.tabs.faceswap import logic

photo = r"D:\PROJECTS\AUTOAUTO\img_editor\edit_1794405323.png"
if not os.path.exists(photo):
    import glob
    cands = glob.glob(r"D:\PROJECTS\AUTOAUTO\img_editor\edit_*.png")
    photo = cands[0] if cands else None
print("foto:", photo)

try:
    out = logic.extract_face_images(photo, (False, 0), target_face_detection=True, is_source_face=True)
    print("resultado:", len(out), "caras")
except Exception:
    traceback.print_exc()

# Cuerpo replicado del wrapper con traceback completo
try:
    import roop.face_util as fu
    faces_data = fu.extract_face_images(
        photo, (False, 0), target_face_detection=True, is_source_face=True
    )
    print("faces_data:", len(faces_data))
    enriched_data = []
    for face_obj, face_img in faces_data:
        if not hasattr(face_obj, 'gender') or face_obj.gender is None:
            from roop.ProcessMgr import get_gender
            face_obj.gender = get_gender(face_obj)
        if hasattr(face_obj, 'bbox'):
            w = face_obj.bbox[2] - face_obj.bbox[0]
            h = face_obj.bbox[3] - face_obj.bbox[1]
            face_obj.area = w * h
        enriched_data.append((face_obj, face_img))
    print("enriquecidas:", len(enriched_data))
except Exception:
    traceback.print_exc()

# Llamada directa al módulo real con traceback completo
try:
    faces = __import__("roop.face_util", fromlist=["extract_face_images"]).extract_face_images(
        photo, (False, 0), target_face_detection=True, is_source_face=True
    )
    print("face_util directo:", len(faces), "caras")
except Exception:
    traceback.print_exc()
