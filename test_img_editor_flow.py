#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test del flujo completo de ImgEditor con dos pasadas
"""

import os
import sys

# Agregar el directorio del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_face_swap_init():
    """Prueba la inicialización del face swap"""
    print("\n=== Test Face Swap Init ===")
    
    try:
        from roop.processors.FaceSwap import FaceSwap
        print("[OK] FaceSwap importado")
        
        swapper = FaceSwap()
        print("[OK] FaceSwap instanciado")
        
        swapper.Initialize({"devicename": "cuda"})
        print("[OK] FaceSwap inicializado")
        
        return True
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_insightface_init():
    """Prueba la inicialización de insightface"""
    print("\n=== Test InsightFace Init ===")
    
    try:
        from insightface.app import FaceAnalysis
        print("[OK] FaceAnalysis importado")
        
        analyzer = FaceAnalysis()
        print("[OK] FaceAnalysis instanciado")
        
        analyzer.prepare(ctx_id=0, det_size=(640, 640))
        print("[OK] FaceAnalysis preparado")
        
        return True
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_img_editor_manager():
    """Prueba el manager de ImgEditor"""
    print("\n=== Test ImgEditor Manager ===")
    
    try:
        from roop.img_editor.img_editor_manager import get_img_editor_manager
        print("[OK] ImgEditorManager importado")
        
        manager = get_img_editor_manager()
        print("[OK] Manager instanciado")
        
        # Verificar inicialización de face swap
        success = manager._init_face_swap()
        if success:
            print("[OK] Face swap inicializado en manager")
        else:
            print("[WARN] Face swap no pudo inicializarse")
        
        return True
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comfy_connection():
    """Prueba la conexión con ComfyUI"""
    print("\n=== Test ComfyUI Connection ===")
    
    try:
        import requests
        
        # Intentar conectar a ComfyUI
        ports = [8188, 8189]
        for port in ports:
            try:
                response = requests.get(f"http://127.0.0.1:{port}/system_stats", timeout=2)
                if response.status_code == 200:
                    print(f"[OK] ComfyUI corriendo en puerto {port}")
                    return True
            except:
                continue
        
        print("[WARN] ComfyUI no esta corriendo")
        print("       Inicia ComfyUI con: python ui/tob/ComfyUI/main.py --port 8188")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def main():
    print("=" * 60)
    print("TEST DE IMG EDITOR - FLUJO DE DOS PASADAS")
    print("=" * 60)
    
    results = []
    
    # Test 1: Face Swap
    results.append(("Face Swap Init", test_face_swap_init()))
    
    # Test 2: InsightFace
    results.append(("InsightFace Init", test_insightface_init()))
    
    # Test 3: ImgEditor Manager
    results.append(("ImgEditor Manager", test_img_editor_manager()))
    
    # Test 4: ComfyUI Connection
    results.append(("ComfyUI Connection", test_comfy_connection()))
    
    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    
    all_ok = True
    for name, success in results:
        status = "[OK]" if success else "[FAIL]"
        print(f"{status} {name}")
        if not success:
            all_ok = False
    
    print("=" * 60)
    
    if all_ok:
        print("\nTODO OK - El sistema esta listo para usar")
    else:
        print("\nHAY ERRORES - Revisa los mensajes arriba")
    
    return all_ok


if __name__ == "__main__":
    main()
