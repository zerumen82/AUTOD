#!/usr/bin/env python3
import sys
import os

filepath = r"D:\PROJECTS\AUTOAUTO\roop\ProcessMgr.py"

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# Find and replace the specific section
old_code = """print(
                            f"[ALIGN_REJECT] Frame {call_num}: recuperación con cara guardada "
                            f"(eye_rel={eye_rel_c:.3f})",
                            flush=True,
                        )
                        return self._process_face_swap_v21(
                            source_face, cand, result_frame, original_frame,
                            person_idx=0, photo_path=photo_path,
                        )"""

new_code = """print(
                            f"[ALIGN_REJECT] Frame {call_num}: recuperación con cara guardada "
                            f"(eye_rel={eye_rel_c:.3f})",
                            flush=True,
                        )
                        # Usar Hyperswap 256 como fallback cuando kps son degenerados (eye_rel <= 0.05)
                        # Esto evita warp deformado y mejora similitud en casos difíciles
                        if use_hyperswap_fallback:
                            # Cargar Hyperswap 256 bajo demanda
                            hyperswap_256 = _get_hyperswap_256()
                            if hyperswap_256 is not None:
                                try:
                                    # Ejecutar swap con Hyperswap 256
                                    result = _hyperswap_256_get_result(
                                        hyperswap_256, source_face, cand, result_frame, original_frame
                                    )
                                    if result is not None:
                                        return result
                                    print("[HYPERSWAP_FALLBACK] Hyperswap 256 falló, intentando normal")
                                except Exception as e:
                                    print(f"[HYPERSWAP_FALLBACK] Error: {e}, intentando normal")
                        return self._process_face_swap_v21(
                            source_face, cand, result_frame, original_frame,
                            person_idx=0, photo_path=photo_path,
                        )"""

if old_code in content:
    content = content.replace(old_code, new_code)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print("Reemplazo exitoso")
else:
    print("No se encontró el código antiguo")
    sys.exit(1)