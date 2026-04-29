import os
import threading
import numpy as np
import insightface
import cv2
import roop.globals
from typing import Any
from roop.types import Face, Frame


def detect_mouth_open(target_face: Face, landmarks_106=None, target_img=None):
    """Detecta si la boca está abierta usando landmarks 2D o 106"""
    try:
        if landmarks_106 is not None:
            top_lip = landmarks_106[52]
            bottom_lip = landmarks_106[58]
            dist = np.linalg.norm(top_lip - bottom_lip)
            eye_dist = np.linalg.norm(landmarks_106[35] - landmarks_106[93])
            ratio = dist / eye_dist
            return ratio > 0.15, {'top': top_lip, 'bottom': bottom_lip}, ratio
        
        if hasattr(target_face, 'kps') and target_face.kps is not None:
            kps = target_face.kps
            dist = np.linalg.norm(kps[3] - kps[4])
            eye_dist = np.linalg.norm(kps[0] - kps[1])
            ratio = dist / eye_dist
            return ratio > 0.45, None, ratio
            
        return False, None, 0.0
    except:
        return False, None, 0.0


class FaceSwap:
    def __init__(self) -> None:
        self.model = None
        self.model_path = None
        self.devicename = None

    def Initialize(self, options: dict) -> None:
        devnm = options.get('devicename', 'cpu')
        model_path = options.get('model', 'inswapper_128.onnx')
        
        if not os.path.isabs(model_path):
            model_path = os.path.abspath(os.path.join(os.getcwd(), 'models', model_path))

        if self.model is None or self.model_path != model_path or self.devicename != devnm:
            self.model_path = model_path
            self.devicename = devnm
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'cuda' in devnm else ['CPUExecutionProvider']
            
            print(f"[FaceSwap] Buscando modelo en: {model_path}")
            print(f"[FaceSwap] ¿Existe el modelo?: {os.path.exists(model_path)}")
            
            try:
                self.model = insightface.model_zoo.get_model(model_path, providers=providers)
                print(f"[FaceSwap] Modelo cargado correctamente con providers: {providers}")
            except Exception as e:
                print(f"[FaceSwap] Error cargando modelo: {e}")

    def Run(self, source_face: Face, target_face: Face, temp_frame: Frame, paste_back: bool = True) -> Any:
        if self.model is None: return None
        try:
            if not isinstance(temp_frame, np.ndarray): temp_frame = np.array(temp_frame)
            
            # Ejecutar el swap (sin pegado automático)
            swapped_face, M = self.model.get(temp_frame, target_face, source_face, paste_back=False)
            if swapped_face is None: return None
            
            if not paste_back: return swapped_face
            
            # Pegado mejorado
            return self.paste_back_improved(temp_frame, swapped_face, M, target_face)
        except Exception as e:
            print(f"[ERROR] FaceSwap Run: {e}")
            return None

    @staticmethod
    def paste_back_improved(target_img, source_face_img, M, target_face=None):
        """Pegado de alta precisión con máscara sólida garantizada."""
        try:
            if M is None or source_face_img is None: return target_img
            
            # Dimensiones del frame destino
            h, w = target_img.shape[:2]

            # 1. Calcular matriz de retorno
            M_inv = cv2.invertAffineTransform(M)

            # 2. Warp de la cara swappeada
            warped_face = cv2.warpAffine(source_face_img, M_inv, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)

            # 3. GENERACIÓN DE MÁSCARA SÓLIDA (128x128 -> Frame)
            mask_128 = np.zeros((128, 128), dtype=np.float32)
            cv2.ellipse(mask_128, (64, 64), (52, 70), 0, 0, 360, 1.0, -1)
            mask_128 = cv2.GaussianBlur(mask_128, (7, 7), 0) # Suavizado ligero de bordes
            
            # Proyectar máscara al frame
            face_mask = cv2.warpAffine(mask_128, M_inv, (w, h), flags=cv2.INTER_LINEAR)
            
            # Asegurar opacidad en el centro (fuerza 100%)
            face_mask = np.clip(face_mask * 2.0, 0, 1)
            if face_mask.ndim == 2: face_mask = np.expand_dims(face_mask, axis=2)

            # 4. Mezclar (Blending)
            blend_ratio = getattr(roop.globals, 'blend_ratio', 0.98)
            target_float = target_img.astype(np.float32)
            warped_float = warped_face.astype(np.float32)
            
            alpha = face_mask * blend_ratio
            result = target_float * (1.0 - alpha) + warped_float * alpha
            
            print(f"[DEBUG] FaceSwap OK - Visibilidad: {alpha.max():.2f}")
            return np.clip(result, 0, 255).astype(np.uint8)

        except Exception as e:
            print(f"[ERROR] paste_back_improved: {e}")
            return target_img

    def Release(self):
        self.model = None
