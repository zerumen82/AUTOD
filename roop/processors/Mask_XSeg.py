import threading
import os
import cv2
import numpy as np
import onnxruntime

import roop.globals
from roop.types import Frame
from roop.utilities import resolve_relative_path

THREAD_LOCK_CLIP = threading.Lock()


class Mask_XSeg:
    """XSeg Masker - segmentación facial de alta precisión (v5.4)"""

    def __init__(self):
        self.model_xseg = None
        self.devicename = None

    def Initialize(self, params: dict):
        self.devicename = params.get("devicename", "cpu")
        model_path = resolve_relative_path("../models/xseg.onnx")

        if not os.path.exists(model_path):
            print(f"[ERROR] Modelo XSeg no encontrado en: {model_path}")
            return

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if self.devicename == "cpu":
            providers = ["CPUExecutionProvider"]

        try:
            self.model_xseg = onnxruntime.InferenceSession(model_path, providers=providers)
        except Exception as e:
            print(f"[ERROR] Error cargando modelo XSeg: {e}")

    def Run(self, img: Frame, masking_text: str = "") -> np.ndarray:
        """
        Genera una máscara de segmentación facial de 256x256.
        """
        if self.model_xseg is None or img is None:
            # Si no hay modelo, devolver máscara blanca parcial (elipse genérica)
            if img is not None:
                h, w = img.shape[:2]
                mask = np.zeros((h, w), dtype=np.float32)
                cv2.ellipse(mask, (w // 2, h // 2), (int(w * 0.45), int(h * 0.50)), 0, 0, 360, 1.0, -1)
                return mask
            return np.zeros((256, 256), dtype=np.float32)

        validated_img = None
        try:
            # Asegurar tamaño y formato para el modelo (256x256)
            if isinstance(img, np.ndarray):
                validated_img = img.copy()
            else:
                return np.zeros((256, 256), dtype=np.float32)

            # Redimensionar con validación
            temp_frame = cv2.resize(validated_img, (256, 256), cv2.INTER_CUBIC)

            if temp_frame is None or temp_frame.size == 0:
                print("[ERROR] Mask_XSeg: resize devolvió imagen vacía")
                height, width = validated_img.shape[:2]
                return np.zeros((height, width), dtype=np.float32)

            # Normalizar: [0, 255] -> [-1, 1]
            blob = temp_frame.astype(np.float32) / 127.5 - 1.0
            
            # v5.6.3: Asegurar forma NHWC [1, 256, 256, 3] para xseg.onnx
            blob = np.expand_dims(blob, axis=0)
            if blob.shape != (1, 256, 256, 3):
                blob = blob.reshape(1, 256, 256, 3)

            # Inferencia
            inputs = {self.model_xseg.get_inputs()[0].name: blob}
            outputs = self.model_xseg.run(None, inputs)
            
            # Post-procesamiento
            mask = outputs[0][0][0]  # El modelo devuelve [1, 1, 256, 256]
            
            # Limpieza de máscara: Sigmoide manual simplificada
            mask = np.clip(mask, 0, 1.0)
            
            # Redimensionar máscara al tamaño original de la cara alineada
            height, width = validated_img.shape[:2]
            result = cv2.resize(mask, (width, height), cv2.INTER_LINEAR)
            
            # Refinar bordes
            result[result < 0.05] = 0
            
            # NO INVERTIR - mantener como máscara de la cara directamente
            # La máscara debe indicar dónde está la cara (1) y dónde no (0)
            
            return result

        except Exception as e:
            print(f"[ERROR] Mask_XSeg: error en procesamiento: {str(e)}")
            # En caso de error, devolver máscara negra del tamaño correcto
            if validated_img is not None:
                height, width = validated_img.shape[:2]
            else:
                height, width = 256, 256
            return np.zeros((height, width), dtype=np.float32)

    def Release(self):
        if self.model_xseg:
            del self.model_xseg
            self.model_xseg = None
