import threading

import cv2
import numpy as np
import onnxruntime

import roop.globals
from roop.types import Frame
from roop.utilities import resolve_relative_path

THREAD_LOCK_CLIP = threading.Lock()


class Mask_XSeg:
    plugin_options: dict = None

    model_xseg = None

    processorname = "mask_xseg"
    type = "mask"

    def Initialize(self, plugin_options: dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.model_xseg is None:
            model_path = resolve_relative_path("../models/xseg.onnx")
            onnxruntime.set_default_logger_severity(3)
            self.model_xseg = onnxruntime.InferenceSession(
                model_path, None, providers=roop.globals.execution_providers
            )
            self.model_inputs = self.model_xseg.get_inputs()
            self.model_outputs = self.model_xseg.get_outputs()

            # replace Mac mps with cpu for the moment
            self.devicename = self.plugin_options["devicename"].replace("mps", "cpu")

    def validate_image(self, img):
        """
        CORRECCIÓN: Validación robusta de imagen antes del procesamiento
        """
        if img is None:
            return False, None

        if not isinstance(img, np.ndarray):
            return False, None

        if img.size == 0:
            return False, None

        if len(img.shape) < 2:
            return False, None

        # Verificar dimensiones válidas
        height, width = img.shape[:2]
        if width <= 0 or height <= 0:
            return False, None

        if width < 8 or height < 8:  # Imagen demasiado pequeña
            return False, None

        return True, img

    def Run(self, img1, keywords: str) -> Frame:
        """
        CORRECCIÓN: Validación exhaustiva antes del resize para evitar el error de OpenCV
        """
        # Validar imagen de entrada
        is_valid, validated_img = self.validate_image(img1)
        if not is_valid:
            print(
                f"[ERROR] Mask_XSeg: imagen inválida para procesamiento - shape: {img1.shape if img1 is not None else 'None'}"
            )
            # Crear una máscara por defecto (imagen negra del tamaño esperado)
            if img1 is not None and hasattr(img1, "shape"):
                height, width = img1.shape[:2]
            else:
                height, width = 256, 256
            return np.zeros((height, width), dtype=np.float32)

        try:
            # CORRECCIÓN: Redimensionar con validación
            print(
                f"[INFO] Mask_XSeg: redimensionando imagen de {validated_img.shape} a (256, 256)"
            )
            temp_frame = cv2.resize(validated_img, (256, 256), cv2.INTER_CUBIC)

            if temp_frame is None or temp_frame.size == 0:
                print("[ERROR] Mask_XSeg: resize devolvió imagen vacía")
                height, width = validated_img.shape[:2]
                return np.zeros((height, width), dtype=np.float32)

            temp_frame = temp_frame.astype("float32") / 255.0
            temp_frame = temp_frame[None, ...]

            io_binding = self.model_xseg.io_binding()
            io_binding.bind_cpu_input(self.model_inputs[0].name, temp_frame)
            io_binding.bind_output(self.model_outputs[0].name, self.devicename)
            self.model_xseg.run_with_iobinding(io_binding)
            ort_outs = io_binding.copy_outputs_to_cpu()
            result = ort_outs[0][0]
            result = np.clip(result, 0, 1.0)
            # SOLUCIÓN DEFINITIVA: Máscara directa sin inversión incorrecta
            # Umbral optimizado para capturar área facial correcta
            threshold = 0.3  # Umbral balanceado para área facial
            result[result < threshold] = 0
            
            # Suavizar la máscara para transiciones naturales
            result = cv2.GaussianBlur(result, (5, 5), 0)
            
            # Normalizar máscara a rango [0, 1]
            if np.max(result) > 0:
                result = result / np.max(result)
            
            # Eliminar píxeles muy débiles para evitar ruido
            result[result < 0.1] = 0
            
            # NO INVERTIR - mantener como máscara de la cara directamente
            # La máscara debe indicar dónde está la cara (1) y dónde no (0)
            
            print(f"[DEBUG] Mask_XSeg: máscara final - min={np.min(result):.3f}, max={np.max(result):.3f}, mean={np.mean(result):.3f}")
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
        del self.model_xseg
        self.model_xseg = None
