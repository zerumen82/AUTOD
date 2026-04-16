#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ControlNet Utils - Utilidades para ControlNet en SD Editor

Soporta:
- OpenPose: Detección y generación de poses
- Tile: Upscale y mejora de detalles
- Depth: Mapas de profundidad
- Canny: Detección de bordes
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Dict, Any


class ControlNetUtils:
    """Utilidades para ControlNet"""
    
    def __init__(self):
        self.openpose_model = None
        self.tile_model = None
        self.comfy_client = None
        
    def init_comfy_client(self):
        """Inicializa cliente de ComfyUI"""
        if self.comfy_client is None:
            from roop.comfy_client import ComfyClient
            self.comfy_client = ComfyClient()
        return self.comfy_client
    
    def detect_pose(self, image: Image.Image) -> Optional[np.ndarray]:
        """
        Detecta pose de una imagen usando OpenPose.
        
        Returns:
            pose_image: Imagen de la pose detectada (o None si falla)
        """
        try:
            # Intentar con OpenCV DNN (más ligero, no requiere instalación extra)
            return self._detect_pose_opencv(image)
        except Exception as e:
            print(f"[ControlNet] Error detectando pose: {e}")
            return None
    
    def _detect_pose_opencv(self, image: Image.Image) -> Optional[np.ndarray]:
        """Detecta pose usando OpenCV DNN"""
        try:
            # Convertir a numpy array
            img_array = np.array(image)
            if img_array.ndim == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
            height, width = img_array.shape[:2]
            
            # Crear lienzo negro para la pose
            pose_canvas = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Detectar puntos clave del cuerpo con OpenCV
            # Usar modelo pre-entrenado de pose detection
            try:
                # Intentar cargar modelo de pose de OpenCV
                # Si no está disponible, usar detección básica
                pose_points = self._detect_body_points(img_array)
                
                # Dibujar esqueleto
                self._draw_skeleton(pose_canvas, pose_points)
                
            except Exception as e:
                print(f"[ControlNet] Usando detección básica: {e}")
                # Fallback: devolver imagen vacía (ComfyUI generará sin ControlNet)
                return None
            
            return pose_canvas
            
        except Exception as e:
            print(f"[ControlNet] Error en detección OpenCV: {e}")
            return None
    
    def _detect_body_points(self, image: np.ndarray) -> Dict[str, Tuple[int, int]]:
        """
        Detecta puntos clave del cuerpo.
        
        Returns:
            Dict con nombres de puntos y sus coordenadas (x, y)
        """
        height, width = image.shape[:2]
        points = {}
        
        # Detección básica usando proporciones del cuerpo
        # Esto es un fallback - idealmente usar MediaPipe o OpenPose real
        
        try:
            # Intentar con MediaPipe Pose si está disponible
            import mediapipe as mp
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(static_image_mode=True)
            
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Extraer puntos clave
                key_points = {
                    'nose': mp_pose.PoseLandmark.NOSE,
                    'left_shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER,
                    'right_shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    'left_elbow': mp_pose.PoseLandmark.LEFT_ELBOW,
                    'right_elbow': mp_pose.PoseLandmark.RIGHT_ELBOW,
                    'left_wrist': mp_pose.PoseLandmark.LEFT_WRIST,
                    'right_wrist': mp_pose.PoseLandmark.RIGHT_WRIST,
                    'left_hip': mp_pose.PoseLandmark.LEFT_HIP,
                    'right_hip': mp_pose.PoseLandmark.RIGHT_HIP,
                    'left_knee': mp_pose.PoseLandmark.LEFT_KNEE,
                    'right_knee': mp_pose.PoseLandmark.RIGHT_KNEE,
                    'left_ankle': mp_pose.PoseLandmark.LEFT_ANKLE,
                    'right_ankle': mp_pose.PoseLandmark.RIGHT_ANKLE,
                }
                
                for name, landmark in key_points.items():
                    lm = landmarks[landmark.value]
                    x = int(lm.x * width)
                    y = int(lm.y * height)
                    if 0 <= x < width and 0 <= y < height:
                        points[name] = (x, y)
                
            pose.close()
            
        except ImportError:
            print("[ControlNet] MediaPipe no disponible, usando detección básica")
            # Fallback: puntos estimados
            points = self._estimate_body_points(height, width)
        except Exception as e:
            print(f"[ControlNet] Error en MediaPipe: {e}")
            points = self._estimate_body_points(height, width)
        
        return points
    
    def _estimate_body_points(self, height: int, width: int) -> Dict[str, Tuple[int, int]]:
        """Estima puntos del cuerpo (fallback)"""
        cx, cy = width // 2, height // 2
        
        # Puntos estimados para una persona de pie
        return {
            'nose': (cx, cy - height // 4),
            'left_shoulder': (cx - width // 6, cy - height // 6),
            'right_shoulder': (cx + width // 6, cy - height // 6),
            'left_hip': (cx - width // 8, cy),
            'right_hip': (cx + width // 8, cy),
            'left_knee': (cx - width // 10, cy + height // 4),
            'right_knee': (cx + width // 10, cy + height // 4),
        }
    
    def _draw_skeleton(self, canvas: np.ndarray, points: Dict[str, Tuple[int, int]]):
        """Dibuja esqueleto en el canvas"""
        # Conexiones del esqueleto
        connections = [
            ('nose', 'left_shoulder'),
            ('nose', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle'),
            ('left_hip', 'right_hip'),
        ]
        
        # Dibujar líneas
        for pt1_name, pt2_name in connections:
            if pt1_name in points and pt2_name in points:
                pt1 = points[pt1_name]
                pt2 = points[pt2_name]
                cv2.line(canvas, pt1, pt2, (255, 255, 0), 3)
        
        # Dibujar puntos
        for name, point in points.items():
            cv2.circle(canvas, point, 5, (0, 255, 255), -1)
    
    def enhance_tile(self, image: Image.Image, scale: int = 4) -> Image.Image:
        """
        Mejora imagen usando ControlNet Tile.
        
        Args:
            image: Imagen original
            scale: Factor de upscale (2, 4, 8)
            
        Returns:
            Imagen mejorada con upscale
        """
        # Nota: El upscale real se hace en ComfyUI workflow
        # Esta función prepara la imagen para el workflow
        return image
    
    def create_canny_edges(self, image: Image.Image, 
                          low_threshold: int = 100,
                          high_threshold: int = 200) -> Image.Image:
        """
        Crea mapa de bordes Canny.
        
        Returns:
            Imagen con bordes detectados
        """
        img_array = np.array(image.convert('RGB'))
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(img_gray, low_threshold, high_threshold)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(edges_colored)
    
    def create_depth_map(self, image: Image.Image) -> Optional[Image.Image]:
        """
        Crea mapa de profundidad estimado.
        
        Returns:
            Mapa de profundidad (o None si falla)
        """
        try:
            # Intentar con MiDaS si está disponible
            import torch
            # ... implementación MiDaS ...
            pass
        except ImportError:
            print("[ControlNet] MiDaS no disponible, usando profundidad estimada")
        
        # Fallback: profundidad estimada desde gradientes
        img_array = np.array(image.convert('L'))
        depth = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX)
        depth_colored = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        return Image.fromarray(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB))
    
    def check_controlnet_available(self, controlnet_type: str = "all") -> Dict[str, bool]:
        """
        Verifica qué ControlNets están disponibles en ComfyUI.
        
        Returns:
            Dict con disponibilidad de cada ControlNet
        """
        result = {
            "openpose": False,
            "tile": False,
            "depth": False,
            "canny": False,
        }
        
        try:
            client = self.init_comfy_client()
            object_info = client.get_object_info()
            
            if not object_info:
                # Si no hay object_info, verificar por archivos en disco
                return self._check_controlnet_files()
            
            # Buscar nodos de ControlNet
            for node_name, node_info in object_info.items():
                if "ControlNet" in node_name:
                    if "openpose" in node_name.lower():
                        result["openpose"] = True
                    elif "tile" in node_name.lower():
                        result["tile"] = True
                    elif "depth" in node_name.lower():
                        result["depth"] = True
                    elif "canny" in node_name.lower():
                        result["canny"] = True
            
        except Exception as e:
            print(f"[ControlNet] Error verificando disponibilidad: {e}")
            # Fallback a verificación por archivos
            return self._check_controlnet_files()
        
        return result
    
    def _check_controlnet_files(self) -> Dict[str, bool]:
        """Verifica ControlNets por archivos en disco"""
        import os
        result = {"openpose": False, "tile": False, "depth": False, "canny": False}
        
        # Rutas posibles de ComfyUI
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "..", "..", "tob", "ComfyUI", "models", "controlnet"),
            os.path.join(os.path.expanduser("~"), "ComfyUI", "models", "controlnet"),
        ]
        
        for base_path in possible_paths:
            if os.path.exists(base_path):
                files = os.listdir(base_path)
                for f in files:
                    f_lower = f.lower()
                    if "openpose" in f_lower:
                        result["openpose"] = True
                    elif "tile" in f_lower:
                        result["tile"] = True
                    elif "depth" in f_lower:
                        result["depth"] = True
                    elif "canny" in f_lower:
                        result["canny"] = True
        
        return result


# Instancia global
_controlnet_utils = None

def get_controlnet_utils() -> ControlNetUtils:
    """Obtiene instancia global de ControlNetUtils"""
    global _controlnet_utils
    if _controlnet_utils is None:
        _controlnet_utils = ControlNetUtils()
    return _controlnet_utils
