import os
import threading
import numpy as np
import insightface
import cv2
import roop.globals
from typing import Any
from roop.types import Face, Frame


def detect_mouth_open(target_face: Face, landmarks_106=None, target_image=None) -> tuple:
    """
    Detecta si la boca está abierta en la cara destino.
    USA MEDIAPIPE 468 LANDMARKS SI ESTÁ DISPONIBLE (95% precisión)
    Fallback a landmarks de 106 puntos o 5 keypoints básicos.
    
    Args:
        target_face: Objeto Face con bbox y kps
        landmarks_106: Landmarks de 106 puntos (opcional)
        target_image: Imagen completa del target (necesaria para MediaPipe)

    Returns:
        tuple: (is_open: bool, mouth_region: dict with landmarks, open_ratio: float)
    """
    try:
        # INTENTAR MEDIAPIPE PRIMERO (mejor precisión) - requiere imagen completa
        if target_image is not None:
            try:
                from roop.mouth_detector import detect_mouth_open_advanced
                is_open, open_ratio, mouth_data = detect_mouth_open_advanced(target_image)
                
                if mouth_data is not None:
                    # MediaPipe exitoso
                    mouth_region = {
                        'upper_lip': mouth_data.get('upper_lip_top'),
                        'lower_lip': mouth_data.get('lower_lip_bottom'),
                        'corners': [mouth_data.get('mouth_left'), mouth_data.get('mouth_right')],
                        'center': (
                            (mouth_data['mouth_left'][0] + mouth_data['mouth_right'][0]) // 2,
                            (mouth_data['upper_lip_top'][1] + mouth_data['lower_lip_bottom'][1]) // 2
                        ),
                        'height': mouth_data['mouth_height'],
                        'width': mouth_data['mouth_width'],
                        'open_ratio': open_ratio,
                        'is_open': is_open,
                        'method': 'mediapipe_468'  # Marcar que usó MediaPipe
                    }
                    
                    print(f"[MOUTH_DETECT] MediaPipe: boca {'abierta' if is_open else 'cerrada'} (ratio={open_ratio:.2f})")
                    return is_open, mouth_region, open_ratio
                    
            except ImportError:
                pass  # MediaPipe no disponible, usar fallback
            except Exception as e:
                print(f"[MOUTH_DETECT] MediaPipe falló: {e}, usando fallback")
        
        # FALLBACK: landmarks de 106 puntos
        mouth_open_threshold = 0.12  # Umbral reducido para mejor detección

        if landmarks_106 is not None and len(landmarks_106) >= 106:
            # Landmarks de boca en insightface 106:
            # Labio superior: 52-55, 62-65
            # Labio inferior: 58-61, 66-69
            # Comisuras: 55, 65 (izquierda), 61, 69 (derecha)

            upper_lip_top = landmarks_106[52]  # Punto superior del labio superior
            lower_lip_bottom = landmarks_106[58]  # Punto inferior del labio inferior
            left_mouth_corner = landmarks_106[55]  # Comisura izquierda
            right_mouth_corner = landmarks_106[61]  # Comisura derecha

            # Calcular altura de la boca
            mouth_height = abs(lower_lip_bottom[1] - upper_lip_top[1])
            mouth_width = abs(right_mouth_corner[0] - left_mouth_corner[0])

            if mouth_width > 0:
                open_ratio = mouth_height / mouth_width
            else:
                open_ratio = 0

            # Umbral más bajo para detectar bocas entreabiertas
            is_open = open_ratio > mouth_open_threshold

            mouth_region = {
                'upper_lip': landmarks_106[52:56],  # Puntos del labio superior
                'lower_lip': landmarks_106[58:62],  # Puntos del labio inferior
                'corners': [left_mouth_corner, right_mouth_corner],
                'center': ((left_mouth_corner[0] + right_mouth_corner[0]) / 2,
                          (upper_lip_top[1] + lower_lip_bottom[1]) / 2),
                'height': mouth_height,
                'width': mouth_width
            }

            return is_open, mouth_region, open_ratio

        # Fallback: usar los 5 keypoints básicos (kps)
        elif hasattr(target_face, 'kps') and target_face.kps is not None:
            kps = np.array(target_face.kps)
            if len(kps) >= 5:
                # kps[3] = comisura izquierda, kps[4] = comisura derecha
                left_mouth = kps[3]
                right_mouth = kps[4]

                # Estimar apertura basándose en la distancia entre comisuras
                # y la posición de la nariz
                nose = kps[2]

                mouth_width = abs(right_mouth[0] - left_mouth[0])
                mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
                nose_y = nose[1]

                # Si la boca está más abajo de lo normal respecto a la nariz,
                # puede indicar que está abierta
                expected_mouth_y = nose_y + mouth_width * 0.6
                actual_mouth_y = mouth_center_y

                # Distancia vertical normalizada
                vertical_diff = abs(actual_mouth_y - expected_mouth_y) / mouth_width if mouth_width > 0 else 0

                # Heurística MEJORADA: umbral más bajo para detectar bocas entreabiertas
                # También considerar apertura si la boca está significativamente abajo de la nariz
                is_open = vertical_diff > 0.08 or (mouth_width > 0 and vertical_diff > 0.05)

                mouth_region = {
                    'corners': [left_mouth, right_mouth],
                    'center': ((left_mouth[0] + right_mouth[0]) / 2,
                              (left_mouth[1] + right_mouth[1]) / 2),
                    'width': mouth_width
                }

                return is_open, mouth_region, vertical_diff

        return False, None, 0.0

    except Exception as e:
        print(f"[MOUTH_DETECT] Error: {e}")
        return False, None, 0.0


def create_mouth_preservation_mask(target_img: np.ndarray, mouth_region: dict,
                                    blend_ratio: float = 0.7) -> np.ndarray:
    """
    Crea una máscara para preservar la zona de la boca del destino.
    OPTIMIZADO: Mejor cobertura para bocas abiertas (comiendo, hablando)

    Args:
        target_img: Imagen original del destino
        mouth_region: Diccionario con los landmarks de la boca
        blend_ratio: Cuánto preservar (0.7 = 70% boca original, 30% swap)

    Returns:
        Máscara con valores de blending para la zona de la boca
    """
    try:
        h, w = target_img.shape[:2]
        mask = np.ones((h, w), dtype=np.float32)  # 1 = usar swap, 0 = usar original

        if mouth_region is None:
            return mask

        # Obtener centro y dimensiones de la boca
        center = mouth_region.get('center')
        mouth_width = mouth_region.get('width', 30)
        mouth_height = mouth_region.get('height', mouth_width * 0.5)

        if center is None:
            return mask

        cx, cy = int(center[0]), int(center[1])

        # OPTIMIZADO: Región más amplia para cubrir mejor la boca abierta
        # Más ancha que alta para cubrir toda la boca + zona alrededor
        radius_x = int(mouth_width * 1.0) if mouth_width else 30  # 100% del ancho para mejor cobertura
        radius_y = int(mouth_height * 2.0) if mouth_height else 25  # 200% del alto para boca abierta

        # Crear máscara elíptica suave para la boca
        mouth_mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mouth_mask, (cx, cy), (radius_x, radius_y), 0, 0, 360, 1.0, -1)

        # Aplicar blur para transición suave
        blur_size = int(max(7, min(radius_x, radius_y) // 2) | 1)  # Blur más suave
        mouth_mask = cv2.GaussianBlur(mouth_mask, (blur_size, blur_size), 0)
        
        # NORMALIZAR después del blur para evitar valores > 1
        max_val = mouth_mask.max()
        if max_val > 0:
            mouth_mask = mouth_mask / max_val

        # Invertir la máscara: donde hay boca, reducir el swap
        # mouth_mask = 1 en la boca, 0 fuera
        # Queremos: mask = 1 - mouth_mask * blend_ratio
        # Así en la boca: mask = 1 - 0.7 = 0.3 (30% swap, 70% original)
        mask = mask - mouth_mask * blend_ratio
        mask = np.clip(mask, 0.0, 1.0)

        return mask

    except Exception as e:
        print(f"[MOUTH_MASK] Error: {e}")
        return np.ones((target_img.shape[0], target_img.shape[1]), dtype=np.float32)


class FaceSwap:
    """Versión SIMPLIFICADA del procesador de face swap - sin efectos de mejora"""
    plugin_options: dict = None
    model = None
    model_xseg = None
    name = 'inswapper'
    devicename = None
    processorname = 'inswapper'
    type = 'faceswap'
    lock = threading.Lock()

    def Initialize(self, plugin_options: dict):
        """Inicializa el modelo inswapper_128.onnx"""
        with self.lock:
            if self.model is not None:
                print("[FaceSwap] Modelo ya inicializado")
                return

            self.plugin_options = plugin_options
            
            devnm = plugin_options.get("devicename", "cpu").lower()
            if 'cuda' in devnm:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']

            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/inswapper_128.onnx'))
            print(f"[FaceSwap] Buscando modelo en: {model_path}")
            print(f"[FaceSwap] ¿Existe el modelo?: {os.path.exists(model_path)}")
            
            try:
                self.model = insightface.model_zoo.get_model(model_path, providers=providers)
                self.devicename = devnm
                print(f"[FaceSwap] Modelo cargado correctamente con providers: {providers}")
            except Exception as e:
                print(f"[FaceSwap] Error cargando modelo con {providers}: {e}")
                if 'cuda' in devnm:
                    print("[FaceSwap] Intentando con CPU only...")
                    try:
                        self.model = insightface.model_zoo.get_model(model_path, providers=['CPUExecutionProvider'])
                        self.devicename = 'cpu'
                        print("[FaceSwap] Modelo cargado con CPU")
                    except Exception as e2:
                        print(f"[FaceSwap] Error también con CPU: {e2}")
                        import traceback
                        traceback.print_exc()

    def Run(self, source_face: Face, target_face: Face, temp_frame: Frame, paste_back: bool = True) -> Any:
        """Ejecuta inswapper con soporte completo para keypoints"""
        if self.model is None:
            raise Exception("FaceSwap not initialized")

        try:
            if isinstance(temp_frame, list):
                if len(temp_frame) > 0 and isinstance(temp_frame[0], np.ndarray):
                    temp_frame = temp_frame[0]
                else:
                    temp_frame = np.array(temp_frame)
            elif not isinstance(temp_frame, np.ndarray):
                temp_frame = np.array(temp_frame)

            if not hasattr(target_face, 'kps') or target_face.kps is None:
                return None

            if not isinstance(target_face.kps, np.ndarray):
                target_face.kps = np.array(target_face.kps, dtype=np.float32)
            if target_face.kps.shape != (5, 2):
                return None

            # Ejecutar el swap
            res = self.model.get(temp_frame, target_face, source_face, paste_back=True)

            if res is None:
                return None
            return res

        except Exception as e:
            print(f"[ERROR] FaceSwap Run: {e}")
            return None

    def Release(self):
        self.model = None
        self.model_xseg = None

    @staticmethod
    def paste_back_improved(target_img, source_face_img, M, target_face=None, landmarks_106=None):
        """
        Versión MEJORADA del paste back con:
        - WARP con INTER_LANCZOS4 (mejor calidad)
        - Máscara basada en keypoints REAL para mejor ajuste a la forma de la cara
        - PRESERVACIÓN DE BOCA ABIERTA: Detecta si la boca está abierta y preserva la zona
        - Blending NATURAL para evitar efecto de máscara
        """
        try:
            if M is None or source_face_img is None:
                return target_img

            h_src, w_src = source_face_img.shape[:2]
            scale = h_src / 128.0
            
            if abs(scale - 1.0) > 0.01:
                M_scaled = M * scale
                try:
                    M_inv = cv2.invertAffineTransform(M_scaled)
                except:
                    M_inv = cv2.invertAffineTransform(M)
                    source_face_img = cv2.resize(source_face_img, (128, 128), interpolation=cv2.INTER_LANCZOS4)
            else:
                M_inv = cv2.invertAffineTransform(M)

            # Blend ratio OPTIMIZADO para máxima similitud al origen (0.95 = 95% source, 5% target)
            blend_ratio = getattr(roop.globals, 'blend_ratio', 0.95)
            blend_ratio = max(0.0, min(1.0, blend_ratio))

            # WARP con LANCZOS4 (mejor calidad para downscaling)
            warped_struct = cv2.warpAffine(source_face_img, M_inv, (target_img.shape[1], target_img.shape[0]),
                                         flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
            
            # ============================================
            # NUEVO: DETECTAR BOCA ABIERTA Y PRESERVARLA
            # ============================================
            mouth_open = False
            mouth_region = None
            preserve_mouth = getattr(roop.globals, 'preserve_mouth_expression', True)
            
            if preserve_mouth and target_face is not None:
                mouth_open, mouth_region, open_ratio = detect_mouth_open(target_face, landmarks_106)
                if mouth_open:
                    print(f"[MOUTH_PRESERVE] Boca abierta detectada (ratio={open_ratio:.2f}) - Preservando expresión")
            
            # Crear máscara MEJORADA basada en keypoints si están disponibles
            if target_face is not None and hasattr(target_face, 'kps') and target_face.kps is not None:
                try:
                    kps = np.array(target_face.kps, dtype=np.float32)
                    if kps.shape == (5, 2):
                        # Los 5 keypoints son: ojo izquierdo, ojo derecho, nariz, boca izquierda, boca derecha
                        # Transformar keypoints al espacio de la imagen
                        kps_warped = cv2.transform(kps.reshape(1, -1, 2), M_inv).reshape(-1, 2)
                        
                        # Crear máscara usando los puntos de la cara
                        h, w = target_img.shape[:2]
                        mask = np.zeros((h, w), dtype=np.float32)
                        
                        # Puntos clave para la máscara
                        left_eye = tuple(kps_warped[0].astype(int))
                        right_eye = tuple(kps_warped[1].astype(int))
                        nose = tuple(kps_warped[2].astype(int))
                        left_mouth = tuple(kps_warped[3].astype(int))
                        right_mouth = tuple(kps_warped[4].astype(int))
                        
                        # Crear hull convexo de todos los puntos para mejor cobertura
                        # OPTIMIZADO v2: Añadido punto de frente para evitar efecto "flequillo"
                        points = np.array([left_eye, right_eye, nose, left_mouth, right_mouth], dtype=np.int32)
                        
                        # Calcular punto de frente (midpoint entre ojos, desplazado hacia arriba)
                        eye_mid_x = (left_eye[0] + right_eye[0]) // 2
                        eye_mid_y = (left_eye[1] + right_eye[1]) // 2
                        eye_distance = (((left_eye[0] - right_eye[0])**2 + (left_eye[1] - right_eye[1])**2) ** 0.5)
                        # Punto de frente: 40% de la distancia entre ojos hacia arriba
                        forehead_point = (eye_mid_x, int(eye_mid_y - eye_distance * 0.4))
                        
                        # Añadir punto de frente a los puntos del hull
                        all_points = np.vstack([points, [forehead_point]])
                        
                        hull = cv2.convexHull(all_points)
                        cv2.fillConvexPoly(mask, hull, 1.0)

                        # Expandir ligeramente hacia abajo para mejor cobertura de barbilla
                        chin_y = max(left_mouth[1], right_mouth[1]) + int((nose[1] - left_mouth[1]) * 0.3)
                        if chin_y < h:
                            cv2.fillConvexPoly(mask,
                                np.array([[left_mouth[0], chin_y], [right_mouth[0], chin_y],
                                         [right_mouth[0], right_mouth[1]], [left_mouth[0], left_mouth[1]]], dtype=np.int32),
                                0.3)  # Intensidad menor para expandir

                        # Blur suave para bordes naturales
                        blur_size = int(max(7, (min(w, h) // 25) | 1))
                        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
                        
                        # ============================================
                        # NUEVO: APLICAR PRESERVACIÓN DE BOCA
                        # ============================================
                        if mouth_open and mouth_region is not None:
                            # Crear máscara de preservación de boca
                            mouth_preserve_mask = create_mouth_preservation_mask(
                                target_img, mouth_region, blend_ratio=0.8
                            )
                            # Reducir el blending en la zona de la boca
                            mask = mask * mouth_preserve_mask
                        
                        # Convertir a 3 canales
                        if mask.ndim == 2:
                            mask = np.expand_dims(mask, axis=2)
                        
                        face_mask = mask
                    else:
                        raise ValueError("Invalid kps shape")
                except Exception as e:
                    # Si hay error con keypoints, usar máscara elíptica tradicional
                    face_mask = FaceSwap._create_ellipse_mask(target_img.shape[:2])
            else:
                # Usar máscara elíptica tradicional como fallback
                face_mask = FaceSwap._create_ellipse_mask(target_img.shape[:2])
            
            face_mask = np.clip(face_mask, 0, 1)
            if face_mask.ndim == 2:
                face_mask = np.expand_dims(face_mask, axis=2)

            target_float = target_img.astype(np.float32)
            warped_face_float = warped_struct.astype(np.float32)
            final_alpha = face_mask * blend_ratio
            
            # Blending DIRECTO Y NATURAL
            result = target_float * (1.0 - final_alpha) + warped_face_float * final_alpha
            
            # Ajuste de color SUAVE para coherencia - REDUCIDO para preservar identidad de origen
            use_color_correction = getattr(roop.globals, 'use_color_correction', True)
            if use_color_correction and final_alpha.max() > 0.1:
                mask_3ch = np.repeat(final_alpha[:, :, np.newaxis], 3, axis=2)
                original_face_avg = np.average(target_float, weights=mask_3ch[:, :, 0], axis=(0, 1))
                swapped_face_avg = np.average(warped_face_float, weights=mask_3ch[:, :, 0], axis=(0, 1))
                
                color_diff = original_face_avg - swapped_face_avg
                # Reducido de 0.7 a 0.25 para preservar mejor la identidad de la cara origen
                result += color_diff * final_alpha * 0.25
            
            return np.clip(result, 0, 255).astype(np.uint8)

        except Exception as e:
            print(f"[ERROR] paste_back_improved: {e}")
            return target_img
    
    @staticmethod
    def _create_ellipse_mask(shape):
        """Crea una máscara elíptica para la cara como fallback - OPTIMIZADA v2"""
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        center = (w // 2, h // 2)
        # OPTIMIZADO v2: Elipse más amplia para cubrir frente y evitar efecto "flequillo"
        radius_x = int(w * 0.48)  # Más ancho para cobertura lateral (96%)
        radius_y = int(h * 0.52)  # Más alto para cubrir frente y pelo (104%)
        cv2.ellipse(mask, center, (radius_x, radius_y), 0, 0, 360, 1.0, -1)

        # Blur para bordes suaves
        blur_size = int(max(7, (min(w, h) // 25) | 1))
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=2)
        return mask