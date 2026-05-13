# -*- coding: utf-8 -*-
"""
FacePreserver - Preservacion de rostros en ediciones de imagen

Este modulo detecta rostros en imagenes y preserva la identidad facial
durante el proceso de edicion con FLUX.
"""

from typing import Optional, List, Tuple
from PIL import Image
import numpy as np
import cv2


class FacePreserver:
    """
    Maneja la deteccion y preservacion de rostros.
    
    Utiliza insightface para deteccion y puede preservar rostros
    mediante comparacion post-generacion.
    """
    
    def __init__(self):
        self.analyzer = None
        self.face_detector = None
        self._initialized = False

    def initialize(self) -> Tuple[bool, str]:
        try:
            import insightface
            from insightface.app import FaceAnalysis

            self.analyzer = FaceAnalysis(allowed_modules=['detection', 'recognition'])
            self.analyzer.prepare(ctx_id=0, det_size=(640, 640))
            self._initialized = True

            return True, "Face detector inicializado"

        except ImportError:
            return False, "InsightFace no disponible. Instala con: pip install insightface"
        except Exception as e:
            return False, f"Error inicializando face detector: {str(e)}"

    def is_initialized(self) -> bool:
        """Check si el detector esta inicializado"""
        return self._initialized

    def detect_faces(self, image: Image.Image) -> List[dict]:
        if not self._initialized:
            return []

        try:
            np_image = np.array(image)
            faces = self.analyzer.get(np_image)

            results = []
            for face in faces:
                bbox = face.bbox.tolist() if hasattr(face.bbox, 'tolist') else list(face.bbox)
                kps = face.kps.tolist() if hasattr(face.kps, 'tolist') else list(face.kps)
                results.append({
                    "bbox": bbox,
                    "kps": kps,
                    "embedding": face.embedding,
                    "det_score": face.det_score,
                })

            return results

        except Exception as e:
            if 'conv_1_relu' in str(e) or 'broadcast' in str(e):
                print(f"[FacePreserver] ONNX shape error (genderage), redimensionando imagen...")
                try:
                    w, h = image.size
                    scale = 512 / max(w, h)
                    if scale < 1:
                        sm = image.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
                        np_image = np.array(sm)
                        faces = self.analyzer.get(np_image)
                        results = []
                        for face in faces:
                            bbox = [c / scale for c in (face.bbox.tolist() if hasattr(face.bbox, 'tolist') else list(face.bbox))]
                            results.append({"bbox": bbox, "kps": [], "embedding": face.embedding, "det_score": face.det_score})
                        print(f"[FacePreserver] OK tras redimensionar: {len(results)} rostros")
                        return results
                except:
                    pass
            print(f"[FacePreserver] Error detectando rostros: {e}")
            return []
    
    def count_faces(self, image: Image.Image) -> int:
        """
        Cuenta el numero de rostros en una imagen.
        
        Args:
            image: Imagen PIL
            
        Returns:
            Numero de rostros detectados
        """
        faces = self.detect_faces(image)
        return len(faces)
    
    def preserve_faces(
        self,
        original: Image.Image,
        generated: Image.Image,
        method: str = "swap"
    ) -> Image.Image:
        """
        Preserva los rostros del original en la imagen generada usando Face Swap quirúrgico.
        """
        if not self._initialized:
            return generated
        
        try:
            # Convertir PIL a numpy (BGR para OpenCV/InsightFace)
            orig_cv2 = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)
            gen_cv2 = cv2.cvtColor(np.array(generated), cv2.COLOR_RGB2BGR)
            
            # 1. Obtener todas las caras del original y la generada
            # Usamos el analizador de roop.face_util que ya está optimizado
            import roop.face_util as face_util
            orig_faces_data = face_util.extract_face_images(orig_cv2, target_face_detection=True)
            gen_faces_data = face_util.extract_face_images(gen_cv2, target_face_detection=True)
            
            if not orig_faces_data or not gen_faces_data:
                print(f"[FacePreserver] No se detectaron caras en {'original' if not orig_faces_data else 'generada'}")
                return generated
            
            # 2. Emparejar caras por similitud de embedding o posición
            from roop.swapper import get_face_swapper
            swapper = get_face_swapper()
            if swapper is None:
                print("[FacePreserver] Error: No se pudo cargar el swapper")
                return generated

            result_cv2 = gen_cv2.copy()
            swapped_count = 0
            
            for i, (gen_face, _) in enumerate(gen_faces_data):
                # Buscar la cara más similar en el original
                best_match = None
                max_sim = -1
                
                print(f"[FacePreserver] Analizando cara generada #{i+1}...")
                for j, (orig_face, _) in enumerate(orig_faces_data):
                    sim = self.compare_faces(orig_face.embedding, gen_face.embedding)
                    print(f"  - Similitud con cara original #{j+1}: {sim:.4f}")
                    if sim > max_sim:
                        max_sim = sim
                        best_match = orig_face
                
                # Si encontramos un match razonable (>0.25), hacemos el swap quirúrgico
                if best_match is not None and max_sim > 0.25:
                    try:
                        print(f"  [MATCH] Restaurando identidad con sim={max_sim:.4f}")
                        # Realizar el swap directamente en el frame generado
                        result_cv2 = swapper.get(result_cv2, gen_face, best_match, paste_back=True)
                        swapped_count += 1
                    except Exception as e:
                        print(f"  [ERROR] Fallo en swapper.get: {e}")
                else:
                    print(f"  [SKIP] Similitud insuficiente ({max_sim:.4f} <= 0.4)")
            
            print(f"[FacePreserver] Se restauraron {swapped_count} identidad(es) facial(es)")
            
            # Convertir de vuelta a PIL
            return Image.fromarray(cv2.cvtColor(result_cv2, cv2.COLOR_BGR2RGB))
            
        except Exception as e:
            print(f"[FacePreserver] Error crítico preservando rostros: {e}")
            import traceback
            traceback.print_exc()
            return generated

    def _face_swap(self, original: Image.Image, generated: Image.Image) -> Image.Image:
        # Este método ahora es redundante pero lo mantenemos por compatibilidad básica
        return self.preserve_faces(original, generated, method="swap")
    
    def _preserve_best_face(self, original: Image.Image, generated: Image.Image) -> Image.Image:
        """
        Preserva el rostro mas similar del original en la imagen generada.
        
        Args:
            original: Imagen original
            generated: Imagen generada
            
        Returns:
            Imagen con rostro preservado
        """
        try:
            orig_faces = self.detect_faces(original)
            gen_faces = self.detect_faces(generated)
            if len(gen_faces) == 0:
                return generated
            
            # Encontrar el rostro generado con mayor similitud a cualquier rostro original
            best_match = None
            best_score = -1
            
            for gen_face in gen_faces:
                for orig_face in orig_faces:
                    score = self.compare_faces(
                        orig_face.get("embedding"),
                        gen_face.get("embedding")
                    )
                    if score > best_score:
                        best_score = score
                        best_match = gen_face
            
            if best_match is not None and best_score > 0.5:
                # El rostro mas similar es suficientemente bueno
                return generated
            else:
                # Intentar face swap con el mejor rostro original
                return self._face_swap(original, generated)
                
        except Exception as e:
            print(f"[FacePreserver] Error en preserve best face: {e}")
            return generated
    
    def extract_face_embeddings(self, image: Image.Image) -> Optional[np.ndarray]:
        """
        Extrae embeddings faciales de una imagen.
        
        Args:
            image: Imagen PIL
            
        Returns:
            Array de embeddings o None si no hay rostros
        """
        faces = self.detect_faces(image)
        
        if len(faces) == 0:
            return None
        
        # Retornar embedding del primer rostro
        return faces[0].get("embedding")
    
    def compare_faces(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compara dos embeddings faciales.
        
        Args:
            embedding1: Primer embedding
            embedding2: Segundo embedding
            
        Returns:
            Similitud coseno (0-1)
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        try:
            # Similitud coseno
            cos_sim = np.dot(embedding1, embedding2)
            cos_sim /= (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            return float(cos_sim)
        except Exception:
            return 0.0


def create_white_mask(image: Image.Image) -> Image.Image:
    """
    Crea una mascara blanca del tamano de la imagen.
    """
    mask = Image.new("L", image.size, 255)
    return mask


def create_roi_mask(
    image: Image.Image,
    x: int,
    y: int,
    width: int,
    height: int
) -> Image.Image:
    """
    Crea una mascara con region de interes.
    
    Args:
        image: Imagen de referencia
        x, y: Coordenadas de la esquina superior izquierda
        width, height: Dimensiones del ROI
    """
    mask = Image.new("L", image.size, 0)
    
    # Asegurar que esta dentro de los limites
    x = max(0, min(x, image.width))
    y = max(0, min(y, image.height))
    width = min(width, image.width - x)
    height = min(height, image.height - y)
    
    # Dibujar rectangulo blanco
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x, y, x + width, y + height], fill=255)
    
    return mask
