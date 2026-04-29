# -*- coding: utf-8 -*-
"""
FacePreserver - Preservacion de rostros en ediciones de imagen

Este modulo detecta rostros en imagenes y preserva la identidad facial
durante el proceso de edicion con FLUX.
"""

from typing import Optional, List, Tuple
from PIL import Image
import numpy as np


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
        
        try:
            np_image = np.array(image)
            faces = self.analyzer.get(np_image)
            
            results = []
            for face in faces:
                results.append({
                    "bbox": face.bbox.tolist(),
                    "kps": face.kps.tolist(),
                    "embedding": face.embedding,
                    "det_score": face.det_score,
                })
            
            return results
            
        except Exception as e:
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
        Preserva los rostros del original en la imagen generada.
        
        Args:
            original: Imagen original
            generated: Imagen generada por FLUX
            method: Metodo de preservacion
                - "compare": Compara y usa el mejor rostro
                - "swap": Face swap automatico (requiere faceswap)
                - "blend": Combina rostros
                
        Returns:
            Imagen con rostros preservados
        """
        if not self._initialized:
            return generated
        
        try:
            # Detectar rostros en ambas imagenes
            orig_faces = self.detect_faces(original)
            gen_faces = self.detect_faces(generated)
            
            if len(orig_faces) == 0:
                # No hay rostros que preservar
                return generated
            
            if len(gen_faces) == 0:
                # No hay rostros en la generada, intentar face swap
                return self._face_swap(original, generated)
            
            # Hay rostros en ambas, intentar preservar el mas similar
            return self._preserve_best_face(original, generated)
            
        except Exception as e:
            print(f"[FacePreserver] Error preservando rostros: {e}")
            return generated
    
    def _face_swap(self, original: Image.Image, generated: Image.Image) -> Image.Image:
        """
        Face swap basico - intentar pegar rostros del original.
        
        Args:
            original: Imagen original con rostros
            generated: Imagen generada sin rostros
            
        Returns:
            Imagen con rostros pegados
        """
        try:
            from PIL import ImageDraw
            
            orig_faces = self.detect_faces(original)
            if len(orig_faces) == 0:
                return generated
            
            # Usar el rostro mas grande del original
            best_face = max(orig_faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
            
            # Crear mascara para el rostro
            bbox = best_face["bbox"]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            # Extraer rostro del original
            face_img = original.crop((x1, y1, x2, y2))
            face_img = face_img.resize((128, 128), Image.LANCZOS)
            
            # Intentar usar face swap si esta disponible
            try:
                import roop.faceswap as faceswap_module
                if hasattr(faceswap_module, 'swap_face'):
                    # Usar el rostro del original como referencia
                    result = faceswap_module.swap_face(generated, face_img)
                    if result is not None:
                        return result
            except ImportError:
                pass
            
            # Fallback: simplemente pegar el rostro en el centro de la imagen generada
            result = generated.copy()
            face_resized = face_img.resize((x2 - x1, y2 - y1), Image.LANCZOS)
            
            # Pegar con blend
            from PIL import Image as PILImage
            
            # Region central donde pegar
            center_x = generated.width // 2 - (x2 - x1) // 2
            center_y = generated.height // 2 - (y2 - y1) // 2
            
            # Crear region para pegar
            region = result.crop((center_x, center_y, center_x + (x2 - x1), center_y + (y2 - y1)))
            
            # Blend: 50% original, 50% rostro
            blended = PILImage.blend(region, face_resized, 0.5)
            result.paste(blended, (center_x, center_y))
            
            return result
            
        except Exception as e:
            print(f"[FacePreserver] Error en face swap: {e}")
            return generated
    
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
