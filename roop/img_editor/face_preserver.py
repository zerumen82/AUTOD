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
            
            for i, (gen_face, gen_crop) in enumerate(gen_faces_data):
                # Buscar la cara más similar en el original
                best_match = None
                best_source_crop = None
                max_sim = -1
                
                print(f"[FacePreserver] Analizando cara generada #{i+1}...")
                for j, (orig_face, orig_crop) in enumerate(orig_faces_data):
                    sim = self.compare_faces(orig_face.embedding, gen_face.embedding)
                    print(f"  - Similitud con cara original #{j+1}: {sim:.4f}")
                    if sim > max_sim:
                        max_sim = sim
                        best_match = orig_face
                        best_source_crop = orig_crop
                
                # Si encontramos un match razonable (>0.30), restauramos la cara original usando manual paste (más confiable para preserve después de edición fuerte)
                # Evitamos swapper.get porque falla (rank error en source) cuando las caras vienen de imágenes diferentes.
                if best_match is not None and max_sim > 0.30:
                    if max_sim > 0.78:
                        print(f"  [SKIP] Alta similitud ({max_sim:.4f}), mantenemos la cara generada para evitar problemas de escala")
                    else:
                        print(f"  [MATCH] Restaurando identidad con sim={max_sim:.4f} (manual paste)")
                        try:
                            if best_source_crop is not None and gen_face is not None:
                                x1, y1, x2, y2 = [int(v) for v in gen_face.bbox]
                                x1, y1 = max(0, x1), max(0, y1)
                                x2, y2 = min(result_cv2.shape[1], x2), min(result_cv2.shape[0], y2)
                                if x2 > x1 and y2 > y1:
                                    gen_w = x2 - x1
                                    gen_h = y2 - y1
                                    gen_cx = (x1 + x2) / 2.0
                                    gen_cy = (y1 + y2) / 2.0

                                    # Derive natural face size from the ORIGINAL matched face.
                                    if best_match is not None:
                                        ob = best_match.bbox
                                        o_w = max(1, ob[2] - ob[0])
                                        o_h = max(1, ob[3] - ob[1])
                                    else:
                                        o_w, o_h = gen_w, gen_h

                                    # Use the LARGER of original face size and generated head size (plus slight boost).
                                    # Guarantees the pasted face is never smaller than the generated head area.
                                    aspect = o_w / max(1.0, float(o_h))
                                    base = max(o_w, gen_w)
                                    target_w = int(base * 1.15)  # higher boost when we have to paste (low sim) to make face more prominent
                                    target_h = int(target_w / aspect)

                                    # Safety clamp
                                    if target_w > gen_w * 1.35 or target_h > gen_h * 1.35:
                                        scale = min( (gen_w * 1.15) / target_w , (gen_h * 1.15) / target_h )
                                        scale = max(0.85, scale)
                                        target_w = int(target_w * scale)
                                        target_h = int(target_h * scale)

                                    # Center the paste region on the generated face center
                                    px1 = max(0, int(gen_cx - target_w / 2))
                                    py1 = max(0, int(gen_cy - target_h / 2))
                                    px2 = min(result_cv2.shape[1], px1 + target_w)
                                    py2 = min(result_cv2.shape[0], py1 + target_h)
                                    target_w = px2 - px1
                                    target_h = py2 - py1

                                    # Build a better source region: use original face bbox + generous margin from the full original image.
                                    # This gives more head/hair/neck context so the pasted face looks fuller and not "small cutout".
                                    if best_match is not None:
                                        ox1, oy1, ox2, oy2 = [int(v) for v in best_match.bbox]
                                        margin = int(max(o_w, o_h) * 0.30)
                                        ox1 = max(0, ox1 - margin)
                                        oy1 = max(0, oy1 - margin)
                                        ox2 = min(orig_cv2.shape[1], ox2 + margin)
                                        oy2 = min(orig_cv2.shape[0], oy2 + margin)
                                        head_source = orig_cv2[oy1:oy2, ox1:ox2]
                                    else:
                                        head_source = best_source_crop

                                    source_resized = cv2.resize(head_source, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

                                    # Color/lighting match: adapt source face to the edited image's local tone.
                                    # Sample mostly from the upper/central part of the *generated* region (avoids new bare skin skewing the match too much).
                                    try:
                                        target_area = result_cv2[py1:py2, px1:px2].astype(np.float32)
                                        # Use upper ~65% for skin tone reference (more likely to be head area even after body edit)
                                        upper_h = max(1, int(target_h * 0.65))
                                        target_face_sample = result_cv2[py1:py1+upper_h, px1:px2].astype(np.float32)

                                        src_f = source_resized.astype(np.float32)

                                        t_mean = np.mean(target_face_sample, axis=(0, 1))
                                        s_mean = np.mean(src_f, axis=(0, 1))
                                        t_std = np.std(target_face_sample, axis=(0, 1))
                                        s_std = np.std(src_f, axis=(0, 1))
                                        std_scale = np.clip(t_std / (s_std + 1e-6), 0.55, 1.55)
                                        matched = (src_f - s_mean) * std_scale + t_mean
                                        source_resized = np.clip(matched, 0, 255).astype(np.uint8)

                                        # Even more generous mask when pasting to keep the face looking full size
                                        mask = np.zeros((target_h, target_w), dtype=np.uint8)
                                        cx = target_w // 2
                                        cy = int(target_h * 0.48)
                                        rx = max(6, int(target_w * 0.65))
                                        ry = max(6, int(target_h * 0.62))
                                        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
                                        mask = cv2.GaussianBlur(mask, (21, 21), 0)

                                        # Very late + very mild fade so almost the entire pasted face (including jaw) is kept
                                        lower_start = int(target_h * 0.82)
                                        if lower_start < target_h:
                                            for yy in range(lower_start, target_h):
                                                fade = 1.0 - ((yy - lower_start) / max(1, target_h - lower_start)) * 0.55
                                                fade = max(0.30, fade)
                                                mask[yy, :] = (mask[yy, :].astype(np.float32) * fade).astype(np.uint8)

                                        # Seamless clone using the computed (correctly sized) region
                                        center_pt = (target_w // 2, target_h // 2)
                                        blended_region = cv2.seamlessClone(
                                            source_resized,
                                            result_cv2[py1:py2, px1:px2],
                                            mask,
                                            center_pt,
                                            cv2.NORMAL_CLONE
                                        )
                                        result_cv2[py1:py2, px1:px2] = blended_region

                                        # Very bottom strip: gentle transition to edited body (only the very lowest part)
                                        neck_start = int(target_h * 0.90)
                                        if neck_start < target_h - 1:
                                            gen_part = result_cv2[py1 + neck_start:py2, px1:px2].astype(np.float32)
                                            face_part = blended_region[neck_start:, :].astype(np.float32)
                                            mix = (0.30 * face_part + 0.70 * gen_part).astype(np.uint8)
                                            result_cv2[py1 + neck_start:py2, px1:px2] = mix

                                        print("  [PASTE] OK manual face restore (larger head source + 15% boost + very generous mask for low-sim cases)")
                                        swapped_count += 1
                                    except Exception as inner_e:
                                        # Safe fallback: direct weighted paste on the computed region
                                        alpha = 0.78
                                        result_cv2[py1:py2, px1:px2] = (alpha * source_resized + (1 - alpha) * result_cv2[py1:py2, px1:px2]).astype(np.uint8)
                                        print(f"  [PASTE] Fallback alpha used ({inner_e})")
                                        swapped_count += 1
                        except Exception as e2:
                            print(f"  [PASTE ERROR] {e2}")
                else:
                    print(f"  [SKIP] Similitud insuficiente ({max_sim:.4f} <= 0.30)")
            
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
