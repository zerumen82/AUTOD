import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

import roop.globals
from roop.utils import run_ffmpeg
from roop.quality_enhancements import (
    match_color_histogram,
    create_soft_mask,
    blend_with_poisson,
    adjust_face_brightness
)


def is_valid_progress_callback(callback):
    """Verifica si un callback de progreso es válido sin causar errores"""
    if callback is None:
        return False
    try:
        return hasattr(callback, '__call__')
    except Exception as e:
        print(f"[WARNING] Error verificando callback: {e}")
        return False

def get_gender(face):
    """Obtiene el género de una cara"""
    try:
        gender_confidence = 0.0
        detected_gender = None
        validation_score = 0.0
        
        strictness_mode = getattr(roop.globals, 'gender_strictness_mode', 'balanced')
        
        if strictness_mode == 'permissive':
            threshold_low, threshold_high = getattr(roop.globals, 'gender_threshold_permissive', (0.30, 0.70))
        elif strictness_mode == 'strict':
            threshold_low, threshold_high = getattr(roop.globals, 'gender_threshold_strict', (0.45, 0.55))
        else:
            threshold_low, threshold_high = getattr(roop.globals, 'gender_threshold_balanced', (0.40, 0.60))
        
        min_confidence_threshold = getattr(roop.globals, 'gender_confidence_threshold', 0.3)
  
        if hasattr(face, 'gender') and face.gender is not None:
            gender_value = face.gender

            if isinstance(gender_value, (list, np.ndarray)):
                gender_value = gender_value[0] if len(gender_value) > 0 else None
              
            if gender_value is not None:
                gender_value = float(gender_value)
                  
                if gender_value == 0 or gender_value == 0.0:
                    detected_gender = 'female'
                    gender_confidence = 0.98
                    validation_score += 2.0
                elif gender_value == 1 or gender_value == 1.0:
                    detected_gender = 'male'
                    gender_confidence = 0.98
                    validation_score += 2.0
                elif 0.0 < gender_value < 1.0:
                    if gender_value < threshold_low:
                        detected_gender = 'female'
                        gender_confidence = 0.70 + (threshold_low - gender_value) * 0.5
                        gender_confidence = min(0.95, gender_confidence)
                        validation_score += 1.5
                    elif gender_value > threshold_high:
                        detected_gender = 'male'
                        gender_confidence = 0.70 + (gender_value - threshold_high) * 0.5
                        gender_confidence = min(0.95, gender_confidence)
                        validation_score += 1.5
                    else:
                        if strictness_mode == 'permissive':
                            if gender_value < 0.5:
                                detected_gender = 'female'
                                gender_confidence = 0.55
                            else:
                                detected_gender = 'male'
                                gender_confidence = 0.55
                            validation_score += 0.8
                        else:
                            detected_gender = None
                            gender_confidence = 0.30
                            validation_score += 0.3
                else:
                    if gender_value < 0:
                        detected_gender = 'female'
                        gender_confidence = 0.60
                    else:
                        detected_gender = 'male'
                        gender_confidence = 0.60
  
        if hasattr(face, 'sex') and face.sex is not None:
            sex_value = str(face.sex).upper()
            if sex_value in ['F', 'FEMALE', 'WOMAN', 'FEM', 'W']:
                if detected_gender == 'female':
                    validation_score += 1.0
                    gender_confidence = min(0.98, gender_confidence + 0.15)
                elif detected_gender is None or detected_gender == 'male':
                    detected_gender = 'female'
                    gender_confidence = max(gender_confidence, 0.80)
                    validation_score += 0.8
            elif sex_value in ['M', 'MALE', 'MAN', 'MASC']:
                if detected_gender == 'male':
                    validation_score += 1.0
                    gender_confidence = min(0.98, gender_confidence + 0.15)
                elif detected_gender is None or detected_gender == 'female':
                    detected_gender = 'male'
                    gender_confidence = max(gender_confidence, 0.80)
                    validation_score += 0.8
  
        if hasattr(face, 'age') and face.age is not None:
            try:
                age_value = float(face.age)
                if detected_gender and 10 <= age_value <= 90:
                    validation_score += 0.5
                    gender_confidence = min(0.98, gender_confidence + 0.05)
            except:
                pass
  
        if hasattr(face, '__setattr__'):
            try:
                face.gender_confidence = gender_confidence
            except:
                pass
  
        if detected_gender and gender_confidence >= min_confidence_threshold:
            return detected_gender
        else:
            if strictness_mode == 'permissive' and detected_gender and gender_confidence > 0.20:
                return detected_gender
            return None
  
    except Exception as e:
        print(f"[GENDER_ERROR] Exception in get_gender: {e}")
        import traceback
        traceback.print_exc()
        return None

class ProcessMgr:
    """Gestor de procesamiento de face swapping v2.1"""
    _swap_call_count = 0  # Contador global para debug
    
    def __init__(self, progress_callback=None):
        self.progress_callback = progress_callback
        self.input_facesets = []
        self.target_faces = []
        self.options = None
        self.is_initialized = False
        self.processors = {}
        self.thread_lock = threading.Lock()
        
        self.source_embeddings_cache = {}
        self.previous_result = None
        self.frame_count = 0
        
        self.face_assignment_cache = {}
        self.face_position_history = {}
        self.global_source_for_all_id = None
        self.selected_face_assignment_cache = {}
        
        self.selected_assignment_ttl = getattr(roop.globals, 'selected_assignment_ttl', 60)
        self.selected_top_k = getattr(roop.globals, 'selected_top_k', 3)
        
        if is_valid_progress_callback(self.progress_callback):
            print(f"ProcessMgr v2.1 inicializado con progress_callback (Top-K={self.selected_top_k}, TTL={self.selected_assignment_ttl})")
        else:
            print(f"ProcessMgr v2.1 inicializado SIN progress_callback (Top-K={self.selected_top_k}, TTL={self.selected_assignment_ttl})")
        
        self._initialize_processors()

    def initialize(self, input_facesets, target_faces, options):
        self.input_facesets = input_facesets or []
        self.target_faces = target_faces or []
        self.options = options
        self.is_initialized = True

        self._cache_source_embeddings()

        print(f"ProcessMgr v2.1 inicializado: {len(self.input_facesets)} facesets, {len(self.target_faces)} target faces")
        
        if len(self.input_facesets) == 0:
            print("[WARNING] No hay facesets de origen cargados")
        if len(self.target_faces) == 0:
            print("[WARNING] No hay caras destino seleccionadas")
        
        self.global_source_for_all_id = None

    def _cache_source_embeddings(self):
        self.source_embeddings_cache.clear()
        
        for faceset in self.input_facesets:
            if hasattr(faceset, "faces") and faceset.faces:
                for face in faceset.faces:
                    if hasattr(face, 'embedding') and face.embedding is not None:
                        emb = np.array(face.embedding, dtype=np.float32)
                        norm = np.linalg.norm(emb)
                        if norm > 0:
                            self.source_embeddings_cache[id(face)] = emb / norm
        
        print(f"v2.1: {len(self.source_embeddings_cache)} embeddings cacheados")

    def _initialize_processors(self):
        try:
            from roop.processors.FaceSwap import FaceSwap
            from roop.processors.Enhance_GFPGAN import Enhance_GFPGAN
            from roop.processors.Enhance_GPEN import Enhance_GPEN
            from roop.processors.Enhance_CodeFormer import Enhance_CodeFormer
            from roop.processors.Enhance_RestoreFormerPPlus import Enhance_RestoreFormerPPlus

            try:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                use_cuda = "CUDAExecutionProvider" in providers
            except Exception:
                use_cuda = False

            devname = "cuda" if use_cuda else "cpu"
            
            print(f"[ProcessMgr] Inicializando FaceSwap con devicename={devname}")
            self.processors["faceswap"] = FaceSwap()
            self.processors["faceswap"].Initialize({"devicename": devname})
            print(f"[ProcessMgr] FaceSwap inicializado correctamente")
            
            # Obtener el enhancer seleccionado
            selected_enhancer = getattr(roop.globals, 'selected_enhancer', 'GFPGAN')
            use_enhancer = getattr(roop.globals, 'use_enhancer', True)
            
            if use_enhancer and selected_enhancer and selected_enhancer != "None":
                try:
                    if selected_enhancer == "GFPGAN":
                        self.processors["enhance_gfpgan"] = Enhance_GFPGAN()
                        self.processors["enhance_gfpgan"].Initialize({"devicename": devname})
                        print(f"GFPGAN Enhancer initialized (devicename={devname})")
                    elif selected_enhancer == "CodeFormer":
                        self.processors["enhance_codeformer"] = Enhance_CodeFormer()
                        self.processors["enhance_codeformer"].Initialize({"devicename": devname})
                        print(f"CodeFormer Enhancer initialized (devicename={devname})")
                    elif selected_enhancer == "Restoreformer++":
                        self.processors["enhance_restoreformer"] = Enhance_RestoreFormerPPlus()
                        self.processors["enhance_restoreformer"].Initialize({"devicename": devname})
                        print(f"Restoreformer++ Enhancer initialized (devicename={devname})")
                    elif selected_enhancer == "GPEN":
                        self.processors["enhance_gpen"] = Enhance_GPEN()
                        self.processors["enhance_gpen"].Initialize({"devicename": devname})
                        print(f"GPEN Enhancer initialized (devicename={devname})")
                except Exception as e:
                    print(f"Failed to initialize {selected_enhancer} Enhancer: {e}")

            print(f"ProcessMgr: Procesadores inicializados (devicename={devname})")
        except Exception as e:
            print(f"Failed to initialize processors: {e}")
            import traceback
            traceback.print_exc()

    def run_batch(self, input_paths, output_paths, num_threads=1):
        if not self.is_initialized:
            print("ProcessMgr not initialized")
            return

        def process_single_image(input_path, output_path):
            try:
                frame = cv2.imread(input_path)
                if frame is None:
                    print(f"Could not read image: {input_path}")
                    return
                processed_frame = self.process_frame(frame, file_path=input_path)
                if processed_frame is not None:
                    cv2.imwrite(output_path, processed_frame)
                    print(f"Processed: {os.path.basename(input_path)} -> {os.path.basename(output_path)}")
            except Exception as e:
                print(f"Processing {input_path}: {e}")

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for ip, op in zip(input_paths, output_paths):
                futures.append(executor.submit(process_single_image, ip, op))
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Batch processing error: {e}")

    def process_frame(self, frame, enable_temporal_smoothing=False, file_path=None):
        try:
            if not self.is_initialized or frame is None:
                print(f"[WARNING] Process_frame llamado pero no inicializado o frame None")
                return frame
        except:
            pass
        
        # NUEVO: Tracking de caras ya procesadas en este frame para evitar doble swap
        self._frame_processed_faces = getattr(self, '_frame_processed_faces', {})
        current_frame_key = id(frame)
        if current_frame_key != getattr(self, '_last_frame_id', None):
            self._frame_processed_faces = {}
            self._last_frame_id = current_frame_key
        
        # NUEVO: Asegurar que selected_face_references exista
        if not hasattr(roop.globals, 'selected_face_references'):
            roop.globals.selected_face_references = {}
        
        try:
            if "faceswap" in self.processors and len(self.input_facesets) > 0:
                from roop.face_util_rotation import get_all_faces_smart
                target_faces_detected = get_all_faces_smart(frame, min_score=None)

                valid_faces = target_faces_detected
                
                if not valid_faces:
                    return frame

                face_swap_mode = getattr(self.options, 'face_swap_mode', 'selected')
                
                # DEBUG: Verificar qué modo se está recibiendo
                print(f"[DEBUG] process_frame: face_swap_mode='{face_swap_mode}' (tipo: {type(face_swap_mode).__name__})")
                
                faces_to_process = []

                if face_swap_mode == 'selected_faces_frame':
                    video_path = getattr(self.options, 'current_video_path', None)
                    if video_path is None and file_path:
                        video_path = file_path
                    
                    if video_path:
                        # FIX: Unificar busqueda en selected_face_references
                        video_basename = os.path.basename(video_path)
                        video_key = f"selected_face_ref_{video_basename}"
                        
                        # Verificar si hay referencia guardada en selected_face_references
                        has_ref = False
                        if hasattr(roop.globals, 'selected_face_references'):
                            if video_key in roop.globals.selected_face_references:
                                has_ref = True
                            else:
                                for k in roop.globals.selected_face_references.keys():
                                    if k == video_basename or k.endswith(f"_{video_basename}"):
                                        has_ref = True
                                        break
                        
                        if has_ref:
                            target_face = self._find_target_face_for_selected_mode(video_path, valid_faces)
                            if target_face:
                                target_key = self._get_face_key(target_face)
                                if target_key in self._frame_processed_faces:
                                    print(f"[DEBUG] Cara ya procesada en este frame, saltando")
                                    return frame
                                self._frame_processed_faces[target_key] = True
                                faces_to_process = [target_face]
                            else:
                                # No se encontró la cara objetivo, NO procesar
                                faces_to_process = []
                        else:
                            # No hay selección del usuario, NO procesar este video
                            print(f"[SELECTED_FACES_FRAME] ⚠️ No hay cara seleccionada para {video_basename} - OMITIENDO procesamiento")
                            faces_to_process = []
                    else:
                        # Sin video_path, no procesar
                        faces_to_process = []

                elif face_swap_mode == 'all':
                    faces_to_process = valid_faces

                elif face_swap_mode in ['selected', 'selected_faces']:
                    # Modo "Selected Faces": Procesa UNA cara por imagen automáticamente
                    # Target = user-selected face per image (de selected_face_references)
                    # Source = ALEATORIO del faceset de origen
                    # IMPORTANTE: Si el usuario NO seleccionó una cara para esta imagen, NO se procesa
                    
                    if file_path:
                        filename = os.path.basename(file_path)
                        video_key = f"selected_face_ref_{filename}"
                        face_found = False
                        
                        if hasattr(roop.globals, 'selected_face_references'):
                            if video_key in roop.globals.selected_face_references:
                                face_ref_data = roop.globals.selected_face_references[video_key]
                                target_face = face_ref_data.get('face_obj')
                                
                                if target_face is not None and hasattr(target_face, 'bbox') and target_face.bbox is not None:
                                    best_match = None
                                    best_iou = 0.0
                                    for face in valid_faces:
                                        if not hasattr(face, 'bbox'):
                                            continue
                                        iou = self._bbox_iou(target_face.bbox, face.bbox)
                                        if iou > best_iou:
                                            best_iou = iou
                                            best_match = face
                                    
                                    if best_match and best_iou > 0.3:
                                        faces_to_process = [best_match]
                                        face_found = True
                                    # Si no hay match por bbox, intentar por embedding
                                    elif hasattr(target_face, 'embedding') and target_face.embedding is not None:
                                        # Buscar por similitud de embedding
                                        best_emb_match = None
                                        best_score = 0.0
                                        for face in valid_faces:
                                            if hasattr(face, 'embedding') and face.embedding is not None:
                                                score = self._calculate_similarity(target_face.embedding, face.embedding)
                                                if score > best_score:
                                                    best_score = score
                                                    best_emb_match = face
                                        if best_emb_match and best_score > 0.3:
                                            faces_to_process = [best_emb_match]
                                            face_found = True
                            else:
                                # Buscar con clave sin prefijo para compatibilidad
                                for k, v in roop.globals.selected_face_references.items():
                                    if k == filename or k.endswith(f"_{filename}"):
                                        face_ref_data = v
                                        target_face = face_ref_data.get('face_obj')
                                        
                                        if target_face is not None and hasattr(target_face, 'bbox') and target_face.bbox is not None:
                                            best_match = None
                                            best_iou = 0.0
                                            for face in valid_faces:
                                                if not hasattr(face, 'bbox'):
                                                    continue
                                                iou = self._bbox_iou(target_face.bbox, face.bbox)
                                                if iou > best_iou:
                                                    best_iou = iou
                                                    best_match = face
                                            
                                            if best_match and best_iou > 0.3:
                                                faces_to_process = [best_match]
                                                face_found = True
                                            elif hasattr(target_face, 'embedding') and target_face.embedding is not None:
                                                best_emb_match = None
                                                best_score = 0.0
                                                for face in valid_faces:
                                                    if hasattr(face, 'embedding') and face.embedding is not None:
                                                        score = self._calculate_similarity(target_face.embedding, face.embedding)
                                                        if score > best_score:
                                                            best_score = score
                                                            best_emb_match = face
                                                if best_emb_match and best_score > 0.3:
                                                    faces_to_process = [best_emb_match]
                                                    face_found = True
                                        break
                        
                        # Si no se encontró cara seleccionada, NO procesar esta imagen
                        if not face_found:
                            print(f"[SELECTED_FACES] ⚠️ No hay cara seleccionada para {filename} - OMITIENDO procesamiento")
                            faces_to_process = []
                    else:
                        # Sin file_path, no procesar
                        faces_to_process = []
                    
                else:
                    faces_to_process = valid_faces
                  
                processed_faces = 0
                result_frame = frame.copy()
                
                faces_to_process_limited = faces_to_process[:1] if face_swap_mode in ['selected_faces_frame', 'selected_faces'] else faces_to_process
                
                for i, target_face in enumerate(faces_to_process_limited):
                    try:
                        all_faces = []
                        for faceset in self.input_facesets:
                            if hasattr(faceset, "faces") and faceset.faces:
                                all_faces.extend(faceset.faces)
                          
                        if not all_faces:
                            continue
                        
                        video_path = getattr(self.options, 'current_video_path', None)
                        source_face = self._select_source_face(target_face, all_faces, face_swap_mode, video_path)
                        
                        if source_face is None:
                            continue
                        
                        result_frame = self._process_face_swap_v21(
                            source_face, target_face, result_frame, frame, enable_temporal_smoothing
                        )
                        processed_faces += 1
                        
                        if face_swap_mode == 'selected_faces_frame':
                            break
                        
                    except Exception as e:
                        print(f"Processing face {i+1}: {e}")
                        continue
                  
                return result_frame
            else:
                return frame
        except Exception as e:
            print(f"Frame processing error: {e}")
            return frame

    def _select_source_face(self, target_face, candidate_faces, face_swap_mode, video_path=None):
        try:
            if not candidate_faces:
                return None

            # Según la guía: docs/FACESWAP_SD_TAB_GUIDE.md
            # DEBUG: Contar cuántas caras únicas hay en el faceset
            unique_faces = set()
            for f in candidate_faces:
                if hasattr(f, 'embedding') and f.embedding is not None:
                    try:
                        # Convertir a numpy array si no lo es
                        emb_array = np.array(f.embedding) if not isinstance(f.embedding, np.ndarray) else f.embedding
                        emb_tuple = tuple(emb_array[:10].flatten()) if len(emb_array) > 10 else tuple(emb_array.flatten())
                        unique_faces.add(emb_tuple)
                    except:
                        pass
            print(f"[DEBUG] _select_source_face: modo={face_swap_mode}, candidatos={len(candidate_faces)}, unicos={len(unique_faces)}")
            
            # Modo "Selected Faces Frame" (Tracking de video):
            # - Source: MISMA CARA para todo el video (la más grande del faceset)
            if face_swap_mode == 'selected_faces_frame':
                best_face = max(candidate_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                print(f"[SELECT_SOURCE] Modo selected_faces_frame: usando cara más grande (área={ (best_face.bbox[2]-best_face.bbox[0])*(best_face.bbox[3]-best_face.bbox[1]):.0f})")
                return best_face
            
            # Modo "Selected" y "Selected Faces" (Selección por imagen):
            # - Source: ALEATORIO del faceset de origen
            if face_swap_mode in ['selected', 'selected_faces']:
                if len(candidate_faces) > 1:
                    # Importante: usar random.choice directamente sin caching
                    source_face = random.choice(candidate_faces)
                    print(f"[SELECT_SOURCE] Modo {face_swap_mode}: cara aleatoria #{candidate_faces.index(source_face)+1} de {len(candidate_faces)}")
                    return source_face
                else:
                    return candidate_faces[0]

            # Modo "All" (Automático):
            # - Source: Cara de origen más grande/del primer faceset
            best_face = max(candidate_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            return best_face

        except Exception as e:
            print(f"[ERROR] _select_source_face: {e}")
            return candidate_faces[0] if candidate_faces else None

    def _setup_selected_faces_frame_for_video(self, video_path, valid_faces):
        try:
            # NUEVO: Asegurar que selected_face_references exista
            if not hasattr(roop.globals, 'selected_face_references'):
                roop.globals.selected_face_references = {}
            
            if not valid_faces:
                return

            video_basename = os.path.basename(video_path)
            video_key = f"selected_face_ref_{video_basename}"
            
            selected_face = None
            selection_method = "auto"
            
            face_ref_data = None
            user_has_selected_face = False  # Flag para saber si el usuario seleccionó una cara
            
            if hasattr(roop.globals, 'selected_face_references'):
                if video_key in roop.globals.selected_face_references:
                    face_ref_data = roop.globals.selected_face_references[video_key]
                    user_has_selected_face = True
                else:
                    clean_video_name = video_basename
                    
                    for k, v in roop.globals.selected_face_references.items():
                        if not k.startswith("selected_face_ref_"):
                            continue
                        
                        ref_filename = k.replace("selected_face_ref_", "")
                        
                        if ref_filename.lower() == clean_video_name.lower():
                            face_ref_data = v
                            user_has_selected_face = True
                            break
                        if ref_filename in clean_video_name or clean_video_name in ref_filename:
                            face_ref_data = v
                            user_has_selected_face = True
                            break
            
            # Guardar flag para saber si el usuario seleccionó una cara
            setattr(self, f'_user_selected_face_{video_path}', user_has_selected_face)
            
            # En modo selected_faces_frame, SIEMPRE usamos la selección del usuario o el fallback
            # Si el usuario NO seleccionó ninguna cara, usamos fallback a la cara más grande
            if not user_has_selected_face:
                print(f"[SELECTED_FACES_FRAME] ℹ️ No hay cara seleccionada por el usuario para {video_basename}, usando fallback automático")
            
            if face_ref_data:
                user_selected_face = face_ref_data.get('face_obj')
                user_embedding = face_ref_data.get('embedding')
                user_bbox = face_ref_data.get('bbox')
                
                if user_selected_face is not None:
                    best_match = None
                    best_score = -1
                    
                    for face in valid_faces:
                        if not hasattr(face, 'embedding') or face.embedding is None:
                            continue
                        
                        if user_embedding is not None:
                            score = self._calculate_similarity(user_embedding, face.embedding)
                        else:
                            score = self._bbox_iou(user_bbox, face.bbox) if user_bbox is not None else 0
                        
                        if score > best_score:
                            best_score = score
                            best_match = face
                    
                    if best_match and best_score > 0.03:
                        selected_face = best_match
                        selection_method = f"usuario (score={best_score:.2f})"
                    else:
                        selected_face = user_selected_face
                        selection_method = f"usuario-guardado (score={best_score:.2f})"
            
            if selected_face is None:
                selected_face = max(valid_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                selection_method = f"auto-grande ({len(valid_faces)} caras)"

            attr_name = f'_reference_face_selected_{video_path}'
            setattr(self, f'global_source_for_all_id_{video_path}', id(selected_face))
            setattr(self, f'global_source_bbox_{video_path}', selected_face.bbox)
            setattr(self, f'_target_face_assigned_{video_path}', selected_face)
            
            has_embedding = False
            if hasattr(selected_face, 'embedding') and selected_face.embedding is not None:
                try:
                    emb = np.array(selected_face.embedding, dtype=np.float32)
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                        setattr(self, f'global_source_embedding_{video_path}', emb)
                        setattr(self, f'_original_embedding_{video_path}', emb.copy())
                        has_embedding = True
                except Exception as e:
                    print(f"[SETUP] {video_basename}: Error embedding: {e}")

            setattr(self, attr_name, True)
            print(f"[SETUP] {video_basename}: {selection_method}, embedding={'✓' if has_embedding else '✗'}")

        except Exception as e:
            print(f"[SETUP ERROR] {os.path.basename(video_path)}: {e}")
            import traceback
            traceback.print_exc()

    def _find_target_face_for_selected_mode(self, video_path, valid_faces):
        """Encuentra la cara objetivo que coincide con la seleccionada por el usuario."""
        try:
            # Verificar si el usuario ha seleccionado una cara
            user_selected_attr = f'_user_selected_face_{video_path}'
            user_selected = getattr(self, user_selected_attr, True)  # Por defecto True para permitir fallback
            
            # Si el usuario NO seleccionó ninguna cara, usamos fallback (cara más grande)
            # Esto solo aplica en modo selected_faces_frame
            use_fallback = not user_selected
            
            # Asegurar que selected_face_references exista
            if not hasattr(roop.globals, 'selected_face_references'):
                roop.globals.selected_face_references = {}

            if not valid_faces:
                return None

            video_basename = os.path.basename(video_path)
            video_key = f"selected_face_ref_{video_basename}"

            assigned_attr = f'_target_face_assigned_{video_path}'
            original_embedding_attr = f'_original_embedding_{video_path}'
            tracking_lost_attr = f'_tracking_lost_count_{video_path}'
            position_history_attr = f'_position_history_{video_path}'
            frame_count_attr = f'_frame_count_{video_path}'

            MIN_SCORE_THRESHOLD = 0.02
            REACQUIRE_EMB_THRESHOLD = 0.03
            HIGH_CONFIDENCE_THRESHOLD = 0.06
            # Umbral de distancia mínima entre caras para considerar que hay superposición significativa
            MIN_FACE_DISTANCE = 50  # píxeles mínimos entre centros de caras (aumentado para mejor separación)
            # Si dos caras están más cerca que esto, considerar que hay conflicto (beso/abrazo)

            if not hasattr(self, tracking_lost_attr):
                setattr(self, tracking_lost_attr, 0)
            if not hasattr(self, position_history_attr):
                setattr(self, position_history_attr, [])
            if not hasattr(self, frame_count_attr):
                setattr(self, frame_count_attr, 0)

            lost_count = getattr(self, tracking_lost_attr)
            position_history = getattr(self, position_history_attr)
            frame_count = getattr(self, frame_count_attr) + 1
            setattr(self, frame_count_attr, frame_count)

            def is_face_too_close(face1, face2, min_distance=MIN_FACE_DISTANCE):
                """Check if two faces are too close (could cause confusion in kisses/embraces)"""
                if not hasattr(face1, 'bbox') or not hasattr(face2, 'bbox'):
                    return False
                c1 = ((face1.bbox[0] + face1.bbox[2]) / 2, (face1.bbox[1] + face1.bbox[3]) / 2)
                c2 = ((face2.bbox[0] + face2.bbox[2]) / 2, (face2.bbox[1] + face2.bbox[3]) / 2)
                distance = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                return distance < min_distance

            def get_face_center(face):
                """Get center coordinates of a face"""
                if not hasattr(face, 'bbox'):
                    return None
                return ((face.bbox[0] + face.bbox[2]) / 2, (face.bbox[1] + face.bbox[3]) / 2)

            def is_consistent_with_history(face, history, tolerance=50):
                """Check if face position is consistent with position history"""
                if not history or len(history) < 2:
                    return True
                last_pos = history[-1]
                center = get_face_center(face)
                if not center:
                    return False
                distance = np.sqrt((center[0] - last_pos[0])**2 + (center[1] - last_pos[1])**2)
                return distance < tolerance

            # Detectar si hay múltiples caras muy cerca (escenario de beso/abrazo)
            nearby_faces_count = 0
            assigned_face = getattr(self, assigned_attr, None) if hasattr(self, assigned_attr) else None
            if assigned_face:
                for face in valid_faces:
                    if face is assigned_face:
                        continue
                    if is_face_too_close(assigned_face, face, MIN_FACE_DISTANCE):
                        nearby_faces_count += 1

            # Si hay caras muy cerca, aumentar estricto en la verificación
            strict_mode = nearby_faces_count > 0

            if not hasattr(self, assigned_attr) or getattr(self, assigned_attr) is None:
                if hasattr(roop.globals, 'selected_face_references') and video_key in roop.globals.selected_face_references:
                    face_ref_data = roop.globals.selected_face_references[video_key]
                    user_bbox = face_ref_data.get('bbox')
                    user_embedding = face_ref_data.get('embedding')
                    user_face_obj = face_ref_data.get('face_obj')

                    best_match = None
                    best_score = -1

                    for face in valid_faces:
                        if not hasattr(face, 'bbox'):
                            continue

                        # Verificar que la cara no esté demasiado cerca de otras caras (evita confusión en besos)
                        too_close = False
                        for other_face in valid_faces:
                            if face is other_face:
                                continue
                            if is_face_too_close(face, other_face, MIN_FACE_DISTANCE):
                                too_close = True
                                break

                        if user_bbox is not None:
                            iou = self._bbox_iou(user_bbox, face.bbox)
                            if iou > best_score:
                                best_score = iou
                                best_match = face

                        if user_embedding is not None and hasattr(face, 'embedding') and face.embedding is not None:
                            emb_score = self._calculate_similarity(user_embedding, face.embedding)
                            if emb_score > best_score:
                                best_score = emb_score
                                best_match = face

                    # Si hay caras muy cerca, priorizar la que mejor coincida con el embedding
                    if best_match and too_close and user_embedding is not None:
                        # Recalcular solo con embedding para evitar ambiguity
                        best_emb_score = -1
                        best_emb_match = None
                        for face in valid_faces:
                            if not hasattr(face, 'embedding') or face.embedding is None:
                                continue
                            emb_score = self._calculate_similarity(user_embedding, face.embedding)
                            if emb_score > best_emb_score:
                                best_emb_score = emb_score
                                best_emb_match = face

                        if best_emb_match and best_emb_score > 0.02:
                            best_match = best_emb_match
                            best_score = best_emb_score

                    if best_match and best_score > 0.03:
                        setattr(self, assigned_attr, best_match)
                        if hasattr(best_match, 'embedding') and best_match.embedding is not None:
                            emb = np.array(best_match.embedding, dtype=np.float32)
                            norm = np.linalg.norm(emb)
                            if norm > 0:
                                setattr(self, original_embedding_attr, emb / norm)

                        face_center = ((best_match.bbox[0] + best_match.bbox[2]) / 2,
                                       (best_match.bbox[1] + best_match.bbox[3]) / 2)
                        setattr(self, position_history_attr, [face_center])
                        setattr(self, tracking_lost_attr, 0)

                        print(f"[TRACK] Primer frame: cara seleccionada, score={best_score:.2f}")
                        return best_match

            if hasattr(self, assigned_attr):
                assigned_face = getattr(self, assigned_attr)
                if assigned_face and hasattr(assigned_face, 'bbox'):
                    best_match = None
                    best_score = -1

                    expected_center = None
                    if len(position_history) >= 2:
                        last_pos = position_history[-1]
                        prev_pos = position_history[-2]
                        velocity = (last_pos[0] - prev_pos[0], last_pos[1] - prev_pos[1])
                        expected_center = (last_pos[0] + velocity[0], last_pos[1] + velocity[1])

                    for face in valid_faces:
                        if not hasattr(face, 'bbox'):
                            continue

                        # Verificar que la cara no esté demasiado cerca de otras caras
                        too_close = False
                        for other_face in valid_faces:
                            if face is other_face:
                                continue
                            if is_face_too_close(face, other_face, MIN_FACE_DISTANCE):
                                too_close = True
                                break

                        # Si está muy cerca de otra cara, aumentar el umbral para esta cara
                        proximity_penalty = 0.0
                        if too_close:
                            # Penalizar caras que están muy cerca de otras
                            proximity_penalty = 0.08

                        iou = self._bbox_iou(assigned_face.bbox, face.bbox)

                        emb_score = 0.0
                        original_embedding = getattr(self, original_embedding_attr, None)

                        if hasattr(face, 'embedding') and face.embedding is not None:
                            if original_embedding is not None:
                                emb_score = self._calculate_similarity(original_embedding, face.embedding)
                            else:
                                if hasattr(assigned_face, 'embedding') and assigned_face.embedding is not None:
                                    emb_score = self._calculate_similarity(assigned_face.embedding, face.embedding)

                        # Si hay caras cerca, requerir embedding score mayor
                        min_emb_required = 0.03 if not too_close else 0.05

                        position_bonus = 0.0
                        if expected_center:
                            face_center = get_face_center(face)
                            if face_center:
                                distance = np.sqrt((face_center[0] - expected_center[0])**2 + (face_center[1] - expected_center[1])**2)
                                max_distance = 200
                                position_bonus = max(0, 0.10 * (1 - distance / max_distance))

                        # Solo considerar caras con embedding score suficiente cuando hay conflicto
                        if too_close and emb_score < min_emb_required:
                            score = -1  # Descartar esta cara
                        elif emb_score > 0:
                            score = (iou * 0.30) + (emb_score * 0.60) + position_bonus - proximity_penalty
                        else:
                            score = (iou * 0.90) + position_bonus - proximity_penalty

                        if score > best_score:
                            best_score = score
                            best_match = face

                    if best_match and best_score >= MIN_SCORE_THRESHOLD:
                        face_center = get_face_center(best_match)
                        if face_center:
                            position_history.append(face_center)
                            if len(position_history) > 10:
                                position_history.pop(0)
                            setattr(self, position_history_attr, position_history)

                        if best_score >= HIGH_CONFIDENCE_THRESHOLD:
                            setattr(self, assigned_attr, best_match)
                        elif strict_mode:
                            # En modo estricto, solo actualizar si hay alta confianza
                            if best_score >= 0.05:
                                setattr(self, assigned_attr, best_match)

                        setattr(self, tracking_lost_attr, 0)

                        if strict_mode:
                            print(f"[TRACK] Frame {frame_count}: cara encontrada (modo estricto, score={best_score:.2f}, caras_cerca={nearby_faces_count})")
                        return best_match

            lost_count += 1
            setattr(self, tracking_lost_attr, lost_count)

            # Si hay demasiados fallos seguidos, resetear el tracking completamente
            MAX_CONSECUTIVE_FAILURES = 15
            if lost_count >= MAX_CONSECUTIVE_FAILURES:
                print(f"[TRACK] Reset completo después de {lost_count} fallos")
                # Resetear todo y empezar de nuevo
                setattr(self, assigned_attr, None)
                setattr(self, tracking_lost_attr, 0)
                setattr(self, position_history_attr, [])
                return None

            original_embedding = getattr(self, original_embedding_attr, None) if hasattr(self, original_embedding_attr) else None

            if original_embedding is None:
                if hasattr(roop.globals, 'selected_face_references') and video_key in roop.globals.selected_face_references:
                    face_ref_data = roop.globals.selected_face_references[video_key]
                    user_embedding = face_ref_data.get('embedding')
                    if user_embedding is not None:
                        emb = np.array(user_embedding, dtype=np.float32)
                        norm = np.linalg.norm(emb)
                        if norm > 0:
                            original_embedding = emb / norm
                            setattr(self, original_embedding_attr, original_embedding)

            # Reacquire con embedding - requerir score mayor si hay caras cerca
            min_reacquire_score = 0.03 if not strict_mode else 0.06

            if original_embedding is not None:
                best_reacquire = None
                best_reacquire_score = -1

                for face in valid_faces:
                    if not hasattr(face, 'embedding') or face.embedding is None:
                        continue

                    # Verificar si esta cara está muy cerca de otra
                    face_too_close = False
                    for other_face in valid_faces:
                        if face is other_face:
                            continue
                        if is_face_too_close(face, other_face, MIN_FACE_DISTANCE):
                            face_too_close = True
                            break

                    emb_score = self._calculate_similarity(original_embedding, face.embedding)

                    # Si está muy cerca de otra cara, requerir embedding mayor
                    if face_too_close and emb_score < 0.06:
                        continue

                    if emb_score > best_reacquire_score:
                        best_reacquire_score = emb_score
                        best_reacquire = face

                if best_reacquire and best_reacquire_score >= min_reacquire_score:
                    setattr(self, assigned_attr, best_reacquire)
                    setattr(self, tracking_lost_attr, 0)

                    face_center = get_face_center(best_reacquire)
                    if face_center:
                        position_history = [face_center]
                        setattr(self, position_history_attr, position_history)

                    print(f"[REACQUIRE] Frame {frame_count}: emb={best_reacquire_score:.2f} ({'modo estricto' if strict_mode else 'normal'})")
                    return best_reacquire

            # En modo selected_faces_frame, NO usar fallback a la cara más grande
            # Si no se encuentra la cara seleccionada, se debe omitir el frame
            if lost_count == 1 or lost_count % 50 == 0:
                print(f"[LOST] Tracking perdido #{lost_count}")
            return None

        except Exception as e:
            print(f"[ERROR] _find_target_face_for_selected_mode: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _bbox_iou(self, a, b):
        """Compute IoU between two bboxes (x1,y1,x2,y2). Returns 0 if invalid."""
        try:
            if not a or not b:
                return 0.0
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih
            area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
            area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
            union = area_a + area_b - inter
            if union <= 0:
                return 0.0
            return inter / union
        except Exception:
            return 0.0

    def _select_intelligent_face(self, target_face, candidate_faces):
        try:
            if not candidate_faces:
                return None
                
            best_face = None
            best_similarity = -1
            
            similarity_threshold = getattr(roop.globals, 'face_similarity_threshold', 0.2)
            use_gender_filter = getattr(roop.globals, 'use_gender_filter', True)
            
            target_emb = None
            if hasattr(target_face, "embedding") and target_face.embedding is not None:
                target_emb = np.array(target_face.embedding, dtype=np.float32)
                if np.linalg.norm(target_emb) > 0:
                    target_emb = target_emb / np.linalg.norm(target_emb)
            
            for face in candidate_faces:
                similarity = 0.0
                
                if target_emb is not None and hasattr(face, 'embedding') and face.embedding is not None:
                    similarity = self._calculate_similarity(target_emb, face.embedding)
                    
                    if use_gender_filter:
                        target_gender = get_gender(target_face)
                        face_gender = get_gender(face)
                        if target_gender and face_gender and target_gender == face_gender:
                            similarity += 0.1
                    
                    if similarity < similarity_threshold:
                        continue
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_face = face
                    
            return best_face if best_face else candidate_faces[0]
        except Exception as e:
            return candidate_faces[0] if candidate_faces else None

    def _process_face_swap_v21(self, source_face, target_face, result_frame, original_frame, enable_temporal_smoothing=False):
        ProcessMgr._swap_call_count += 1
        call_num = ProcessMgr._swap_call_count
        try:
            if not hasattr(source_face, 'embedding') or source_face.embedding is None:
                return original_frame
                
            if not hasattr(target_face, 'bbox') or target_face.bbox is None:
                return original_frame
            
            # ============================================
            # NUEVO: DETECTAR BOCA ABIERTA ANTES DEL SWAP
            # ============================================
            preserve_mouth = getattr(roop.globals, 'preserve_mouth_expression', True)
            mouth_open = False
            mouth_region = None
            mouth_open_ratio = 0.0
            
            if preserve_mouth:
                from roop.processors.FaceSwap import detect_mouth_open, create_mouth_preservation_mask
                landmarks_106 = getattr(target_face, 'landmark_106', None)
                mouth_open, mouth_region, mouth_open_ratio = detect_mouth_open(target_face, landmarks_106)
                
                if mouth_open:
                    print(f"[MOUTH_PRESERVE] Frame {call_num}: Boca abierta (ratio={mouth_open_ratio:.2f})")
            
            # Obtener bbox de la cara
            x1, y1, x2, y2 = target_face.bbox
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(result_frame.shape[1], int(x2)), min(result_frame.shape[0], int(y2))
            
            if x2 <= x1 or y2 <= y1:
                return original_frame
            
            # ============================================
            # 1. FACE SWAP
            # ============================================
            res = self.processors["faceswap"].Run(source_face, target_face, result_frame, paste_back=True)
            if res is None:
                return original_frame
            result_frame = res
            
            # Extraer región swappeada
            swapped_face = result_frame[y1:y2, x1:x2].copy()
            original_face_region = original_frame[y1:y2, x1:x2].copy()
            
            # ============================================
            # 2. ENHANCER (aplicar con BLENDING CONSERVADOR para evitar efectos raros)
            # ============================================
            use_enhancer = getattr(roop.globals, 'use_enhancer', True)
            selected_enhancer = getattr(roop.globals, 'selected_enhancer', 'GFPGAN')
            
            enhancer_key = None
            if selected_enhancer == "GFPGAN":
                enhancer_key = "enhance_gfpgan"
            elif selected_enhancer == "CodeFormer":
                enhancer_key = "enhance_codeformer"
            elif selected_enhancer == "Restoreformer++":
                enhancer_key = "enhance_restoreformer"
            elif selected_enhancer == "GPEN":
                enhancer_key = "enhance_gpen"
            
            # NUEVO: Factor de blending del enhancer (más bajo para preservar más la cara original)
            # Esto evita el efecto "blur" o "flash" causado por el enhancer
            enhancer_blend_factor = getattr(roop.globals, 'enhancer_blend_factor', 0.3)  # Por defecto 30% enhancer, 70% swap original
            
            if use_enhancer and enhancer_key and enhancer_key in self.processors:
                try:
                    class MockFaceSet:
                        pass
                    mock_faceset = MockFaceSet()
                    
                    enhancer_result = self.processors[enhancer_key].Run(mock_faceset, None, swapped_face)
                    
                    if enhancer_result is not None:
                        if isinstance(enhancer_result, tuple):
                            enhanced_face = enhancer_result[0]
                        else:
                            enhanced_face = enhancer_result
                        
                        if enhanced_face is not None:
                            # Resize al tamaño correcto
                            if enhanced_face.shape[:2] != swapped_face.shape[:2]:
                                enhanced_face = cv2.resize(enhanced_face, (swapped_face.shape[1], swapped_face.shape[0]), cv2.INTER_LANCZOS4)
                            
                            # NUEVO: Blending más conservador del enhancer
                            # Esto evita el efecto flash y preserva más la identidad de la cara swappeada
                            region_h, region_w = enhanced_face.shape[:2]
                            soft_mask = create_soft_mask((0, 0, region_w, region_h), (region_h, region_w), feather=30)
                            
                            # Aplicar factor de blending conservador
                            mask_3ch = np.stack([soft_mask] * 3, axis=-1)
                            # blend_factor controla cuánto del enhanced_face se mezcla con swapped_face
                            # 0.3 significa 30% enhanced + 70% original swap
                            swapped_face = (enhanced_face.astype(np.float32) * enhancer_blend_factor * mask_3ch + 
                                           swapped_face.astype(np.float32) * (1 - enhancer_blend_factor * mask_3ch)).astype(np.uint8)
                            
                            print(f"[QUALITY] {selected_enhancer} aplicado (blend={enhancer_blend_factor})")
                except Exception as e:
                    print(f"[ENHANCER] Error: {e}")
            
            # ============================================
            # 3. COLOR MATCHING Y AJUSTE DE BRILLO
            # ============================================
            # NOTA: Reducimos estos valores para preservar mejor la identidad de la cara origen
            # Si el color matching es muy fuerte, la cara swappeada se parece menos a la origen
            try:
                if swapped_face.size > 0 and original_face_region.size > 0:
                    # NUEVO: Valor más bajo para preservar más la identidad original
                    # Esto evita que la cara swappeada pierda parecido con la origen
                    brightness_strength = getattr(roop.globals, 'brightness_strength', 0.15)
                    swapped_face = adjust_face_brightness(swapped_face, original_face_region, strength=brightness_strength)
                    
                    # NUEVO: Color matching más conservador para preservar los colores de la cara origen
                    # Valores muy bajos de color matching hacen que la cara se parezca menos a la origen
                    color_match_strength = getattr(roop.globals, 'color_match_strength', 0.15)
                    use_color_match = getattr(roop.globals, 'use_color_matching', True)
                    if use_color_match:
                        swapped_face = match_color_histogram(swapped_face, original_face_region, blend_factor=color_match_strength)
            except Exception as e:
                print(f"[COLOR_MATCH] Error: {e}")
            
            # ============================================
            # 4. PRESERVACIÓN DE BOCA
            # ============================================
            if mouth_open and mouth_region is not None:
                try:
                    mouth_mask = create_mouth_preservation_mask(original_frame, mouth_region, blend_ratio=0.85)
                    mouth_mask_region = mouth_mask[y1:y2, x1:x2]
                    mouth_mask_3ch = np.expand_dims(mouth_mask_region, axis=2)
                    
                    # Combinar: swapped_face donde mouth_mask=1, original donde mouth_mask=0
                    swapped_face = (swapped_face.astype(np.float32) * mouth_mask_3ch + 
                                   original_face_region.astype(np.float32) * (1 - mouth_mask_3ch)).astype(np.uint8)
                except Exception as e:
                    print(f"[MOUTH_PRESERVE] Error: {e}")
            
            # ============================================
            # 5. BLENDING FINAL CON MÁSCARA SUAVE
            # ============================================
            # Reducido feather de 30 a 15 para preservar más la identidad de la cara swappeada
            try:
                # Crear máscara suave elíptica para el blending final
                face_h, face_w = y2 - y1, x2 - x1
                soft_mask = create_soft_mask((x1, y1, x2, y2), result_frame.shape[:2], feather=15)
                
                # Aplicar blending suave con el frame original
                mask_3ch = np.stack([soft_mask] * 3, axis=-1)
                result_frame = (swapped_face.astype(np.float32) * mask_3ch[y1:y2, x1:x2] + 
                               original_face_region.astype(np.float32) * (1 - mask_3ch[y1:y2, x1:x2])).astype(np.uint8)
                
                # Colocar de vuelta en el frame
                full_result = original_frame.copy()
                full_result[y1:y2, x1:x2] = result_frame
                result_frame = full_result
                
            except Exception as e:
                print(f"[BLENDING] Error: {e}")
                # Fallback: simplemente colocar la cara
                result_frame[y1:y2, x1:x2] = swapped_face
            
            return result_frame if result_frame is not None else original_frame
            
        except Exception as e:
            print(f"[DEBUG] Error en _process_face_swap_v21: {e}")
            return original_frame
            
        finally:
            # Limpieza de memoria
            import gc
            gc.collect()

    def _get_gender_confidence(self, face):
        try:
            if hasattr(face, 'gender_confidence'):
                return float(face.gender_confidence)
            
            if hasattr(face, 'gender'):
                gender_value = face.gender
                if isinstance(gender_value, (list, np.ndarray)):
                    gender_value = gender_value[0] if len(gender_value) > 0 else None
                
                if gender_value is not None:
                    gender_value = float(gender_value)
                    
                    if gender_value == 0 or gender_value == 0.0 or gender_value == 1 or gender_value == 1.0:
                        return 0.95
                    
                    if 0.0 < gender_value < 1.0:
                        if gender_value < 0.35 or gender_value > 0.65:
                            return 0.90
                        elif gender_value < 0.45 or gender_value > 0.55:
                            return 0.75
                        else:
                            return 0.50
            
            return 0.50
            
        except Exception as e:
            return 0.50

    def _get_face_tracking_key(self, face):
        try:
            if hasattr(face, 'bbox') and face.bbox is not None:
                x1, y1, x2, y2 = face.bbox
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                return f"face_{center_x}_{center_y}"
            return f"id_{id(face)}"
        except Exception:
            return f"fallback_{id(face)}"
    
    def _get_face_key(self, face):
        """Genera una clave única para una cara basada en su posición (para evitar doble procesamiento)"""
        try:
            if hasattr(face, 'bbox') and face.bbox is not None:
                x1, y1, x2, y2 = face.bbox
                # Usar coordenadas discretizadas para tolerate small movements
                grid_size = 20
                grid_x = int((x1 + x2) / 2 / grid_size) * grid_size
                grid_y = int((y1 + y2) / 2 / grid_size) * grid_size
                return f"face_{grid_x}_{grid_y}"
            return f"id_{id(face)}"
        except Exception:
            return f"fallback_{id(face)}"

    def _calculate_similarity(self, emb1, emb2):
        try:
            if emb1 is None or emb2 is None:
                return 0.0
                
            emb1 = np.array(emb1, dtype=np.float32)
            emb2 = np.array(emb2, dtype=np.float32)
            
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            emb1_norm = emb1 / norm1
            emb2_norm = emb2 / norm2
            
            similarity = np.dot(emb1_norm, emb2_norm)
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            return 0.0

    def run_batch_inmem(self, video_path, output_path, start_frame=0, end_frame=None, fps=24.0, num_threads=1, skip_audio=False):
        try:
            if not self.is_initialized:
                print("ProcessMgr not initialized")
                return
            
            import cv2
            import os
            from pathlib import Path

            if self.options:
                self.options.current_video_path = video_path
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Could not open video: {video_path}")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_video = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if end_frame is None or end_frame > total_frames:
                end_frame = total_frames
            if start_frame < 0:
                start_frame = 0

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            temp_output = output_path + ".temp.mp4"
            out = cv2.VideoWriter(temp_output, fourcc, fps_video, (width, height))

            if not out.isOpened():
                print(f"Could not create output video: {temp_output}")
                cap.release()
                return

            print(f"Processing video: {os.path.basename(video_path)} ({start_frame}-{end_frame}/{total_frames} frames)")

            frame_count = 0
            processed_frames = 0
            current_frame = start_frame
            total_frames_to_process = end_frame - start_frame
            
            yield (0, "Iniciando procesamiento de video...")
            
            while current_frame < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    processed_frame = self.process_frame(frame, enable_temporal_smoothing=True)
                    if processed_frame is not None:
                        out.write(processed_frame)
                        processed_frames += 1
                    else:
                        out.write(frame)
                except Exception as e:
                    print(f"[ERROR] Frame {frame_count}: error - {e}")
                    out.write(frame)

                frame_count += 1
                current_frame += 1

                if frame_count % 2 == 0 or frame_count == total_frames_to_process:
                    progress_percent = (frame_count / total_frames_to_process) * 100 if total_frames_to_process > 0 else 0
                    msg = f"Procesando frame {frame_count}/{total_frames_to_process} ({progress_percent:.1f}%)"
                    yield (progress_percent, msg)
                    
                    if is_valid_progress_callback(self.progress_callback):
                        try: self.progress_callback(progress_percent, desc=msg)
                        except: pass

            cap.release()
            out.release()
            
            yield (100, "Frames procesados, finalizando video...")

            if not skip_audio and os.path.exists(temp_output):
                yield (100, "Fusionando audio...")
                try:
                    from roop.util_ffmpeg import restore_audio
                    restore_audio(temp_output, video_path, None, None, output_path)
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
                    print(f"Audio merged successfully: {output_path}")
                except Exception as e:
                    print(f"Audio merge failed: {e}")
                    if os.path.exists(temp_output):
                        try: os.rename(temp_output, output_path)
                        except: pass
            else:
                if os.path.exists(temp_output):
                    try: os.rename(temp_output, output_path)
                    except: pass

            yield (100, "Video completado con éxito")

        except Exception as e:
            print(f"Video processing error: {e}")
            import traceback
            traceback.print_exc()
            yield (100, f"Error: {str(e)}")

    def Release(self):
        try:
            for processor in self.processors.values():
                if hasattr(processor, "Release"):
                    processor.Release()
            self.processors.clear()
            self.source_embeddings_cache.clear()
            self.face_assignment_cache.clear()
            self.face_position_history.clear()
            self.global_source_for_all_id = None
            self.selected_face_assignment_cache.clear()
        except:
            pass

    def _log_audio_merge_error(self, error, video_path, temp_output):
        error_type = type(error).__name__
        error_message = str(error)
        
        audio_exists = os.path.exists(video_path)
        temp_exists = os.path.exists(temp_output)
        
        print(f"[AUDIO_ERROR] Tipo: {error_type}")
        print(f"[AUDIO_ERROR] Mensaje: {error_message}")
        print(f"[AUDIO_ERROR] Archivo de video existe: {audio_exists}")
        print(f"[AUDIO_ERROR] Archivo temporal existe: {temp_exists}")
        
        if not audio_exists:
            print(f"[AUDIO_ERROR] El archivo de video no existe: {video_path}")
        if not temp_exists:
            print(f"[AUDIO_ERROR] El archivo temporal no existe: {temp_output}")

    def release_resources(self):
        self.Release()
