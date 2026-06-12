# AGENTS.md — AutoAuto Face Swap Project

## Goal
- Fix white rectangle/ellipse artifact (model input normalization)
- Fix `'NoneType' object is not callable` error (Face copy bug)
- Corregir halo sharp alrededor de la cara, mejorar tracking y perfiles en movimiento, y revisar oclusión de video/boca cuando se introducen objetos.

## Constraints & Preferences
- Usar `inswapper_128_facefusion.onnx`, XSeg Masker, GFPGAN Enhancer, MediaPipe Face Mesh 468 landmarks, FFmpeg con libx264 CRF 14.
- Modo `selected_faces` con 655 facesets de origen, 1 archivo destino (SSondoson_...mp4, 1280x720, 850 frames).
- GPU NVIDIA GeForce RTX 3060 Ti, providers: CUDAExecutionProvider, CPUExecutionProvider.

## Progress
### Done
- **v5.13**: Content feathering — tent `2.0→1.5` (zona de transición más ancha), `content_blur_sz=max(31, blur_sz+10)` (blur más fuerte en transición). Blur_sz general `max(31,w//8)→max(41,w//6)` para máscara más suave. EMA smoothing en `master_weight` (coeff 0.08) y `swap_weight` (coeff 0.15) para eliminar flicker "popco" en transiciones perfil/frontal y fluctuaciones de track_score. Resultado: halo aún presente, popco reducido.
- **v5.12**: ADN Maestro aumentado `0.35/0.40→0.45/0.50` (frontal/perfil) y fallback `0.55→0.65` — más identidad source. Content feathering blur ampliado `15→max(21, blur_sz)` para eliminar halo remanente.
- **v5.11**: Content feathering — blur `warped_face` en la zona de transición del mask (`0 < final_mask < 1`) para eliminar mismatch de textura GFPGAN vs fondo. Usa tent function `clip(1 - |mask-0.5|*2, 0, 1)` como peso de blending.
- **v5.11**: Enhancer blend reducido `0.55→0.40` (GFPGAN ~38% efectivo con user_blend~0.945) + sharpening eliminado por completo.
- **v5.11**: `effective_threshold` bajado `0.20→0.10` en `face_util_rotation.py` para que RetinaFace detecte caras en perfiles extremos (frame 489-490).
- **v5.11**: Rotación (90°, 270°) ahora se prueba **siempre** (no solo cuando orientación original no detecta nada) — captura perfiles que el detector frontal omite.
- **v5.11**: Ghost tracking mejorado: `max_shift=60→80`, `inertia=0.85→0.90`, umbral de re-adquisición `rec_threshold=0.35→0.25` para reconectar más fácil tras pérdida.
- **v5.11**: Umbrales progresivos de re-adquisición periódica más permisivos (`0.15→0.12`, `0.10→0.08`).
- **v5.10**: Sharpening mínimo `1.08→1.02` + detail enhancement eliminado por completo (segundo pase de unsharp que acentuaba bordes de máscara).
- **v5.10**: `blur_sz` aumentado a `max(31, w//8)` (antes `max(21, w//12)`) — aproximadamente +50% de radio de difuminado.
- **v5.10**: Eliminado `final_mask * warped_content` — redundante porque `warped_face` ya está compuesto con `original_frame` vía `wc3`; eliminaba la multiplicación de Gaussianas que creaba transición más sharp.
- **v5.10**: Eliminado `final_mask * (1.0 - warped_edge_mask * 0.15)` — tercera transición innecesaria que apilaba efecto sharp.
- **v5.10**: Eliminado blur post-edge `(5,5)` — sobrante sin las multiplicaciones previas.
- **v5.10**: Tracking `min_track_score` para perfiles `0.35→0.27`, `base_threshold` para perfiles `max(350,0.8*)→max(400,0.9*)` y cámara en movimiento `max(400,0.9*)→max(450,1.0*)`.
- **v5.10**: Refuerzo de máscara débil: blur `(9,9)→(11,11)`.
- v5.9: `_is_profile_face` corregida para requerir 2/3 condiciones (eye_ratio<0.16, eye_y>0.55, nose_asymmetry<0.15) o nose_asymmetry<0.10 extrema.
- v5.9: Blurs aumentados en toda la cadena de máscara (warped_content, edge, ellipse, combined_face, blur_sz).
- v5.9: Edge attenuation reducida `0.40→0.25`.
- v5.9: `quality_enhancements.py` — feather por defecto 25→35, Poisson blur (9,9)→(15,15), padding 12→18, oclusión post-process mejorado.

### Done
- **v5.41**: **Fix white rectangle/ellipse** — input normalization: `aimg_tgt` must be [0,1] not [0,255] (both `inswapper_128_facefusion.onnx` and `inswapper_128.onnx` expect this). Fix at `FaceSwap.py:240`.
- **v5.41**: **Fix `'NoneType' object is not callable`** — `copy.copy(insightface.Face)` crashes because Face is a `dict` subclass with `__setstate__=None`. Fix: replace `copy.copy(source_face)` with `Face(source_face)` (dict copy via constructor). Fix at `ProcessMgr.py:1685`.
- **v5.37**: **Fix swap pequeño/gris (direction corregido)**. Bug: `M *= scale_factor` multiplica M (frame→face). El warp-back usa `M_inv = invertAffineTransform(M)`. Escalar M→ `M_inv_lin /= S` → coords de face / S en vez de × S → lectura incorrecta. **Solución**: `M[0:2,0:2] /= scale_factor` → `M_inv_lin *= S` → coords 2× en 512×512 = features correctas. FaceSwap: modelo 128→256, `M *= 2` (mapea 256_crop→frame). GFPGAN: 256→512, features doblan en px → `M_lin /= 2`. Posición invariante (M_inv_trans no cambia con S) — v5.36 ya posición correcta pero contenido gris.
- **v5.36**: **Fix swap pequeño/gris (primer intento — descartado)**. En `_process_face_swap_v21`, el `scale_factor` para escalar M cuando GFPGAN upscalea usaba `128.0` hardcoded, pero `swapped_face_aligned` ya fue redimensionado de 128→256 en `FaceSwap.Run()` (FaceSwap.py:262). Cuando GFPGAN produce 512x512, `scale_factor = 512/128 = 4` en vez de `512/256 = 2`. Esto duplica el escalado de M, haciendo que `M_inv` proyecte a la mitad del tamaño correcto → swap aparece al 50% del tamaño y desplazado. **Solución**: guardar `orig_face_h` antes del resize y usar `scale_factor = enhanced.shape[0] / orig_face_h`.
- **v5.35**: **Fix real del halo oscuro**: el `warped_face` se genera con `cv2.BORDER_CONSTANT` (fondo negro) en el warp. Donde la máscara (extendida por GaussianBlur) supera el área válida del warp (7% de píxeles, ~7,000 px), `warped_face=0` pero `final_mask>0`, creando `resultado=original*(1-mask)` → halo oscuro de ~8.8 unidades. **Solución**: cambiar a `cv2.BORDER_REFLECT` para que los píxeles fuera del warp sean reflejados del borde de la cara (color piel) en vez de 0. Esto elimina el oscurecimiento en la transición sin crear bordes sharp.
- **v5.34**: **INEFECTIVO** — `np.minimum(final_mask, _warp_valid)` no cambia nada porque ambos usan el mismo `blur_sz`.
- **v5.33**: **REVERTIDO en v5.34** — rellenar warped_face con original causaba parpadeo.
- **v5.32**: Fix broadcasting error (warped_face 3ch vs final_mask 1ch) usando `np.max(..., axis=2)` para reducir a single-channel. ADN Maestro eliminado (master_weight=0) — 100% source embedding puro, máxima identidad source posible. Anti-halo: erosión de máscara (0.06 threshold shift) + blur amplio (blur_mult 14/16→6, min kernel 11→15) para que la transición sea invisible aunque GFPGAN añada textura distinta. Enhancer_blend reducido 0.50→0.30 (GFPGAN ~27% efectivo con user_blend=1.0). Preservación de boca más agresiva: m_blend max 0.40→0.55 (base), 0.65→0.85 (con oclusión) para mostrar objetos en la boca.
- **v5.31**: Fix máscara cuadrada y swap invisible. Elipse ampliada (0.44→0.55 frontal, 0.37→0.50 perfil) + altura (0.48→0.52). Aporte de elipse 0.80→1.0. Dilatación XSeg (3x3, 1 iter). Bugfix warped_content→warped_face (NameError cuando mask < 0.35). Enhancer_blend 0.30→0.50. Blur final reducido (blur_mult 20/18→16/14, min 7/9→11). Fallback elipse frontal 0.46→0.54, perfil 0.42→0.50.
- **v5.18**: Eliminación definitiva de halo en perfiles mediante erosión de máscara (7x7) y truncamiento de tail agresivo (0.12). Esto encoge el swap hacia el interior de la cara evitando que el borde toque el fondo original. Optimizado para modelos de 128px que tienden a desalinear bordes en perfiles.
- **v5.17**: Corrección de halo en perfiles. Estrechar elipse de respaldo (0.55→0.42), aumentar top-fade (10%→15%), reducir dilatación XSeg (4%→2.5%) y aumentar color_match_strength (0.20→0.35) para fundir mejor con el cuello/oreja. Aumentado el desenfoque final de máscara en perfiles para transición invisible.
- **v5.16**: Alpha blending espacial del resultado de Poisson. OpenCV `seamlessClone` trata la máscara como binaria, lo que creaba una costura dura (borde recortado tipo sticker) muy visible en movimiento. Mezclar `poisson_result` con `original_frame` usando `final_mask * swap_weight` como máscara de transparencia suaviza completamente la costura. Reducido el truncamiento del tail `0.08→0.05` para ampliar el rango de blending.
- **v5.15**: Poisson NORMAL_CLONE + content feathering eliminado. NORMAL_CLONE preserva textura source completa y mezcla solo bordes del mask. Eliminar content feathering evita que difumine los gradientes. Además: brightness_strength 0.15→0.25, color_match_strength frontal 0.30→0.40, blur_sz min 41→25, mask tail 0.05→0.08, y eliminado el re-blur interno (15,15) de blend_with_poisson que re-extendía la máscara anulando el tail truncation.
- **v5.14**: Mask tail truncation 0.05 para recortar cola Gaussiana de máscara + enhancer_blend reducido `0.40→0.30` (GFPGAN ~29% efectivo). Poisson MIXED_CLONE REVERTIDO — content feathering elimina los gradientes que Poisson necesita, resultando en swap invisible. Se mantiene alpha blend clásico.

### Known Issues
- **Popco reducido**: EMA smoothing mantiene estabilidad.
- **`copy.copy(Face)` crashes**: `insightface.app.common.Face` is a `dict` subclass with `__setstate__=None`. Calling `copy.copy(face_obj)` raises `TypeError: 'NoneType' object is not callable`. Fix: use `Face(source_face)` dict constructor instead.
- **Both model variants need [0,1] input**: Both `inswapper_128_facefusion.onnx` and `inswapper_128.onnx` expect target input normalized to [0,1] range. Original code passed raw [0,255] — model activations saturate → ~93% white output.

## Key Decisions
- **v5.37**: `M *= scale_factor` es INCORRECTO (multiplica M frame→face; M_inv_lin se divide, dando coords más pequeñas). La dirección correcta es `M[0:2,0:2] /= scale_factor` → M_inv_lin *= escala → coords 2× más grandes en la cara ampliada = features correctas.
- **v5.36**: El `scale_factor` para M debe usar el tamaño REAL de `swapped_face_aligned` (256px, no 128px) porque `FaceSwap.Run()` ya lo redimensiona de 128→256. Guardar `orig_face_h` antes del resize es necesario para el scale_factor, pero la dirección (multiplicar vs dividir) es el error real.
- **v5.35**: BORDER_REFLECT elimina el halo oscuro sin crear bordes sharp ni necesidad de recortar la máscara o rellenar warped_face.
- **v5.34 Root cause confirmado**: El `GaussianBlur` final de la máscara (tras la erosión) extiende valores de máscara más allá de la región warp válida. **Fix incorrecto**: `np.minimum(final_mask, GaussianBlur(_warp_valid))` no funcionó porque ambos blur tienen el mismo kernel → no hay clipping efectivo.
- **v5.16 Costuras duras por cv2.seamlessClone**: El algoritmo de clonación de Poisson de OpenCV trata el mask como binario, ignorando el degradado de transparencia (alpha). Para resolverlo, realizamos un blending alpha clásico sobre el resultado de Poisson usando la máscara suave.
- **v5.14 Root cause del halo sharp**: Content feathering difumina gradualmente pero no elimina el mismatch de textura en el borde. Poisson MIXED_CLONE iguala gradientes en el borde para integración perfecta. Enhancer_blend 0.30 reduce la textura diferente de GFPGAN.
- **v5.14 Poisson NO viable con content feathering**: Content feathering blur elimina los gradientes de textura del warped_face en la zona de transición. Poisson MIXED_CLONE necesita esos gradientes para guiar la mezcla. Resultado: swap invisible. Mantener alpha blend.
- **v5.15 NORMAL_CLONE viable sin content feathering**: Al eliminar content feathering, NORMAL_CLONE preserva la textura source completa y solo mezcla bordes del mask. No compite con fondos porque usa el color source directamente en la zona del mask.
- **v5.13**: EMA smoothing en master_weight y swap_weight para eliminar popco en transiciones perfil/frontal y fluctuaciones de track_score.
- **v5.11 Root cause del halo sharp**: Aunque el mask alpha sea smooth, la textura diferente del rostro GFPGAN vs el fondo original crea una transición visible. Content feathering ataca esto.
- **Root cause del tracking fallido (489-490)**: RetinaFace no detecta la cara por giro extremo. Ghost tracking extendido como respaldo.
- No usar MIXED_CLONE con content feathering activo; NORMAL_CLONE no compensa fondo.

## Next Steps
- Probar v5.41 con el video real (SSondoson) — ambos fixes activos ([0,1] norm + Face copy fix).
- Si la identidad source sigue sin verse bien, considerar reducir `enhancer_blend` de 0.30 a 0.15.
- Si la máscara es débil (face_roi bajo), reducir blur/erosión acumulados en la cadena de máscara.

## Critical Context
- v5.11 log: procesamiento completado en `447.41s (1.90 fps)` — más lento que v5.10 (335.27s) por rotaciones siempre activas, pero tracking mejoró drásticamente: frame 653 ya NO falla (Success score=0.85, dist=2.7px), ghost tracking en 489-491 con re-adquisición a score=0.46.
- `[DEBUG_MASK] Mask mean=0.072 face_roi=0.863 max=1.000 applied` — valores casi idénticos a v5.9 (0.072/0.864), indicando que la máscara post-blur no cambió estadísticamente.
- Los fails de tracking en frame 489-490 son idénticos entre v5.9 y v5.10: scores previos iguales (488: score=0.73, dist=12.7px), mismo patrón de recuperación (491: score=0.67, dist=61.9px). Confirma que el problema es de detección (RetinaFace no encuentra cara), no de umbrales de tracking.
- Muchos `[SOURCE_DETECT] No se detectó cara directamente...` durante carga de facesets — imágenes con caras muy pequeñas o de baja calidad.
- `Invalid SOS parameters for sequential JPEG` aparece en algunos .jpeg pero no interrumpe el proceso.
 - **v5.35 Fix real del halo**: ~7,000 píxeles donde `mask>0` pero `warped_face=0` (negro). El `GaussianBlur` final de la máscara extiende valores de máscara a zonas donde `M_inv` no puede proyectar contenido del source 256×256. Fix v5.33 (rellenar warped_face) causó parpadeo. Fix v5.34 (recortar mask con blurred warp_valid) inefectivo. Fix v5.35 (BORDER_REFLECT en warp) elimina borde negro sin crear artifacts.
- **v5.36 Fix swap pequeño (descartado)**: Usó `orig_face_h` (256) pero multiplicó M. M_inv_lin /= 2 → coordenadas de face 2× más pequeñas → swap lee del área equivocada. Sintomático: `[DEBUG_MASK] Mask mean=0.020 face_roi=0.167` — swap lejos del centro facial → detección débil.
- **v5.37 divide fix**: `M_lin /= scale_factor` → `M_inv_lin *= 2` → lectura correcta de features en 512×512. Elipse gris reportada en v5.36 era posición correcta pero lectura de frente/entre ojos (gris por estar en borde de la cara 512x512).

## Build Process
- Root `AutoAuto.exe` (7MB) is a **PyInstaller one-file launcher** that runs `venv/Scripts/python.exe run.py`.
- Build command: `pyinstaller --onefile --console --icon assets/icon.ico --name AutoAuto --distpath . launcher.py`
- Must build from `venv_ext/` (Python 3.11) to match original `python311.dll`.
- No hidden imports needed: only stdlib (os, sys, subprocess, shutil).
- At runtime, the EXE changes dir to its own location, finds `venv/Scripts/python.exe`, and runs `run.py`.

## Relevant Files
- `launcher.py`: Source for the root EXE (launcher that invokes venv's Python).
- `roop/ProcessMgr.py`: Pipeline principal — `_process_face_swap_v21`, `_is_profile_face`, ghost tracking (`_predict_next_position`), mask/blending (content feathering v5.11), profile detection.
- `roop/face_util_rotation.py`: `get_all_faces_with_rotation` — detección con umbral 0.10 y rotaciones forzadas (v5.11).
- `roop/quality_enhancements.py`: `create_soft_mask`, `detect_foreground_occlusion`, `blend_with_poisson`, `apply_quality_enhancements`.
