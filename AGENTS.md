# AGENTS.md — AutoAuto Face Swap Project

## Goal
- Eliminar blur elíptico fuera de cara, mejorar tracking de perfiles, aumentar nitidez y parecido de source.

## Constraints & Preferences
- `inswapper_128_facefusion.onnx` (128→256px), XSeg Masker, GFPGAN Enhancer (blend=0.80), MediaPipe Face Mesh 468 landmarks, FFmpeg libx264 CRF 14.
- Modo `selected_faces`: 655 facesets origen, 1 destino (1280×720, 850 frames, 30 fps).
- GPU RTX 3060 Ti 8 GB, CUDA 12.4, providers: CUDAExecutionProvider, CPUExecutionProvider.

## Progress
### v5.57 (current) — enhancer_blend 0.80 + unsharp 2.5 + radial GFPGAN fade + profile tracking responsive
- **enhancer_blend 0.80** (antes 0.70): más identidad source (GFPGAN ~76% vs ~66%)
- **Unsharp amount 2.5** (antes 2.0): más nitidez general
- **Radial GFPGAN fade**: centro 100% GFPGAN, bordes raw — elimina textura GAN en zona de transición
- **Profile kps_EMA más responsive**: 0.60/0.70/0.80 (antes 0.45/0.55/0.70) para tracking de giros rápidos
- **M-EMA adaptivo**: m_alpha *= 0.7 en perfiles — matriz afín responde más rápido
- **Erosión 5×5**: encoge máscara ~2-3px para blur no se extienda
- **GaussianBlur //30**: kernel 7px para cara de 200px
- **Content feathering pre-truncation**: tent en toda la transición gaussiana
- **Tail truncation 0.10**: recorta 25% más cola que 0.08
- **Log v5.56**: 430.96s (1.97 fps), enhancer 0.70, mask mean 0.059 ✅
- **Log v5.54**: 431.61s (1.97 fps), enhancer 0.70, [SKIP] visible en frames 565-566 ✅

### Known Issues
- ~60% del video son perfiles → scores bajan a 0.37-0.54; v5.57 mejora responsividad
- Similitud source baja (0.111) — persona diferente al source; v5.57 aumenta enhancer_blend 0.80
- Frames 565-566 skip correcto (det_score < 0.15 + vel > 50px)

## Key Decisions
- Erosión antes de GaussianBlur: el orden correcto es erode → blur (el blur rellena suavemente desde dentro de la cara)
- Radial fade: la textura GFPGAN en bordes causa blur circle; eliminarla en transición es más efectivo que solo encoger máscara
- Perfiles: menos smoothing en KPS y M-EMA = más lag, pero más tracking preciso durante giros

## Next Steps
- Testear v5.57 — tracking de perfiles debe responder más rápido, swap debe verse más nítido y parecido al source
- Verificar mask mean ~0.059 en log
- Si identidad source aún baja, probar enhancer_blend 0.85

## Critical Context
- Log v5.56: 430.96s (1.97 fps), enhancer 0.70, mask mean=0.059 ✅
- Log v5.54: 431.61s (1.97 fps), enhancer 0.70, mask mean=0.064
- BORDER_REFLECT (v5.35) eliminó halo oscuro

## Relevant Files
- `roop/ProcessMgr.py`: Pipeline principal — v5.57 enhancer_blend 0.80, unsharp 2.5, radial GFPGAN fade, erosión 5×5, //30, truncation 0.10, profile EMA responsive
- `ui/tabs/faceswap/ui.py`: slider enhancer_blend default 0.70
- `roop/face_util_rotation.py`: RetinaFace + MediaPipe fallback + rotaciones forzadas
- `roop/globals.py`: enhancer_blend_factor = 0.85
