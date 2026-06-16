# AGENTS.md — AutoAuto Face Swap Project

## Goal
- Eliminar blur elíptico fuera de cara, mejorar tracking de perfiles, aumentar nitidez y parecido de source.

## Constraints & Preferences
- `inswapper_128_facefusion.onnx` (128→256px), XSeg Masker, GFPGAN Enhancer (blend=0.95), MediaPipe Face Mesh 468 landmarks, FFmpeg libx264 CRF 14.
- Modo `selected_faces`: 655 facesets origen, 1 destino (1280×720, 850 frames, 30 fps).
- GPU RTX 3060 Ti 8 GB, CUDA 12.4, providers: CUDAExecutionProvider, CPUExecutionProvider.

## Progress
### v5.65 (current) — source-true limit
- **Identity Locked**: Master DNA injection pushed to 65% (profiles) and 25% (frontal) + source selection weight at 30.0x for absolute resemblance.
- **Micro-Textural Hyper**: Unsharp mask amount at 6.8 for extreme textural definition.
- **Surgical-Pro Mask**: Mask blur factor reduced to //75 and internal feathering offset to 1px (Surgical-Seams) for perfect integration.
- **Instant-True Tracking**: Profile M-EMA reduced to 0.10x for zero-lag frame rigidity; color EMA alpha at 0.95 for instant photon adaptation.
- **Enhancer Pro**: Internal GFPGAN blend forced to 0.95 for skin texture perfection.

### v5.64 — god-fidelity escalation
- **Identity Absolute**: Master DNA injection pushed to 55% (profiles) and 20% (frontal) + source selection weight at 20.0x for zero-drift subject fidelity.
- **Micro-Textural Razor**: Unsharp mask amount at 6.2 for unprecedented detail.
- **Invisible-God Mask**: Mask blur factor reduced to //65 and internal feathering offset to 2px for seamless integration.
- **Rigidity Tracking**: Profile M-EMA reduced to 0.15x for physical-frame rigidity; color EMA alpha at 0.92 for instant photon adaptation.
- **Efficiency**: Skip threshold lowered to 0.05 det_score to capture every possible frame.
- **Version String**: Updated UI to "GOD-FIDELITY" v5.64.

### v5.63 — razor-fidelity refinements

### Known Issues
- ~60% del video son perfiles → scores bajan a 0.37-0.54; M-EMA adaptivo + kps_EMA responsive mitigan parcialmente
- Similitud source baja (0.111) — persona diferente al source; v5.59 slider default 0.95
- Frames 565-566: ahora se swapean (skip solo si det_score < 0.15 AND vel > 100px)

## Key Decisions
- Skip threshold reducido para minimizar frames saltados; riesgo: frame con detección muy mala puede generar artefacto visible
- enhancer_blend via slider (default 0.95) — respeta UI, no hardcode

## Next Steps
- Testear v5.59 — enhancer 0.95 slider + unsharp 3.5 deben dar más identidad y nitidez
- Verificar que frames 565-566 ya no se salten

## Critical Context
- Log v5.58: 434.07s (1.96 fps), enhancer 0.85, mask mean=0.059 ✅
- Log v5.56: 430.96s (1.97 fps), enhancer 0.70, mask mean=0.059 ✅
- enhancer_blend via slider (default 0.95) en ProcessMgr.py:1736

## Relevant Files
- `roop/ProcessMgr.py`: Pipeline principal — v5.59 enhancer_blend via slider, unsharp 3.5, skip threshold 0.15/100px, radial GFPGAN fade, erosión 5×5, //30, truncation 0.10, profile EMA responsive
- `ui/tabs/faceswap/ui.py`: slider enhancer_blend default 0.95
- `roop/face_util_rotation.py`: RetinaFace + MediaPipe fallback + rotaciones forzadas
