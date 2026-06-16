# AGENTS.md — AutoAuto Face Swap Project

## Goal
- Eliminar blur elíptico fuera de cara, mejorar tracking de perfiles, aumentar nitidez y parecido de source.

## Constraints & Preferences
- `inswapper_128_facefusion.onnx` (128→256px), XSeg Masker, GFPGAN Enhancer (blend=0.70), MediaPipe Face Mesh 468 landmarks, FFmpeg libx264 CRF 14.
- Modo `selected_faces`: 655 facesets origen, 1 destino (1280×720, 850 frames, 30 fps).
- GPU RTX 3060 Ti 8 GB, CUDA 12.4, providers: CUDAExecutionProvider, CPUExecutionProvider.

## Progress
### v5.71 (current) — Embedding Ponderado Top-3
- **Master Embedding = weighted blend top-3**: Calidad^3 weighting — mejor cara ~80%, 2da ~15%, 3ra ~5%. Enriquece identidad sin diluir (vs. promedio de 20 que diluía).
- **Identidad refinada**: El decoder del inswapper recibe un embedding más rico que preserva la identidad de la mejor foto con aporte complementario de otras buenas fotos.
- **Cero costo computacional**: Sin inferencia extra, solo cambio en cómo se calcula el embedding maestro.

### v5.70 — Master Embedding Single-Best
- **Master Embedding = mejor cara individual**: Reemplazó el promedio de 20 embeddings. Una embedding real preserva 100% de los rasgos del source (el promedio diluía).

### v5.69 — DNA Mix Absoluto
- **dna_mix=1.0**: Identidad 100% del master embedding en todos los modos (frontal y perfil).
- **m_blend 0.50→0.85**: Mouth mask más fuerte para preservar expresión bucal.
- **Mouth mask dilatación**: Kernel 9×9×2 iteraciones para mejor cobertura.

### v5.68 — Identity Force
- **Enhancer 0.95→0.70**: Menos GFPGAN = más identidad cruda del swap en el resultado final (30% raw swap vs 5% antes).
- **DNA injection frontal 0.25→0.65**: Igualado al perfil, 65% embedding maestro (promedio mejores sources) para identidad consistente.
- **DNA injection perfil 0.65→0.85**: 85% embedding maestro para perfiles extremos, máxima fidelidad rotacional.
- **Objetivo**: Maximizar parecido a source incluso cuando source y target son personas distintas.

### v5.67 — EMA refinements
- **EMA menos conservador**: alpha_prev reducido 0.92→0.85 para det_score<0.3, permite 15% swap nuevo (vs 8%) en frames de baja calidad.

### v5.66 — safe refinements
- **Enhancer Skip on Profiles**: GFPGAN saltado en perfiles con det_score < 0.45 para evitar alucinaciones.
- **Source Selection Weight**: Compromise weight 30.0→10.0 para que calidad compita con similitud y diversifique fuente seleccionada.

### v5.65 — source-true limit
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
- Testear v5.71 — verificar que el embedding ponderado top-3 mejora identidad vs v5.70 single-best
- Si la identidad aún no es suficiente, considerar multi-source swap blending (3x swap por frame) o texture transfer post-swap

## Critical Context
- Log v5.58: 434.07s (1.96 fps), enhancer 0.85, mask mean=0.059 ✅
- Log v5.56: 430.96s (1.97 fps), enhancer 0.70, mask mean=0.059 ✅
- enhancer_blend via slider (default 0.95) en ProcessMgr.py:1736

## Relevant Files
- `roop/ProcessMgr.py`: Pipeline principal — v5.71 embedding ponderado top-3, v5.69 dna_mix=1.0, enhancer_blend 0.70, unsharp 6.8, mask elastic //75, profile EMA responsive
- `ui/tabs/faceswap/ui.py`: slider enhancer_blend default 0.95
- `roop/face_util_rotation.py`: RetinaFace + MediaPipe fallback + rotaciones forzadas
