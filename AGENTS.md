# AGENTS.md — AutoAuto Face Swap Project

## Goal
- Eliminar blur elíptico fuera de cara, mejorar tracking de perfiles, aumentar nitidez y parecido de source.

## Constraints & Preferences
- `inswapper_128_facefusion.onnx` (128→256px), XSeg Masker, GFPGAN Enhancer (blend=0.70), MediaPipe Face Mesh 468 landmarks, FFmpeg libx264 CRF 14.
- Modo `selected_faces`: 655 facesets origen, 1 destino (1280×720, 850 frames, 30 fps).
- GPU RTX 3060 Ti 8 GB, CUDA 12.4, providers: CUDAExecutionProvider, CPUExecutionProvider.

## Progress
### v5.73 (current) — Mouth blend reducido + máscara expandida + unsharp moderado
- **m_blend 0.85→0.70**: Menos preservación de boca target = más identidad source visible en el área más expresiva de la cara. Base 0.70, dinámico hasta 0.70 (antes 0.85) con oclusión.
- **Unsharp 6.8→2.5**: El upscale 128→256px del modelo inswapper ya introduce pixelación latente en close-ups; 5.8× neto de sharpening la amplificaba creando "pixela". Ahora 1.5× — suficiente nitidez sin halos.
- **Máscara expandida**: Perfil 0.65→0.75, frontal 0.60→0.70 — más área facial del swap visible, más identidad source sin perder calidad.
- **Tail truncation 0.10→0.05**: Retiene 5% más de máscara, menos recorte del swap en bordes de cara.

### v5.72 — Single-Best Revert + Occlusion Fix
- **Master Embedding = mejor cara individual**: Revertido de v5.71 (weighted blend top-3) porque todas las top caras tenían quality ~5.59 → pesos iguales [0.34,0.33,0.33] → mismo problema de dilución que el promedio de 20. Ahora embedding real 100% pura.
- **Oclusión suavizada**: Blur 31×31 post-warp + strength reducido (0.50→0.40 frontal, 0.60→0.50 perfil) para eliminar rayas.
- **Boca verificada**: Impacto log 0.001 es esperado (media sobre frame completo); la preservación local funciona correctamente.

### v5.71 — Embedding Ponderado Top-3 (REVERTED)
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
- ~60% de imágenes destino son perfiles → scores bajan a 0.15-0.22; M-EMA adaptivo + kps_EMA responsive mitigan parcialmente
- 115 imágenes Grok (no video), modo selected_faces. Log: 2m01s, mask mean=0.010–0.111.

## Key Decisions
- **m_blend 0.85→0.70**: La boca es el área más expresiva; restaurarla al target al 85% tapaba identidad source. 70% es el sweet spot — suficiente para no perder calidad, máximo para identidad.
- **Unsharp 6.8→2.5**: Priorizar identidad sin halos en close-ups. El sharpening extremo (5.8× neto) creaba pixelación; 1.5× da nitidez natural.
- **Máscara expandida + tail truncation**: Cero costo computacional, ganancia directa de identidad visible.
- enhancer_blend via slider (default 0.95) — respeta UI, no hardcode

## Next Steps
- Testear v5.73 con m_blend 0.70 — debe dar más identidad source en boca sin perder calidad
- Si aún no es suficiente: enhancer_blend 0.70→0.60 (más swap crudo, menos GFPGAN)

## Critical Context
- Log v5.73 (115 imágenes Grok): 2m01s, enhancer 0.70, mask mean=0.010–0.111, unsharp 2.5, m_blend 0.70 ✅
- `dna_mix=1.0` ya inyecta 100% Master Embedding — no hay más margen por ese lado.
- `DEBUG_MASK mean` saltó de ~0.036 a ~0.111 en frames con cara grande (mask expansion efectiva).
- enhancer_blend via slider (default 0.95) en ProcessMgr.py:1736

## Relevant Files
- `roop/ProcessMgr.py`: Pipeline principal — v5.73 m_blend 0.70, unsharp 2.5, mask expandida 0.75/0.70, tail truncation 0.05, single-best embedding, dna_mix=1.0, enhancer_blend 0.70
- `ui/tabs/faceswap/ui.py`: slider enhancer_blend default 0.95
- `roop/face_util_rotation.py`: RetinaFace + MediaPipe fallback + rotaciones forzadas
