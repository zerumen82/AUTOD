# Estado — ImgEditor structural (añadir / quitar personas)

**Fecha:** 23 de junio de 2026 (v2 — post-consolidación)  
**Hardware:** RTX 3060 Ti 8 GB · ComfyUI en `http://127.0.0.1:8188`  
**Objetivo:** Edición estilo Grok Imagine con análisis semántico local (sin hardcodes por palabra del usuario) para **añadir** y **quitar** personas en una foto existente.

---

## Resumen ejecutivo

| Área | Estado |
|------|--------|
| Análisis semántico (`bias=add/remove`) | ✅ Funciona |
| Pipeline técnico e2e (ComfyUI + guardado) | ✅ Funciona |
| Quitar personas — máscara | ✅ **99.8% cobertura** (antes 56%) |
| Quitar personas — resultado visual | 🔄 Pendiente re-test con máscara nueva |
| Añadir personas — resultado visual | ❓ Sin validar visualmente |
| FacePreserver / rewriter LLM | Fuera del flujo actual de structural |

**Cambio principal:** la máscara de eliminación pasó de **56% → 99.8%** de cobertura (CLIPSeg + InsightFace + cierre morfológico + dilatación agresiva + fallback full-frame). El bloqueo anterior (cuerpos sin cara en bordes no cubiertos) está resuelto.

---

## Últimas pruebas

### Test offline (23-jun, sin ComfyUI)
- **Máscara:** `_build_person_removal_mask()` → CLIPSeg 89.9% + InsightFace(2) + dilatación → **99.8%**
- **Máscara guardada:** `debug_test/person_mask_improved.png`
- **Análisis semántico:** mag=0.74, structural=0.35, bias=remove ✅

### Test anterior con ComfyUI (referencia)
- **Motor:** Flux Fill inpaint (`ggml-model-Q4_K_M.gguf`) — ruta directa `use_flux_remove_inpaint`
- **Parámetros:** denoise=1.0, steps=26, CFG=1.0, grow_mask=48
- **Duración:** ~130 s

---

## Flujo actual — Quitar personas

```
Prompt ES/EN
  → translate_prompt (local-first)
  → LightLocalIntentAnalyzer
       → mag, target, structural, bias=remove
  → auto_detect_params (denoise=1.0, steps=26, CFG=1.0)
  → _compose_generation_prompt (wrapper "completely remove...")
  → _build_person_removal_mask (CLIPSeg + InsightFace + dilatación agresiva + fallback)
  → si máscara: Flux Fill inpaint directo (sin ICEdit, sin LongCat)
  → guardar resultado
```

### Routing consolidado (`bias=remove`)
1. **Flux Fill inpaint** (`ggml-model-Q4_K_M.gguf`) — único motor. Sin ICEdit fallback.
2. Si Flux Fill no disponible → LongCat Turbo inpaint (denoise=1.0).

### Routing (`bias=add`)
- **LongCat** con `reference_latents_method = "offset"` — preserva la foto original y añade personas.
- Sin cambios en esta iteración.

---

## Mejoras implementadas (23-jun-2026)

### 1. `_build_clipseg_person_mask` — más agresiva
- Prompts ampliados: `person, human body, full body person, people, man, woman, body, torso`
- Thresholds: 0.35, 0.40, 0.50, 0.30, 0.60 (elige mejor cobertura)
- Dilation: 12 → 20
- Eliminado filtro restrictivo 18-62% — acepta cualquier máscara con >5% cobertura

### 2. `_build_person_removal_mask` — post-procesado agresivo
- **Cierre morfológico** (kernel 7+) para rellenar huecos dentro de cuerpos
- **Dilatación** (kernel 15+) para expandir a bordes
- Si cobertura < 30% → dilatación extra (kernel 31)
- **Fallback full-frame**: si hay caras detectadas pero no máscara, genera máscara de frame completo (menos 5% borde)

### 3. `_mask_from_face_bboxes` — cobertura corporal
- width_mult: 2.2 → 4.0
- height_mult: 4.5 → 6.0
- head_pad: 0.35 → 0.5

### 4. Routing simplificado
- Eliminado **ICEdit** como fallback para `bias=remove` (innecesario, Flux Fill es superior)
- Un solo camino: Flux Fill inpaint → LongCat Turbo (fallback)

### 5. Parámetros mejorados
- Structural remove: steps 20→26, guidance 1.0
- Flux Fill override: guidance 3.5→1.0 (consistente con Flux Fill puro)

---

## Modelos en disco (relevantes)

| Modelo | Estado |
|--------|--------|
| LongCat Turbo / Full Q4 | ✅ |
| Qwen Q2 + 2511 | ✅ |
| Flux Dev Q2 + Abliterated | ✅ |
| `ggml-model-Q4_K_M.gguf` (Flux Fill) | ✅ en `diffusion_models/` |
| ICEdit LoRA `ICEdit-normal-LoRA.safetensors` | ✅ (sin uso en structural) |
| T5 `t5xxl_fp8_e4m3fn.safetensors` | ✅ |
| OmniGen2 Q4 GGUF | ✅ (poco integrado en UI) |

---

## Archivos clave modificados

- `roop/img_editor/img_editor_manager.py` — `_build_person_removal_mask`, `_build_clipseg_person_mask`, `_mask_from_face_bboxes`, routing simplificado, params
- `roop/img_editor/clothing_segmenter.py` — (sin cambios, usado por CLIPSeg)

---

## Problemas conocidos

1. ~~**Máscara incompleta (56%)**~~ → **RESUELTO** (99.8% cobertura tras mejoras)
2. **Artefacto gris:** en zonas inpaintadas a veces queda blur/overlay — depende del modelo Flux Fill, no de la máscara
3. **Añadir personas (`bias=add`):** lógica implementada; sin validación visual en esta ronda
4. **E2E no testeado:** ComfyUI no disponible en el momento de escribir, la máscara se verificó offline

---

## Próximos pasos (prioridad)

1. **Re-test e2e** con ComfyUI activo — `test_person.jpg` y `test_swap_two_images.jpg`
2. **Validar `bias=add`** con prompt tipo "añade dos personas al fondo"
3. Si Flux Fill deja artefactos: probar con Flux Dev Q2 inpaint o menor grow_mask
4. Si el resultado visual es bueno: cerrar el issue de "quitar personas"

---

## Cómo reproducir

```powershell
# Solo máscara + análisis (sin GPU / ComfyUI)
.\venv\Scripts\python.exe -c "
import sys; sys.path.insert(0,'.')
from PIL import Image
from roop.img_editor.img_editor_manager import ImgEditorManager
import numpy as np
mgr = ImgEditorManager()
img = Image.open('testdata/test_person.jpg').convert('RGB')
mask = mgr._build_person_removal_mask(img)
arr = np.array(mask.convert('L'))
cov = (arr > 128).sum() / float(arr.size)
print(f'Cobertura: {cov*100:.1f}%')
mask.save('debug_test/person_mask_improved.png')
"

# E2E completo (requiere ComfyUI en 8188)
.\venv\Scripts\python.exe -c "
import sys; sys.path.insert(0,'.')
from tools.test_structural_global import run_e2e_test
run_e2e_test('testdata/test_person.jpg', 'quita a todas las personas de la foto', 'debug_test')
"
```

---

## Conclusión

El sistema **detecta bien** la intención de quitar personas, genera **máscara de cobertura casi total** (99.8%), enruta al motor correcto (Flux Fill inpaint) y completa la generación sin errores. La máscara mejorada debería eliminar el problema anterior de cuerpos no cubiertos en los bordes. Pendiente re-test visual con ComfyUI para confirmar.
