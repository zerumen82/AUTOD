# Plan — pestaña de mejora de imágenes «estilo Topaz» (2026-11-15)

Estado: **F0 implementado y PASS** (v5.1365, 2026-11-16): importe, cola,
sliders por mejora, presets, comparador antes/después, autopilot, lote
secuencial con cancelar y gate de calidad por paso. F1–F4 pendientes.
Alcance: una pestaña nueva en AutoAuto que procesa **grupos de imágenes** con
correcciones de IA (denoise, nitidez, upscale, caras, luz/color, grano,
eliminación de objetos), con pila de mejoras, presets, modo automático y
exportación por lotes.

---

## 1. Qué es Topaz hoy (investigación del 2026-11-15, con fuentes)

Topaz Photo AI **ya no existe como tal**: la última versión fue **v4.0.4
(2025-09-10)** y está descontinuada sin más modelos. El producto actual es
**Topaz Photo v1.8.0 (2026-11-10)**, y el resto de la familia es
**Topaz Gigapixel**, **Topaz Video** y el bundle **Topaz Studio**
(<https://docs.topazlabs.com/topaz-photo/release-summary>,
<https://docs.topazlabs.com/photo-ai/release-summary>).

### 1.1 Las funciones de Topaz Photo que importan para nuestra pestaña

| Mejora | Modelos / controles de Topaz | Nota oficial |
|---|---|---|
| Denoise (RAW y no-RAW) | `Normal`, `Strong`, `Extreme`, `Denoise MAX`; `Strength`, `Minor Deblur`, `Original Detail` | Los generativos exigen **8 GB de VRAM** |
| Sharpen | `Standard`, `Strong`, `Lens Blur`, `Motion Blur`, `Natural`, `Refocus`, `Wildlife`, `Portrait`, `Noise-Aware`; `Strength`, `Minor Denoise` | `Minor Denoise` viene **apagado** por defecto |
| Adjust Lighting | `v1` (Strength 1–25), `v2`/`v3` (Exposure, Highlight, Shadow, color) | |
| Balance Color | `Temperature`, `Opacity` | |
| Recover Faces | `v2`, `v3` (generativo), `Strength`, pelo y cuello | `v3` procesa **cada cara por separado** |
| Upscale & Resize | `Standard`, `High Fidelity`, `Low Resolution`, `Graphics`, `Standard MAX`, `Recover`; `Scale` hasta **600 %**, `Creativity Control` | Tope: 32 000 px o 2 GB |
| Super Focus | `v1`, `v2`, `v3` (desenfoque) | `v3` necesita 8 GB de VRAM |
| Dust & Scratch / Spot Healing | `v2` / pincel | |
| Remove Object | `v2` (inpaint generativo) con pinceles y máscara | |
| Grain | `Gaussian`, `Grey`, `Silver Rich`; `Strength`, `Intensity`, `Size` | Añadido en v1.2.0 (2026-01-21) |
| Wonder | `v1`–`v4` (generativo, sin controles en v4) | `v4` **solo nube** |
| Preserve Text | `Low Resolution`, `Noisy/Compressed` | Exige máscara manual |

Fuente: <https://docs.topazlabs.com/topaz-photo/enhancements> y las páginas
de cada mejora.

**Corrección importante**: *Dehaze* **no** es una función de Topaz Photo (era
un ajuste del antiguo Topaz Studio 2) y *Remove Background* es un modelo **de
nube** (`RemoveBG`), no de la app de escritorio
(<https://topazlabs.com/dehaze/>,
<https://developer.topazlabs.com/image-models/removal-and-cleanup/background-removal>).

### 1.2 Cómo trabaja Topaz por lotes (lo que hay que imitar)

- Importar **varios archivos o carpetas enteras** arrastrando
  (<https://docs.topazlabs.com/topaz-photo/functions/batch-processing>).
- **Autopilot**: analiza cada foto (tipo de archivo, metadatos ISO/cámara,
  severidad del ruido, desenfoque del sujeto, caras, tamaño) y elige filtros y
  fuerza; se puede personalizar y se puede aplicar a todo el lote
  (<https://docs.topazlabs.com/topaz-photo/functions/autopilot-and-configuration>).
- **Presets** que guardan la pila de mejoras completa, no solo los deslizadores;
  algunos modelos no se pueden guardar en un preset (Super Focus, Dust &
  Scratch, Wonder, Standard MAX, Remove).
- Exportación con prefijo/sufijo, «añadir los filtros aplicados al nombre» (por
  defecto **sí**), destino (carpeta original o la elegida), formato (JPEG 95 %,
  PNG, TIFF, DNG, «conservar formato») y cola de exportación con el estado.
- Selección y máscaras: `All / Subject / Background / Portrait / Landscape` +
  pinceles (`Regular`, `Object Selection`, `Super-Pixel`), feathering 100.

### 1.3 La API de Topaz (referencia de parámetros)

Topaz publica una API con el modelo y sus parámetros exactos, con **deslizadores
normalizados 0.0–1.0** (`sharpenStrength`, `denoiseStrength`, `strength`,
`opacity`) y `lighting/highlight/shadow` multiplicativos 0.5–1.5
(<https://developer.topazlabs.com/>). Es la especificación más limpia para
copiar nombres y rangos, y sirve además como **nivel cloud opcional** (con
créditos, no es local).

### 1.4 Requisitos de la familia Topaz

Windows, 16 GB de RAM, **6 GB de VRAM** (8 GB recomendados), 60 GB de disco;
los modelos generativos exigen 8 GB. Sin Linux ni VM.
(<https://docs.topazlabs.com/topaz-photo/system-requirements>)

Nuestro equipo: **RTX 3060 Ti, 8 GB de VRAM, CUDA 12.4, 32 GB de RAM** →
cabe el perfil local, pero **no** la capa generativa (`Wonder`, `Super Focus v3`,
`Recover Faces v3`, `Denoise MAX`) en local.

---

## 2. Qué hay ya en AutoAuto (se reutiliza, no se reescribe)

| Pieza existente | Ubicación | Uso en la pestaña |
|---|---|---|
| Real-ESRGAN x2/x4, LSDIR x4 (ONNX) | `models/Frame/real_esrgan_x2.onnx`, `real_esrgan_x4.onnx`, `lsdir_x4.onnx` | **Upscale** (ya está en `roop/processors/Frame_Upscale.py`) |
| GFPGAN v1.4 (ONNX, Apache-2.0) | `models/GFPGANv1.4.onnx` | **Recover Faces** (motor `roop/processors/Enhance_GFPGAN.py`) |
| LaMa (ONNX, CPU) | `models/lama_fp32.onnx`, `roop/img_editor/lama_inpaint.py` | **Remove Object** (inpaint determinista, ya integrado) |
| ISNet (ONNX) | `models/Frame/isnet-general-use.onnx` | **Separación de sujeto/fondo** (máscara de selección) |
| DeOldify (ONNX) | `models/Frame/deoldify_*.onnx` | Colorizar fotos en blanco y negro |
| InsightFace (buffalo_l) | `models/buffalo_l` | Detección de caras para *Recover Faces* y para el análisis |
| CLIPSeg | `roop/img_editor/clipseg_masker.py` | Análisis de sujeto/fondo y «¿qué hay en la foto?» |
| Medidor de calidad (nitidez, posterización, banding, croma) | `roop/img_editor/image_quality_pipeline.py`, `roop/img_editor/top_tier_finish.py` | **Autopilot**: medir ruido/nitidez/banding antes de elegir modelo |
| Verificación medida | `roop/img_editor/measured_verification.py` (nuevo, v5.1365) | Comprobar que la mejora **no empeora** la foto (gate de nitidez/croma) |
| Lote + progreso + cancelación | `roop/core.py:batch_process_regular`, `ui/job_cancel.py` | **Cola de procesamiento** por lotes |
| ONNX Runtime CUDA | `roop/__init__.py`, `third_party/onnxruntime` | Ejecución local de los modelos ONNX |

Lo que **no** existe hoy: entrada de **múltiples archivos/carpetas** en el
Image Editor (solo una imagen), una **pila de mejoras**, **presets**, un
**autopilot** y una **exportación por lotes** con nomenclatura.

---

## 3. Correspondencia Topaz → local (propuesta)

Modelos con licencia permisiva y ONNX disponible (fuentes en el informe de
investigación: HF API y repos oficiales).

| Topaz | Equivalente local propuesto | Licencia | VRAM | Estado |
|---|---|---|---|---|
| Denoise (Normal/Strong/Extreme) | **SCUNet** ONNX (`Heliosoph/scunet-onnx`) | Apache-2.0 | ~2 GB | Nuevo (~70 MB) |
| Denoise suave (minor) | CLAHE + denoise no local de OpenCV + filtro bilateral | MIT | 0 | Nuevo (código) |
| Sharpen (Standard/Strong/Natural) | Unsharp adaptativo medido + **NAFNet** ONNX (desenfoque, `opencv/deblurring_nafnet`) | MIT | ~2 GB | Nuevo (~60 MB) |
| Recover Faces | **GFPGAN v1.4** (ya local) | Apache-2.0 | ~1–2 GB | **Ya existe** |
| Upscale (Standard / High Fidelity) | **Real-ESRGAN** x2/x4 (ya local) + **RealPLKSR** para el camino rápido (`darktable-org/upscale-realplksr-onnx`) | BSD-3 / MIT | ~2–4 GB | Ya existe + 1 nuevo |
| Low Resolution / Text | SwinIR real-SR ONNX (`rocca/swin-ir-onnx`) | Apache-2.0 | ~4 GB | Nuevo (opcional) |
| Super Focus (desenfoque) | Deconvolución de Wiener / **NAFNet** | MIT / público | 0–2 GB | Nuevo (código) |
| Dust & Scratch | Detección de motas (umbral local) + inpaint LaMa | Apache-2.0 | 0 | Nuevo (código) |
| Remove Object | **LaMa** (ya local) con máscara CLIPSeg/pintada | Apache-2.0 | 0 (CPU) | **Ya existe** |
| Adjust Lighting | OpenCV/NumPy: exposición, altas luces, sombras en Lab | MIT | 0 | Nuevo (código) |
| Balance Color | Balance de blancos por gray-world/PCA + temperatura | MIT | 0 | Nuevo (código) |
| Grain | Ruido gaussiano por separado de luminancia y color | MIT | 0 | Nuevo (código) |
| Remove Background | ISNet (ya local) + recorte alfa | ISNet: MIT | ~1 GB | **Ya existe** |
| Preserve Text | Máscara manual + Real-ESRGAN solo dentro de la máscara | BSD-3 | — | Nuevo |
| Wonder / Denoise MAX / Recover Faces v3 / Standard MAX / Super Focus v3 | **Solo nube** vía API de Topaz (opcional, con créditos) o **no** | Comercial | — | Decisión del usuario |

Nombres de licencia que **no** se deben incrustar: CodeFormer y
RestoreFormer++ (S-Lab, no comercial) — ya están en `models/` por el FaceSwap,
pero la pestaña nueva no los debe usar; GPEN (sin licencia clara); los pesos
de RMBG-1.4 (no comercial).

---

## 4. Diseño de la pestaña

### 4.1 Nombre y sitio

- Pestaña: **«✨ Mejorar fotos»** en `ui/main.py` (`create_ui()`, la segunda
  definición, la que manda), siguiendo el patrón de las demás:
  `ui/tabs/enhance/ui.py` (`build_enhance_ui()`), `ui/tabs/enhance/events.py`
  (`wire_events()`), `ui/tabs/enhance/logic.py`, `ui/tabs/enhance/state.py`, y
  el envoltorio `ui/tabs/enhance_tab.py`.
- No se toca ninguna pestaña existente: Generate, Image Editor, Face Swap,
  Animate y Settings siguen igual.

### 4.2 Pantalla (añadido 2026-11-15: sliders, presets de estilo, antes/después)

El usuario exigió tres cosas de Topaz: **sliders** por mejora, **presets de
estilo** de corrección y **ver antes y después**. Se añaden al diseño:

```
┌─ IMPORTACIÓN ─────────────────────────────────────────────────┐
│ [ Importar fotos ]  [ Importar carpeta ]                       │
│ Cola (tabla): archivo | estado | tamaño | notas                │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌─ VISTA ANTES / DESPUÉS ────────────────────────────────────────┐
│  [ Imagen original ]   ⇄   [ Imagen mejorada ]                  │
│  Slider de comparación ◐━━━━●━━━━ (arrastrar para comparar)    │
│  Zoom: [ 25% ▾ ]  [ 100% ]  [ Ajustar ]                        │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌─ PILA DE MEJORAS (cada fila = slider + modelo + fuerza) ──────┐
│ ☑ 1. Denoise       ▸ Normal/Strong/Extreme   [━━●━━━━ 65]     │
│ ☑ 2. Sharpen       ▸ Standard/Natural/Lens   [━━━●━━ 40]      │
│ ☑ 3. Recover Faces ▸ GFPGAN v1.4             [━━━━●━ 80]      │
│ ☑ 4. Upscale       ▸ Real-ESRGAN x4 / PLKSR  [ 2× ▾ ]         │
│ ☑ 5. Luz y color   ▸ exposición/altas/sombras [━━●━━ 50]      │
│ ☐ 6. Grano         ▸ Gaussian/Grey/Silver    [━━━●━━ 30]      │
│ ☐ 7. Quitar objeto ▸ máscara CLIPSeg/pintada [  pincel  ]     │
│   [+ Añadir mejora]  [↑ ↓ reordenar]  [× quitar]              │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌─ PRESETS DE ESTILO ────────────────────────────────────────────┐
│ [ Retrato ] [ Paisaje ] [ Nocturna ] [ Producto ] [ Archivo ]  │
│ [ Guardar preset actual ]  [ Cargar ]  [ Eliminar ]            │
│ Cada preset = valores de todos los sliders + modelos + orden   │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌─ MODO Y EXPORTACIÓN ───────────────────────────────────────────┐
│ ☐ Modo automático (Autopilot mide y propone)  [ Aplicar a cola]│
│ Destino: [carpeta]  prefijo/sufijo  formato  [ Procesar lote ] │
└────────────────────────────────────────────────────────────────┘
```

**Sliders**: cada mejora tiene su slider de fuerza 0–100 (o 0.0–1.0
normalizado, como la API de Topaz) y un dropdown de modelo. Cambiar un
slider **vuelve a renderizar la vista antes/después al instante** (con un
debounce de 300 ms) sin tocar la cola.

**Presets de estilo**: un preset es un JSON que guarda la pila completa
(modelo + fuerza + orden + activada/desactivada de cada mejora). Viene con
presets de fábrica: *Retrato* (denoise suave + sharpen + recover faces),
*Paisaje* (sharpen fuerte + balance color + upscale), *Nocturna* (denoise
fuerte + luz + grano), *Producto* (sharpen + texto preservado), *Archivo*
(denoise + escala de grises + grano). El usuario puede guardar/cargar/borrar
los suyos en `config/enhance/presets/`.

**Antes/después**: vista dividida con slider de comparación (como Topaz),
zoom sincronizado, y métricas medidas (ruido σ, nitidez Laplaciana, tamaño,
caras detectadas) escritas debajo de cada vista.

### 4.3 Reglas de diseño (heredadas del proyecto)

- **Nada de listas de palabras**: la pestaña no decide por vocabulario; usa
  medición (ruido, nitidez, tamaño, caras detectadas) para el modo automático.
- **Local-first**: todo el procesamiento ocurre en la máquina; la nube es un
  extra **opt-in y etiquetado como tal**.
- **Sin regresiones**: cambiar esta pestaña no puede tocar Image Editor,
  Generate, FaceSwap ni Animate.
- **Todo parámetro en `config/`**: `config/enhance.json` con los ajustes por
  defecto, los umbrales del autopilot y los nombres/rutas de los modelos.
- **Gate de calidad**: cada paso se mide antes/después (nitidez, posterización,
  croma, clipping) y **se revierte si empeora** — el mismo criterio que
  `top_tier_finish._StepGate`. Ninguna mejora puede entregar una foto peor.
- **Sin falsos positivos**: si un modelo no está descargado, la fila se marca
  «no disponible» y **no** se simula el resultado.

### 4.4 Autopilot (modo automático)

Mediciones por foto (todas locales y medibles):

| Señal | Cómo se mide | Decide |
|---|---|---|
| Ruido | varianza en alta frecuencia por tile (σ de laplaciano) | Denoise `Normal`/`Strong` y su fuerza |
| Nitidez | varianza de Laplacian normalizada por contraste | Sharpen `Standard`/`Natural` |
| Tamaño | lado mayor de la imagen | Upscale y su factor ( hasta 12 MP o 6×, como Topaz) |
| Caras | InsightFace: nº y tamaño | Recover Faces (y por cara, como el `v3` de Topaz) |
| Banding/posterización | histograma de tonos + banding score | Luz/color y su fuerza |
| Sujeto/fondo | ISNet | selección por defecto (`Subject` / `Background`) |

El resultado del análisis se puede **ver y corregir** antes de aplicar (como el
Autopilot de Topaz, que deja cambiar los ajustes a mano).

### 4.5 Lote y exportación

- Entrada: `gr.Files(file_count="multiple")` (patrón de Face Swap,
  `ui/tabs/faceswap/ui.py:156`) **y** selector de carpeta.
- Procesamiento en hilo con `gr.Progress()` y cancelación por `ui/job_cancel.py`
  (patrón de Generate, `ui/tabs/gen/events.py`).
- Salida: `roop/output_paths.py` (nueva carpeta `enhance/`), prefijo/sufijo,
  «añadir las mejoras aplicadas al nombre» (por defecto sí), formato
  (conservar/JPEG 95 %/PNG/TIFF), política de sobrescritura.
- Presets: JSON en `config/presets/` (compartibles) + copia por usuario.

### 4.6 Motor y memoria (8 GB)

- ONNX Runtime **CUDA** con *tiles* (la app ya usa tiles en
  `Frame_Upscale.py`) y **fallback a CPU** si no hay VRAM.
- La GPU está compartida con ComfyUI: el procesamiento de la pestaña debe
  **liberar** (`/free`) y **no** competir a la vez con un render en curso; si
  ComfyUI está ocupado, la cola espera y lo dice.
- Techo de salida: 32 000 px por lado (como Topaz) y tope de píxeles en
  `config/enhance.json`.

---

## 5. Fases (cada una con evidencia obligatoria)

| Fase | Entregable | Criterio de cierre |
|---|---|---|
| **F0 — Cimientos** | `ui/tabs/enhance/` funcional: importar N imágenes, cola, barra de progreso, cancelar, **vista antes/después con slider**, **sliders por mejora**, **presets de estilo** | Lote de 10 fotos procesadas de verdad; slider antes/después funciona; preset se aplica; salida correcta; sin tocar otras pestañas |
| **F1 — Upscale + caras** | Upscale (Real-ESRGAN/PLKSR) y Recover Faces (GFPGAN), con tiles y gate de calidad | antes/después medidos (nitidez, tamaño) en 3 fotos reales, sin regresión de nitidez |
| **F2 — Denoise + nitidez** | SCUNet + NAFNet (o equivalente) con fuerza por paso | Ruido medido antes/después (σ laplaciano) y nitidez no peor en fotos con ruido |
| **F3 — Luz, color, grano, quitado** | Ajustes clásicos + grano + LaMa con máscara | Medición de L*a*b* antes/después; quitado de objeto con inpaint medido |
| **F4 — Autopilot + presets + export** | Análisis automático, presets, nomenclatura y formatos | Autopilot sobre 20 fotos: decisiones explicadas y verificables; reejecutar el mismo preset da el mismo resultado |

Cada fase sube versión, se anota en `MEMORY.md` y necesita PASS del tester y
APPROVE del reviewer. **Ninguna fase se declara hecha con clips cortos.**

---

## 6. Riesgos y límites (declarados por adelantado)

1. **Sin equivalente local de la capa generativa** (`Wonder`, `Denoise MAX`,
   `Super Focus v3`, `Recover Faces v3`, `Standard MAX`): o se usa la **API de
   Topaz en nube** (créditos, datos fuera de la máquina) o no existe. Our
   AGENTS exige local-first; la nube quedaría **opt-in y etiquetada**.
2. **VRAM de 8 GB**: los modelos generativos locales no entran; el diseño
   apunta a ONNX lightweight con tiles.
3. **Licencias**: CodeFormer, RestoreFormer++, GPEN y RMBG-1.4 no se usan en la
   pestaña aunque ya estén en `models/`.
4. **Calidad de la imitación**: Topaz lleva años afinando sus modelos; el
   objetivo es **paridad funcional** (el conjunto de operaciones, el flujo de
   lote, la automátización y la exportación), no equivalencia píxel a píxel.
   Eso hay que decirlo sin adornos.
5. **Coste de entrada en disco**: los modelos nuevos son pequeños (SCUNet ~70
   MB, NAFNet ~60 MB, PLKSR ~5 MB), pero su descarga requiere **autorización**
   del usuario (regla de prohibiciones).

---

## 7. Decisiones que necesita el usuario

1. **Nube de Topaz**: ¿permitimos la API de Topaz (pago, datos fuera del
   equipo) para la capa generativa, o la pestaña es **100 % local** y esa capa
   no existe? → *Recomendación: 100 % local; si hace falta algo generativo,
   se usa el motor local de AutoAuto (LongCat/Kontext), que ya está instalado.*
2. **Modelos nuevos**: ¿autorizamos descargar SCUNet + NAFNet + RealPLKSR
   (~135 MB, licencia permisiva) para denoise/desenfoque/upscale rápido?
   → *Recomendación: sí, son la base de denoise y sharpen y no compiten con
   la licencia de los modelos ya instalados.*
3. **Nivel de la primera versión**: ¿F0+F1 (lote + upscale + caras) o el
   conjunto completo? → *Recomendación: F0+F1 primero, porque son las piezas
   que ya existen y dan valor inmediato.*
4. **Salida**: ¿carpeta nueva `enhance/` en el proyecto (como `img_editor/` y
   `generate/`) o la carpeta que elija el usuario en cada lote?
   → *Recomendación: las dos, con la del usuario por defecto en el diálogo.*
5. **Nombre de la pestaña**: «✨ Mejorar fotos» / «✨ Foto IA» / «🔧 Topaz local»
   → *Recomendación: «✨ Mejorar fotos» (Topaz es marca de Topaz Labs).*

---

## 8. Contratos SDD (propuestos, en `spec/02-REQUISITOS.md`)

- **TOP-1**: importar varios archivos o una carpeta y ver una cola con el
  estado de cada foto.
- **TOP-2**: pila de mejoras ordenada, cada mejora con modelo y fuerza, y
  activable/desactivable sin rehacer la foto.
- **TOP-3**: gate de calidad por paso: si una mejora empeora la foto
  (nitidez, posterización, croma, clipping), se revierte y se avisa.
- **TOP-4**: modo automático que elige mejoras y fuerzas **medidas** por foto.
- **TOP-5**: presets que guardan y reproducen una pila completa.
- **TOP-6**: exportación por lotes con prefijo/sufijo, formato y destino.
- **TOP-7**: todo en local; la nube, si la hubiera, es opt-in y etiquetada.
- **TOP-8**: ningún modelo ausente se simula; la fila se marca y el lote sigue.
