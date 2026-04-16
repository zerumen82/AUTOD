# Especificación Técnica: ImgEditor Tab

## Documento de Requisitos y Diseño

---

## 1. Resumen del Proyecto

### 1.1 Objetivo
Crear una nueva pestaña llamada **"ImgEditor"** en la UI principal de AUTO-DEEP que permita editar imágenes mediante prompts de texto usando inteligencia artificial.

### 1.2 Ubicación en la UI
```
AUTO-DEEP v2 - Pestañas
├── [👤] Faceswap
├── [📹] LiveCam  
├── [🎨] ImgEditor ← NUEVA (al lado de SD)
├── [🤖] Animate Photo
├── [⚙️] Config
│   ├── Face Manager
│   ├── Extras
│   └── Settings
└── [SD] Stable Diffusion
```

### 1.3 Hardware Disponible
- **VRAM:** 8GB GPU
- **Optimización:** Tiled VAE + fp16
- **Modelo:** FLUX Fill Pipeline (fp16)

---

## 2. Funcionalidades

### 2.1 Funcionalidades Obligatorias

| # | Funcionalidad | Descripción | Prioridad |
|---|---------------|-------------|-----------|
| F1 | Subir imagen | Drag & drop o selector de archivos | Alta |
| F2 | Input de prompt | Área de texto para describir cambios | Alta |
| F3 | Generación | Botón para ejecutar FLUX Fill | Alta |
| F4 | Gallery | Mostrar resultados múltiples | Alta |
| F5 | Descarga | Guardar imagen resultante | Alta |
| F6 | Sin censura | Sin filtros NSFW ni contenido | Alta |

### 2.2 Funcionalidades Opcionales

| # | Funcionalidad | Descripción | Prioridad |
|---|---------------|-------------|-----------|
| O1 | Editor de máscara | Dibujar área a modificar | Media |
| O2 | Historial de prompts | Ver prompts anteriores | Media |
| O3 | Face swap | Preservar/sustituir rostros | Media |
| O4 | Comparación | Before/After side-by-side | Media |
| O5 | Iteración | Añadir más cambios sobre resultado | Baja |

---

## 3. Arquitectura Técnica

### 3.1 Stack Tecnológico

```
┌─────────────────────────────────────────────────────────────┐
│                      CAPA DE UI                             │
│  Gradio + pywebview (ya existente)                          │
│  - ImageUploader (drag & drop)                              │
│  - Textbox (prompt)                                         │
│  - Button (generar)                                         │
│  - Gallery (resultados)                                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    CAPA DE LÓGICA                           │
│  img_editor_manager.py                                       │
│  - PromptAnalyzer (NLP básico)                              │
│  - ImageProcessor (pre/post procesamiento)                  │
│  - FacePreserver (preservación facial)                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  CAPA DE MODELO                             │
│  FLUX Fill Pipeline (diffusers 0.35.2)                      │
│  - FluxFillPipeline.load_in_fp16                           │
│  - Tiled VAE para optimización                              │
│  - Control de VRAM con torch.cuda.memory_allocated          │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Archivos a Crear/Modificar

#### Nuevos Archivos:

```
roop/img_editor/
├── __init__.py
├── flux_client.py           ← Cliente FLUX Fill Pipeline
├── prompt_analyzer.py       ← Análisis de prompts
├── face_preserver.py        ← Preservación de rostros
└── img_editor_manager.py    ← Orchestrator principal

ui/tabs/
└── img_editor_tab.py        ← UI del tab

output/
└── img_editor/              ← Guardado de imágenes
```

#### Archivos a Modificar:

```
ui/main.py
  - Añadir import de img_editor_tab
  - Añadir llamada a img_editor_tab() después de SD_tab()

roop/core.py (opcional)
  - Añadir funciones helper si son necesarias
```

---

## 4. API del FLUX Fill Pipeline

### 4.1 Carga del Pipeline

```python
# roop/img_editor/flux_client.py

from diffusers import FluxFillPipeline
import torch

class FluxClient:
    def __init__(self):
        self.pipe = None
        self.device = "cuda"
        
    def load(self):
        """Carga FLUX Fill con optimizaciones para 8GB VRAM"""
        self.pipe = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-fill-dev",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        
        # Optimizaciones para VRAM
        self.pipe.enable_vae_tiling()
        self.pipe.enable_sequential_cpu_offload()  # Si es necesario
        self.pipe.to(self.device)
```

### 4.2 Generación Inpainting

```python
def inpaint(
    self,
    image: PIL.Image,
    mask: PIL.Image,
    prompt: str,
    negative_prompt: str = "",
    num_inference_steps: int = 8,
    guidance_scale: float = 7.5,
    strength: float = 1.0,
) -> PIL.Image:
    """
    Args:
        image: Imagen base (PIL.Image)
        mask: Máscara (blanco = área a modificar)
        prompt: Descripción de cambios deseados
        negative_prompt: Lo que NO queremos
        num_inference_steps: Calidad (4-20, default 8)
        guidance_scale: Adherencia al prompt (1-20, default 7.5)
        strength: Intensidad del cambio (0-1, default 1)
    
    Returns:
        Imagen generada (PIL.Image)
    """
    result = self.pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength,
    ).images[0]
    
    return result
```

### 4.3 Generación Outpainting (Fill sin máscara)

```python
def outpaint(
    self,
    image: PIL.Image,
    prompt: str,
    negative_prompt: str = "",
    num_inference_steps: int = 8,
    guidance_scale: float = 7.5,
) -> PIL.Image:
    """
    FLUX Fill puede expandir la imagen automáticamente
    sin necesidad de máscara manual.
    """
    # No pasamos mask_image para outpainting automático
    result = self.pipe(
        prompt=prompt,
        image=image,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]
    
    return result
```

---

## 5. Análisis de Prompts

### 5.1 Clasificación Automática

```python
# roop/img_editor/prompt_analyzer.py

from enum import Enum
from typing import Tuple

class EditingMode(Enum):
    INPAINT = "inpaint"      # Modificar área específica
    OUTPAINT = "outpaint"    # Añadir contenido nuevo
    IMG2IMG = "img2img"      # Modificación global

class PromptAnalyzer:
    OUTPAINT_KEYWORDS = [
        "añade", "add", "más", "more", "al lado", "al fondo",
        "a la izquierda", "a la derecha", "detrás", "frente",
        "alrededor", "around", "expande", "expand", "amplía",
        "pon", "put", "coloca", "place", "añade", "add"
    ]
    
    INPAINT_KEYWORDS = [
        "cambia", "change", "modifica", "modify", "sustituye",
        "replace", "convierte", "convert", "make", "ponlo",
        "cámbiale", "change", "edit"
    ]
    
    def analyze(self, prompt: str) -> Tuple[EditingMode, str]:
        """
        Analiza el prompt y devuelve el modo de edición.
        
        Args:
            prompt: "quiero que este de rodillas y añade dos hombres alrededor"
        
        Returns:
            (EditingMode.OUTPAINT, "hombres alrededor")
        """
        prompt_lower = prompt.lower()
        
        # Detectar outpainting (añadir contenido)
        outpaint_detected = any(
            kw in prompt_lower for kw in self.OUTPAINT_KEYWORDS
        )
        
        # Detectar inpainting (modificar área)
        inpaint_detected = any(
            kw in prompt_lower for kw in self.INPAINT_KEYWORDS
        )
        
        if outpaint_detected:
            return (EditingMode.OUTPAINT, "outpaint")
        elif inpaint_detected:
            return (EditingMode.INPAINT, "inpaint")
        else:
            return (EditingMode.IMG2IMG, "img2img")
```

### 5.2 Extracción de Prompt Específico

```python
def extract_specific_prompt(self, full_prompt: str, mode: EditingMode) -> str:
    """
    Extrae la parte relevante del prompt según el modo.
    """
    # Si solo hay una acción, usar el prompt completo
    return full_prompt
```

---

## 6. Preservación de Rostros

### 6.1 Detección de Rostros

```python
# roop/img_editor/face_preserver.py

import insightface
from PIL import Image
import numpy as np

class FacePreserver:
    def __init__(self):
        self.analyzer = insightface.app.FaceAnalysis()
        self.analyzer.prepare(ctx_id=0, det_size=(640, 640))
    
    def detect_faces(self, image: Image) -> list:
        """Detecta todos los rostros en la imagen"""
        np_image = np.array(image)
        faces = self.analyzer.get(np_image)
        return faces
    
    def preserve_faces(self, original: Image, generated: Image) -> Image:
        """
        Preserva los rostros del original en la imagen generada.
        Usa insightface para face swap si es necesario.
        """
        orig_faces = self.detect_faces(original)
        gen_faces = self.detect_faces(generated)
        
        if len(orig_faces) == 0:
            return generated  # No hay rostros que preservar
        
        # Lógica de preservación (face swap con Reactor)
        # ...
        
        return generated
```

---

## 7. Interfaz de Usuario (Gradio)

### 7.1 Estructura del Tab

```python
# ui/tabs/img_editor_tab.py

import gradio as gr
from roop.img_editor.img_editor_manager import ImgEditorManager

class ImgEditorTab:
    def __init__(self):
        self.manager = ImgEditorManager()
    
    def create(self):
        with gr.Tab("[🎨] ImgEditor"):
            gr.Markdown("""
            # Editor de Imágenes con IA
            
            Sube una imagen y describe los cambios que quieres hacer.
            
            **Ejemplo:** "quiero que este de rodillas y añade dos hombres alrededor"
            """)
            
            # Fila principal
            with gr.Row():
                # Panel izquierdo - Input
                with gr.Column(scale=1):
                    input_image = gr.Image(
                        label="Imagen",
                        sources=["upload", "clipboard"],
                        type="pil",
                        interactive=True
                    )
                    
                    prompt_input = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe los cambios que quieres hacer...",
                        lines=3
                    )
                    
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt (opcional)",
                        placeholder="Lo que NO quieres...",
                        lines=2
                    )
                    
                    with gr.Accordion("Parámetros Avanzados", open=False):
                        num_steps = gr.Slider(
                            label="Pasos",
                            minimum=1,
                            maximum=20,
                            value=8,
                            step=1
                        )
                        
                        guidance_scale = gr.Slider(
                            label="Guidance Scale",
                            minimum=1,
                            maximum=20,
                            value=7.5,
                            step=0.5
                        )
                        
                        strength = gr.Slider(
                            label="Strength",
                            minimum=0,
                            maximum=1,
                            value=1.0,
                            step=0.05
                        )
                    
                    generate_btn = gr.Button(
                        "🎨 Generar",
                        variant="primary",
                        size="lg"
                    )
                
                # Panel derecho - Output
                with gr.Column(scale=1):
                    result_gallery = gr.Gallery(
                        label="Resultados",
                        show_label=False,
                        columns=2,
                        height="auto"
                    )
                    
                    status = gr.Textbox(
                        label="Estado",
                        interactive=False,
                        lines=3
                    )
                    
                    with gr.Row():
                        download_btn = gr.Button(
                            "💾 Descargar",
                            size="sm"
                        )
                        
                        use_as_input_btn = gr.Button(
                            "🔄 Usar como input",
                            size="sm"
                        )
            
            # Eventos
            generate_btn.click(
                fn=self.manager.generate,
                inputs=[
                    input_image,
                    prompt_input,
                    negative_prompt,
                    num_steps,
                    guidance_scale,
                    strength
                ],
                outputs=[result_gallery, status]
            )
            
            use_as_input_btn.click(
                fn=self.manager.set_as_input,
                inputs=[result_gallery],
                outputs=[input_image]
            )
```

---

## 8. Gestor Principal

```python
# roop/img_editor/img_editor_manager.py

import os
import uuid
import gradio as gr
from PIL import Image
from roop.img_editor.flux_client import FluxClient
from roop.img_editor.prompt_analyzer import PromptAnalyzer, EditingMode
from roop.img_editor.face_preserver import FacePreserver

class ImgEditorManager:
    def __init__(self):
        self.flux_client = FluxClient()
        self.prompt_analyzer = PromptAnalyzer()
        self.face_preserver = FacePreserver()
        self.current_image = None
        self.history = []
        self.output_dir = "output/img_editor"
        
        # Crear directorio de salida
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate(
        self,
        image: Image,
        prompt: str,
        negative_prompt: str,
        num_steps: int,
        guidance_scale: float,
        strength: float
    ) -> tuple[gr.Gallery, str]:
        """
        Genera imagen modificada según el prompt.
        """
        try:
            # Validar inputs
            if image is None:
                return None, "❌ Error: Sube una imagen primero"
            
            if not prompt or not prompt.strip():
                return None, "❌ Error: Escribe un prompt"
            
            # Analizar prompt
            mode, mode_desc = self.prompt_analyzer.analyze(prompt)
            
            status_msg = f"🔄 Generando en modo {mode_desc}...\nPrompt: {prompt}"
            
            # Cargar modelo si no está cargado
            if not self.flux_client.is_loaded():
                self.flux_client.load()
                status_msg += "\n📦 Cargando FLUX Fill Pipeline..."
            
            # Generar según modo
            if mode == EditingMode.OUTPAINT:
                result = self.flux_client.outpaint(
                    image=image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                )
            else:
                # Inpaint o img2img
                result = self.flux_client.inpaint(
                    image=image,
                    mask=image,  # TODO: implementar editor de máscara
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    strength=strength,
                )
            
            # Preservar rostros (opcional)
            result = self.face_preserver.preserve_faces(image, result)
            
            # Guardar
            filename = f"{uuid.uuid4()}.png"
            filepath = os.path.join(self.output_dir, filename)
            result.save(filepath)
            
            # Actualizar estado
            self.current_image = result
            self.history.append({
                "prompt": prompt,
                "mode": mode_desc,
                "filepath": filepath
            })
            
            return (result, f"✅ Generado correctamente\n📁 {filepath}")
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    def set_as_input(self, gallery: gr.Gallery) -> Image:
        """Usa el último resultado como nuevo input"""
        if gallery and len(gallery) > 0:
            self.current_image = gallery[0]
            return gallery[0]
        return None
```

---

## 9. Diagrama de Flujo

```mermaid
flowchart TD
    A[Usuario sube imagen] --> B[Escribe prompt]
    B --> C[Selecciona parámetros]
    C --> D[Click "Generar"]
    
    D --> E[PromptAnalyzer]
    E --> F{Modo detectado}
    
    F -->|OUTPAINT| G[FLUX Fill outpaint]
    F -->|INPAINT| H[FLUX Fill inpaint]
    F -->|IMG2IMG| I[FLUX Fill img2img]
    
    G --> J[FacePreserver]
    H --> J
    I --> J
    
    J --> K[Guardar resultado]
    K --> L[Mostrar en Gallery]
    
    L --> M{¿Más cambios?}
    M -->|Sí| N[Usar como input]
    N --> B
    M -->|No| O[Fin]
```

---

## 10. Métricas de Éxito

| Métrica | Objetivo | Medición |
|---------|----------|----------|
| Tiempo de generación | < 30 segundos | Logs de tiempo |
| Calidad visual | Rostros preservados > 90% | Validación manual |
| Detección de modo | Precisión > 80% | Test de prompts |
| VRAM usada | < 7GB | torch.cuda.max_memory_allocated |
| Sin censura | 100% sin filtros | Validación |

---

## 11. Lista de Tareas

### Fase 1: Setup (1 día)
- [ ] Crear estructura `roop/img_editor/`
- [ ] Crear `flux_client.py` con FLUX Fill Pipeline
- [ ] Test de carga del modelo

### Fase 2: Editor UI (2 días)
- [ ] Crear `img_editor_tab.py`
- [ ] Integrar en `ui/main.py`
- [ ] Test de UI básica

### Fase 3: Integración (1 día)
- [ ] Conectar UI con flux_client
- [ ] Implementar PromptAnalyzer
- [ ] Test end-to-end

### Fase 4: Face Preservation (1 día)
- [ ] Crear face_preserver.py
- [ ] Integrar InsightFace
- [ ] Test de preservación

### Fase 5: Testing (2 días)
- [ ] Testing con diferentes prompts
- [ ] Optimización de VRAM
- [ ] Documentación

---

## 12. Notas Técnicas

### VRAM con 8GB
```
FLUX Fill fp16 + Tiled VAE ≈ 4-5GB
Permite ejecutar en paralelo con SD WebUI (4-5GB)
Total ≈ 8-10GB
```

### Optimizaciones si hay problemas
1. `pipe.enable_sequential_cpu_offload()` - Offload a CPU
2. Reducir `num_inference_steps` a 4
3. Usar imágenes más pequeñas

---

## 13. Referencias

- [FLUX Fill Documentation](https://blackforestlabs.ai/flux-1-tools/)
- [Diffusers 0.35.2](https://github.com/huggingface/diffusers)
- [InsightFace](https://github.com/deepinsight/insightface)
- [Reactor Extension](https://github.com/Gourieff/sd-webui-reactor)
