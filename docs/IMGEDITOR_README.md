# ImgEditor - Editor de Imagenes con FLUX

## Descripcion

ImgEditor es un modulo para editar imagenes usando **FLUX Fill Pipeline** de Black Forest Labs. Permite:

- **Inpainting**: Modificar areas especificas de una imagen
- **Outpainting**: Anadir contenido nuevo alrededor de la imagen
- **Img2Img**: Transformacion global de la imagen
- **Preservacion de rostros**: Mantener identidad facial

## Requisitos

### Hardware
- **GPU**: 8GB VRAM (optimizado para fp16)
- **RAM**: 16GB+
- **Almacenamiento**: 30GB+ para el modelo

### Software
```bash
# Dependencias requeridas
pip install diffusers transformers torch pillow
pip install insightface  # Para deteccion de rostros
```

## Instalacion

El modulo ya esta integrado en AUTO-DEEP. Para usarlo:

1. Asegurate de que las dependencias esten instaladas
2. Ejecuta la aplicacion principal
3. Ve a la pestaña **[ImgEditor]**

## Uso desde Codigo

### Ejemplo basico

```python
from roop.img_editor.img_editor_manager import get_img_editor_manager
from PIL import Image

# Obtener el manager
manager = get_img_editor_manager()

# Cargar imagen
image = Image.open("mi_imagen.jpg")

# Generar
result, msg = manager.generate(
    image=image,
    prompt="quiero que este de rodillas y anade dos hombres alrededor",
    negative_prompt="",
    num_inference_steps=8,
    guidance_scale=7.5,
)

if result:
    result.save("resultado.jpg")
    print(msg)
```

### Inpainting con mascara

```python
from roop.img_editor.img_editor_manager import get_img_editor_manager
from PIL import Image

manager = get_img_editor_manager()

# Crear mascara (blanco = area a modificar)
mask = Image.new("L", image.size, 0)

# Dibujar area (ej: cara)
from PIL import ImageDraw
draw = ImageDraw.Draw(mask)
draw.ellipse([100, 50, 300, 250], fill=255)

# Generar inpainting
result, msg = manager.generate_inpaint(
    image=image,
    mask=mask,
    prompt="cara de persona sonriendo",
    num_inference_steps=12,
)
```

## Parametros

### generate()

| Parametro | Tipo | Default | Descripcion |
|-----------|------|---------|-------------|
| `image` | PIL.Image | Requerido | Imagen de entrada |
| `prompt` | str | Requerido | Descripcion de cambios |
| `negative_prompt` | str | "" | Lo que NO queremos |
| `num_inference_steps` | int | 8 | Pasos de inferencia (4-20) |
| `guidance_scale` | float | 7.5 | Adherencia al prompt (1-20) |
| `strength` | float | 1.0 | Intensidad del cambio (0-1) |
| `seed` | int | None | Semilla para reproducibilidad |

## Arquitectura

```
roop/img_editor/
├── __init__.py              # Inicializacion del modulo
├── flux_client.py           # Cliente FLUX Fill Pipeline
├── prompt_analyzer.py       # Analisis de prompts
├── face_preserver.py        # Preservacion de rostros
└── img_editor_manager.py    # Orchestrator principal

ui/tabs/
└── img_editor_tab.py        # Interfaz Gradio
```

## Optimizacion de VRAM

El modulo usa las siguientes optimizaciones para funcionar con 8GB VRAM:

1. **fp16 (half precision)**: Reduce VRAM a la mitad
2. **Tiled VAE**: Procesa la imagen en tiles
3. **Attention Slicing**: Reduce pico de VRAM
4. **Sequential CPU Offload** (opcional): Mueve layers a CPU

## Solucion de Problemas

### "Diffusers no disponible"

```bash
pip install diffusers transformers torch pillow
```

### "Out of memory"

- Reduce `num_inference_steps` a 4
- Usa imagenes mas pequenas
- Habilita CPU offload:
  ```python
  pipe.enable_sequential_cpu_offload()
  ```

### Slow generation

- Reduce el tamano de la imagen de entrada
- Usa menos pasos de inferencia
- Asegurate de tener CUDA instalado correctamente

## Links

- [FLUX.1-fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-fill-dev)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [InsightFace](https://github.com/deepinsight/insightface)
