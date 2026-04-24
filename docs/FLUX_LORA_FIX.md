# Solución de Problemas: LoRA Incompatible para FLUX.2-klein

## Problema Detectado

El LoRA actual `Flux%20Klein%20-%20NSFW%20v2.safetensors` (158MB) es **incompatible** con FLUX.2-klein-4B.

**Errores en log**:
```
ERROR lora diffusion_model.single_blocks.0.linear1.weight shape '[27648, 3072]' is invalid for input of size 150994944
```

Esto significa que el LoRA fue entrenado para FLUX.1-dev (8B) y no para FLUX.2-klein (4B).

## Solución Inmediata Aplicada

1. **LoRAs incompatibles renombrados** (para que no se intenten usar):
   - `models/loras/Flux%20Klein%20-%20NSFW%20v2.safetensors` → `.incompatible`
   - `models/loras/flux2klein_nsfw.safetensors` → `.incompatible`

2. **Mejora en detección**: El código FluxEdit ahora detecta errores de `shape mismatch` y descarta automáticamente LoRAs incompatibles.

3. **Resultado**: El sistema ahora usa el **modelo base FLUX.2-klein-4b** sin LoRA, que funciona correctamente.

## Opciones para NSFW (por orden recomendado)

### Opción 1: Usar el modelo base (NSFW funciona)
El modelo base FLUX.2-klein-4B + text encoder GGUF uncensored (`qwen3-4b-abl-q4_0.gguf`) ya permite generar contenido NSFW con prompts apropiados.
- **Ventaja**: Sin LoRA, más rápido, menos errores.
- **Desventaja**: Calidad menos específica que con un LoRA entrenado.

### Opción 2: Descargar un LoRA compatible
Requisitos para un LoRA compatible con FLUX.2-klein-4B:
- Tamaño: 100-300MB (no >400MB, esos son para 9B)
- Debe estar entrenado específicamente para `FLUX.2-klein-4B` o `flux_klein`
- Nombre debe contener `klein` y **no** `dev`, `schnell`, `flux1_`

**Repositorios públicos recomendados**:
1. `DeverStyle/Flux.2-Klein-Loras` - Varios estilos (algunos NSFW)
2. `fal/flux-2-klein-4B-*` - Oficial de FAL (verificar compatibilidad)
3. `nphSi/Flux.2-klein-4B_Lora` - Experimentos (algunos requieren login)

**Comando de ejemplo** (requiere `huggingface-cli` login):
```bash
huggingface-cli login
huggingface-cli download DeverStyle/Flux.2-Klein-Loras alba_baptista_vrtlalbabaptista_flux2_klein_4b.safetensors --local-dir "D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models\loras"
```

### Opción 3: Entrenar tu propio LoRA
Usa OneTrainer o Kohya SS con:
- Base model: `black-forest-labs/FLUX.2-klein-base-4B`
- Resolución: 1024x1024 (o 512x512 para ahorrar VRAM)
- Pasos: 1000-2000
- Tamaño LoRA: rank=32 o 64 (150-300MB)

## Verificación

Para verificar que el LoRA es compatible:
1. Tamaño debe ser < 400MB (para Klein 4B)
2. Nombre debe contener `klein` o `flux2_klein`
3. NO debe contener `dev`, `schnell`, `flux1_`, `8B`, `9B`
4. Al cargar, NO debe mostrar errores `shape mismatch` en logs

## Cambios en Código Realizados

### ProcessMgr.py (Tracking mejorado)
- Umbral de verificación de embeddings: 0.15 → **0.08**
- Añadido fallback por movimiento consistente (tolerancia 50px)
- Reacquire umbral: 0.45 → **0.30**
- Reacquire dinámico: acepta 0.25-0.30 si hay movimiento consistente
- Reacquire actualiza embedding de referencia (refinado)

### flux_edit_comfy_client.py
- Filtrado mejorado de LoRAs (excluye dev/schnell para flux_klein)
- Detección de errores de shape en mensajes de ComfyUI
- Auto-descarte de LoRAs incompatibles

## Resetear Configuración

Si ya has probado con el LoRA incorrecto:
1. Asegúrate de que los archivos `.incompatible` estén renombrados
2. Reinicia la aplicación (ComfyUI + Gradio)
3. En la UI de Image Editor, selecciona motor `flux_klein`
4. El sistema automáticamente elegirá el modelo base o un LoRA compatible si existe

## Contacto

Si necesitas un LoRA NSFW específico, puedes entrenar uno propio o buscar en HuggingFace colecciones públicas de Flux2.Klein LorAs.
