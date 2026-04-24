# Img Editor UI/Backend Alignment Guide

Fecha: 2026-04-23

## Objetivo

Alinear la UI del editor de imágenes con el comportamiento real de los motores para que:

- los controles visibles correspondan a parámetros que sí se aplican;
- `FLUX.2-klein` funcione como editor de imagen predecible;
- `HART` deje de presentarse como editor mientras no exista una integración real de image editing;
- presets y sliders representen cambios reales y medibles.

## Problemas Detectados

### 1. HART no es editor en el estado actual

Situación actual:

- `hart_edit_comfy_client.py` genera imagen desde `prompt`.
- No recibe `image`, `mask`, `denoise` ni estructura visual de entrada.
- En `img_editor_manager.py` se llama a HART sin imagen base.

Consecuencia:

- La UI actual induce a pensar que HART “edita” la imagen original.
- Los controles `Preservar original`, `denoise` y parte de los presets no tienen sentido para HART.

Conclusión:

- Mientras no haya image-conditioned editing real, HART debe tratarse como motor de generación, no de edición.

### 2. FLUX.2-klein no respeta completamente lo que manda la UI

Situación actual:

- La UI calcula:
  - `creativity -> guidance_scale`
  - `preserve -> denoise`
  - `steps -> num_inference_steps`
- El cliente de FLUX luego fuerza valores para Klein:
  - `guidance=2.0`
  - `steps` capados internamente

Consecuencia:

- El usuario mueve sliders que luego no se respetan.
- Los presets visibles no corresponden al comportamiento real.

Conclusión:

- Hay que centralizar la resolución de parámetros y dejar de sobrescribirlos de forma opaca en el cliente.

### 3. La UI usa una traducción global que no sirve igual para todos los motores

Situación actual:

- `preserve` y `creativity` se convierten con una fórmula fija.
- Esa fórmula sirve de forma aceptable para FLUX, pero no para HART.

Consecuencia:

- La UI parece uniforme, pero el backend no es uniforme.
- Se mezclan controles “semánticos” con motores incompatibles.

Conclusión:

- Cada motor debe declarar qué controles soporta realmente.

### 4. Los textos de ayuda y presets no están alineados con los límites reales

Situación actual:

- La ayuda menciona rangos de pasos y comportamientos que el backend no siempre aplica.
- Algunos presets sugieren pasos altos que Klein no usa de verdad.

Consecuencia:

- La UX es confusa.
- Es difícil ajustar el motor con criterio porque la UI promete más control del que existe.

## Principios de Diseño

### 1. La UI no debe mentir

Si un motor no usa un parámetro:

- no debe mostrarse;
- o debe mostrarse desactivado y claramente etiquetado como no aplicable.

### 2. Un solo lugar decide los parámetros finales

Debe existir una función central que reciba:

- motor
- intención de edición
- preserve
- creativity
- steps
- resolución

y devuelva:

- `final_denoise`
- `final_guidance`
- `final_steps`
- flags específicas del motor

### 3. Los motores deben exponer capacidades

Cada motor debe declarar algo equivalente a:

- `supports_img2img`
- `supports_denoise`
- `supports_guidance`
- `supports_steps`
- `supports_identity_preservation`
- `mode = edit | generate`

### 4. Los presets deben ser semánticos

En vez de depender de números abstractos, conviene definir presets por intención:

- `Retoque suave`
- `Cambio moderado`
- `Cambio fuerte`
- `Regenerar escena`

## Cambios Recomendados en Backend

### A. Separar `edit` y `generate`

Objetivo:

- `HART` debe salir del flujo de edición si no puede editar la imagen original.

Acciones:

- Clasificar `hart` como `generate`.
- Mantener `flux_klein`, `qwen`, `zimage` como `edit`.
- Si el usuario selecciona HART desde una pestaña de edición, dejar claro que generará una imagen nueva inspirada en el prompt.

### B. Centralizar la resolución de parámetros

Crear una función, por ejemplo:

`resolve_engine_params(engine, creativity, preserve, steps, resolution_label, intent=None)`

Debe devolver algo similar a:

```python
{
    "guidance_scale": 2.2,
    "denoise": 0.42,
    "num_inference_steps": 8,
    "target_width": 1024,
    "target_height": 768,
    "ui_summary": "Klein | denoise=0.42 | cfg=2.2 | steps=8"
}
```

Ventajas:

- evita que la UI decida una cosa y el cliente aplique otra;
- permite clamps por motor sin comportamiento oculto;
- hace más fácil depurar.

### C. Hacer que FLUX.2-klein respete sliders con clamps, no con overrides duros

Comportamiento recomendado:

- `guidance_scale`: respetar valor UI, limitado a un rango seguro para Klein.
- `num_inference_steps`: respetar valor UI, limitado por VRAM/tiempo.
- `denoise`: respetar el valor resuelto por backend.

Rangos sugeridos para Klein:

- `guidance_scale`: `1.5 - 3.0`
- `num_inference_steps`: `4 - 12`
- `denoise`: `0.15 - 0.95`

Notas:

- Si por estabilidad hace falta limitar, debe hacerse con `clamp`.
- El log debe mostrar siempre el valor final aplicado.

### D. Crear perfiles de edición reutilizables

En vez de depender solo de sliders libres, tener perfiles internos:

- `subtle_edit`
- `balanced_edit`
- `strong_edit`
- `full_regeneration`

Tabla sugerida para Klein:

| Perfil | Denoise | Guidance | Steps |
|---|---:|---:|---:|
| subtle_edit | 0.20-0.30 | 1.8-2.2 | 6-8 |
| balanced_edit | 0.35-0.55 | 2.0-2.6 | 8-10 |
| strong_edit | 0.60-0.80 | 2.2-3.0 | 8-12 |
| full_regeneration | 0.85-0.95 | 2.4-3.0 | 10-12 |

## Cambios Recomendados en UI

### A. Ocultar controles incompatibles por motor

Si motor = `hart`:

- ocultar `Preservar original`;
- ocultar cualquier texto asociado a `denoise`;
- cambiar la descripción del motor a “genera una imagen nueva”.

Si motor = `flux_klein`:

- mostrar `Preservar original`;
- mostrar `Creatividad`;
- mostrar `Pasos`;
- mostrar resumen de parámetros aplicados.

### B. Mostrar parámetros efectivos

Debajo del botón de generación o en la caja de estado:

- `Motor: FLUX.2-klein`
- `Denoise aplicado: 0.42`
- `Guidance aplicado: 2.2`
- `Steps aplicados: 8`

Esto evita que el usuario piense que el slider tiene un efecto distinto del real.

### C. Redactar mejor los controles

Texto sugerido:

- `Creatividad`: “Cuánta libertad tiene el modelo para cambiar el contenido”
- `Preservar original`: “Cuánto mantiene pose, fondo y estructura de la imagen”
- `Pasos`: “Más pasos = más calidad y más tiempo”

Para HART:

- `Creatividad` puede renombrarse a `Guidance` o mantenerse solo si de verdad se usa.
- `Preservar original` no debe existir.

### D. Rehacer presets visibles

Presets recomendados:

- `Retoque suave`
- `Cambio de ropa / peinado`
- `Cambio fuerte`
- `Regenerar escena`

Ejemplo orientativo para Klein:

- `Retoque suave`: preserve alto, creativity baja-media, steps 6-8
- `Cambio de ropa / peinado`: preserve medio, creativity media, steps 8-10
- `Cambio fuerte`: preserve bajo, creativity alta, steps 8-12
- `Regenerar escena`: preserve muy bajo, creativity alta, steps 10-12

## Secuencia Recomendada de Implementación

### Fase 1. Corregir el contrato backend

1. Añadir una función central de resolución de parámetros.
2. Hacer que `flux_klein` respete `guidance`, `steps` y `denoise` resueltos.
3. Añadir logs con valores finales efectivos.

### Fase 2. Alinear la UI

1. Ocultar controles incompatibles al seleccionar HART.
2. Mostrar resumen de parámetros efectivos.
3. Corregir textos de ayuda y presets.

### Fase 3. Limpieza conceptual

1. Separar visualmente `Editores` y `Generadores`.
2. Decidir si HART permanece en la pestaña actual o pasa a otra pestaña/sección.

## Cambios Mínimos Recomendados para la Próxima Pasada

Si se quiere una mejora rápida y segura:

1. Backend:
   - dejar de forzar `guidance=2.0` en Klein;
   - dejar de capar `steps` a `8` de forma silenciosa;
   - usar clamps visibles en logs.

2. UI:
   - ocultar `Preservar original` cuando `engine == "hart"`;
   - añadir un texto con parámetros efectivos.

3. Copy/UI:
   - cambiar la descripción de HART a “Generación autoregresiva, no edición”.

## Decisión Pendiente

Antes de intentar que HART “edite”, confirmar una de estas dos direcciones:

1. `HART` se mantiene como generador puro y se ajusta la UI a esa realidad.
2. Se busca una integración real de image editing para HART, lo que implicaría otra arquitectura y probablemente otro flujo de entrada.

Recomendación:

- tomar la opción 1 primero;
- dejar la opción 2 para una fase posterior y explícita.
