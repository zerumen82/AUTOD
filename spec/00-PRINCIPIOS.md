# spec/00 — Principios y Definición de Listo

Fuente: `AGENTS.md` (proyecto) + `C:/Users/INdaHouse/.agents/AGENTS.md` (global) + `MEMORY.md`.

## Constitución

1. **El usuario decide alcance, diseño y prioridades.** Ningún agente decide en silencio; siempre opciones concretas con recomendación.
2. **SDD**: toda funcionalidad nueva tiene historia en `spec/02-REQUISITOS.md` antes de código.
3. **Turbo es el motor de edición elegido por el usuario** (`LongCat-Image-Edit-Turbo-Q4_K_S.gguf`, denoise 1,0). No cambiar a Full sin aprobación.
4. **Generate y Image Editor deben convivir**: cambiar un flujo no puede romper el otro (`AGENTS.md` del proyecto, "Objetivo").
5. **Sin hardcode**: parámetros en `config/` (`model_configs.json`, `generation_prompt.json`), no valores mágicos en el código.
6. **Local-first**: todo el cómputo es en la máquina; el único servicio es ComfyUI local en el puerto 8188.
7. **Evidencia real**: una afirmación de calidad exige imagen/log ejecutado con nombre de archivo fresco; los probes de despacho no valen como validación visual.
8. **Bitácora**: decisión/prueba/resultado en `MEMORY.md` con fecha. Sin registrar, no hecho.

## Restricciones del entorno

- RTX 3060 Ti 8 GB, CUDA 12.4, ComfyUI arrancado con `--normalvram` (`ui/tabs/comfy_launcher.py:355-364`).
- Servidor ComfyUI 8188 compartido por Image Editor, Generate y Animate: no reiniciar ni interrumpir sin comprobar estado.
- En Image Editor el usuario prioriza cumplir la instrucción completa con rapidez, incluso si cambia caras; esa preferencia no relaja la identidad en FaceSwap.

## Definición de Listo (mínimo para cerrar una historia)

1. Código ejecutado de verdad (prueba real, no solo unitario).
2. `py_compile` + probes pertinentes en verde.
3. Sin regresión en los flujos existentes (Image Editor, FaceSwap, Animate siguen funcionando).
4. Parámetros nuevos en `config/`, no en línea.
5. `MEMORY.md` actualizado con fecha, evidencia y límites.
6. Validación visual del usuario cuando el objetivo sea calidad de imagen.
