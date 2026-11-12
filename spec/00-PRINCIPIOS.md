# spec/00 — Principios y Definición de Listo

## Versión vigente
- **v5.1337 (2026-11-11)**

Fuente: `AGENTS.md` (proyecto) + `C:/Users/INdaHouse/.agents/AGENTS.md` (global) + `MEMORY.md`.

## Constitución

1. **El usuario decide alcance, diseño y prioridades.** Ningún agente decide en silencio; siempre opciones concretas con recomendación.
2. **SDD**: toda funcionalidad nueva tiene historia en `spec/02-REQUISITOS.md` antes de código.
3. **Conservar los motores elegidos por el usuario.** El Editor tiene Auto como
   predeterminado (`editor_ui.default_engine=auto`); su ruta general configura
   LongCat (`route_editor.default_engine`) y la apariencia pura medida configura
   Kontext (`editor_ui.automatic_appearance_engine`). La elección explícita de
   Kontext Q4_K_M conserva el modelo de `kontext.model_file`. Los
   nombres/versiones legacy resuelven a Kontext, sin borrar archivos históricos.
   La prohibición absoluta de LongCat del 18/10 quedó superada por la autorización
   posterior registrada el 25–28/10 en MEMORY.md: existe una excepción explícita
   para la ruta medida con visión, controlada por `edit_execution.py`. Fuera de
   esa excepción se conserva el motor elegido; no es permiso para otros fallbacks.
   Sin reintroducir pintado manual.
   Aclaración del 18/10: proteger automáticamente a las otras personas mediante
   región medida en ediciones focales, sin recetas por palabras o géneros.
4. **Generate y Image Editor deben convivir**: cambiar un flujo no puede romper el otro (`AGENTS.md` del proyecto, "Objetivo").
5. **Sin hardcode**: parámetros en `config/` (`model_configs.json`, `generation_prompt.json`), no valores mágicos en el código.
6. **Local-first**: el procesamiento se ejecuta en la máquina del usuario;
   ComfyUI local en el puerto 8188 es compartido por las pestañas. Las
   dependencias de red deben declararse y no se instalan sin autorización.
7. **Evidencia real**: una afirmación de calidad exige imagen/log ejecutado con nombre de archivo fresco; los probes de despacho no valen como validación visual.
8. **Bitácora**: decisión/prueba/resultado en `MEMORY.md` con fecha. Sin registrar, no hecho.

## Restricciones del entorno

- RTX 3060 Ti 8 GB, CUDA 12.4, ComfyUI arrancado con `--normalvram`
  (`start()` en `ui/tabs/comfy_launcher.py`).
- Servidor ComfyUI 8188 compartido por Image Editor, Generate y Animate: no reiniciar ni interrumpir sin comprobar estado.
- En Image Editor el usuario prioriza cumplir la instrucción completa con rapidez, incluso si cambia caras; esa preferencia no relaja la identidad en FaceSwap.

## Definición de Listo (mínimo para cerrar una historia)

1. Código ejecutado de verdad (prueba real, no solo unitario).
2. Lint y compilación Python en verde, junto con los probes pertinentes.
3. Build del artefacto afectado en verde (launcher o APK cuando corresponda),
   con comando y resultado registrados. Compilar Python no acredita un build
   empaquetado ni una prueba de la APK instalada.
4. Sin regresión en los flujos existentes: Image Editor, Generate, FaceSwap,
   Animate y los contratos Android pertinentes.
5. Parámetros nuevos en `config/`, no en línea.
6. Reviewer independiente **APPROVE** y tester **PASS**, indicando alcance,
   versión y evidencia. Una aprobación acotada no aprueba todo el workspace.
7. `MEMORY.md` actualizado con fecha, evidencia y límites; deudas en `spec/03`.
8. Validación visual del usuario cuando el objetivo sea calidad de imagen.
   Terminar un render o pasar contratos CPU no satisface este criterio.
