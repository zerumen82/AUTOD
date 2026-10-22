# Relevo — selector visual del Editor

> **Relevo vigente solicitado el19/10: `plans/HANDOFF_editor_selection_20261019.md`.**
> Incluye autorización explícita de SAM2, pruebas posteriores y procesos pendientes.

> Actualización 2026-10-19: este relevo describe el estado histórico del 18/10.
> Ya existen backend de selección múltiple, ruta temprana del manager y helper
> de edición; el inventario abierto se resolvió en dos etapas locales. Estado
> vigente y evidencias en la cabecera de MEMORY.md. No volver a seguir el
> apartado «no implementado» como si describiese el código actual.
> UI e integración también completadas, backend/UI 14/14 y HTTP real PASS;
> reviewer APPROVE del código. La historia permanece abierta por máscaras
> parciales/etiquetas incorrectas e instrucciones visuales incumplidas.

Fecha: 2026-10-18, 23:26 Europe/Madrid. El usuario pide anotar el punto para
la siguiente sesión. El plan está **APROBADO** con «hazlo». Retomar su ejecución;
no volver a pedir aprobación del mismo alcance.

## Petición y límites vigentes

Analizar la foto, mostrar personas/objetos medidos, seleccionar uno por clic o
lista, resaltar su región y editar con Kontext protegiendo el exterior. Una
selección por edición. El usuario insiste: **NO QUIERO HARDCODING**.

- Plan: `plans/editor_object_selection_20261018.md`.
- Historia aprobada: `spec/02-REQUISITOS.md`, H-EDIT-SELECT.
- No listas de objetos/géneros/prendas/ediciones para enrutar. Categorías abiertas
  del análisis visual y regiones verificadas por detección/segmentación.
- Sin instalar paquetes/modelos, recuperar SAM2/GrabCut rechazados, cambiar APK
  o volver a pintado libre. Los antiguos endpoints Android conservan 13/9/8.
- No convertir coordenadas propuestas por Moondream en máscaras. El historial
  ya documenta coordenadas falsas. No modificar fotos explícitas como prueba.

## Estado exacto al dejarlo

**El selector NO está implementado.** No hay UI nueva ni integración nueva en
el manager. No existen `element_selection.py` ni `selected_region_edit.py`.
No confundir el trabajo de protección focal previo con este selector nuevo.

En esta continuación solo se escribieron configuración/probe y documentación:

- `config/editor_selection.json`: CPU, rutas locales, prompt/schema de inventario,
  thresholds de segmentación, preview y sección `edit`. Es un borrador técnico
  aún no validado como configuración final del backend.
- `diagnostics/probe_editor_selection_capability_20261018.py`: preflight real.
- Plan marcado aprobado; aceptación en spec/02; arquitectura propuesta en spec/01;
  bitácora en MEMORY.md. No se hicieron commits ni se tocaron cambios ajenos.

Última comprobación de esos dos archivos: Ruff completo exit0, py_compile exit0
y JSON válido. Esto no prueba inventario ni selector funcionales.

## Pruebas reales realizadas y fallo pendiente

1. Detector ONNX CPU local, sin downloader de FaceAnalysis: 2 caras en el fixture
   no explícito `diagnostics/out/kontext_review_20261018/two_people_source.jpg`,
   ninguna en `venv/Lib/site-packages/skimage/data/coffee.png` y una en astronauta
   de skimage. No afirmar por ello que detecta personas enteras.
2. Moondream2 CPU describe en prosa. Para café devolvió
   `A cup of coffee and a spoon on a saucer.`; en dos personas describió trajes,
   corbatas y reloj. En astronauta alucinó también un cohete.
   La primera corrida falló al cerrar el handler (`close` no existe), antes de
   segmentar/guardar resultados. Carpeta `capability_1792358620050564300` vacía;
   no existe evidencia JSON de esa corrida. No repetir aquel cleanup incorrecto.
3. Corregido cleanup con `_exit_stack.close`. Segunda corrida con schema/gramática
   JSON: café produjo `{"objects":[]}` en 3,33 s. Cargó pesos CLIPSeg CPU reales,
   pero al no haber etiquetas no hubo máscaras. Proceso exit0 significa ejecución
   del probe, **no capacidad de objetos superada**.

Evidencia guardada:
`diagnostics/out/editor_selection_20261018/capability_1792358690175673500/results.json`.

**Punto siguiente inmediato:** resolver y verificar el inventario abierto de
objetos antes de escribir el backend/UI. No rellenar la lista a mano para la foto.
Puede evaluarse extracción semántica de frases nominales de la descripción con
un LLM local ya instalado, conservando la validación por segmentación. No es aún
una solución implementada ni probada. Si los recursos instalados no bastan,
presentar evidencia y consultar una ampliación del diseño antes de instalar nada.

Comando para repetir el preflight (comprobar primero el estado de la máquina):

```powershell
& .\venv\Scripts\python.exe -u -m diagnostics.probe_editor_selection_capability_20261018 --only coffee
```

## Contratos de implementación acordados, todavía sin código

Backend `roop/img_editor/element_selection.py`:

- `SelectionRegion(id, label, mask, bbox, kind)` y
  `SelectionAnalysis(fingerprint, size, regions, message)`.
- `analyze_elements(image)`, `image_fingerprint(image)`,
  `validate_selection(image, analysis, id)`,
  `render_selection(image, analysis, selected_id=None)` y
  `choose_at_point(analysis, x, y)`.
- Para cabezas, reutilizar geometría medida `_head_mask_from_faces`; etiqueta
  honesta «Cabeza/rostro N», `kind="head"`. No llamarla persona entera.
  Objetos: etiquetas de inventario visual abierto + máscaras CLIPSeg reales,
  componentes/coverage/dedup configurables. Un VLM no aporta coordenadas fiables.

UI prevista en `ui/tabs/img_editor/ui.py`, `events.py`, `state.py`:

- Botón Analizar, vista con contornos, dropdown/lista y selección por clic sobre
  la máscara efectiva (no solamente bbox); ambigüedad visible.
- `gr.State` por sesión, con análisis/selección y huella RGB canonical. No usar
  `state.last_input_image` global para resolver selección entre usuarios.
- Imagen nueva/quitar imagen/reutilizar resultado invalidan todas las propuestas.
  Un análisis tardío de A no puede instalarse sobre B: revisión/token por sesión.
- Compartir exclusión de trabajo con generación para evitar cargar modelos
  mientras ComfyUI renderiza. No alterar renders activos.
- Conservar `/edit_select` 13 mediante botón oculto legacy, `/edit_options` 9 y
  `/edit` 8. Botón de escritorio nuevo endpoint separado `edit_region`; una opción
  técnica discutida es 13 controles + un `gr.State` de sesión (14 inputs internos).
- Gradio instalado sí deepcopy del valor inicial de State y después devuelve el
  objeto de estado de sesión por referencia (`SessionState.__getitem__`). Puede
  usarse contenedor mutable de sesión con versión para invalidación concurrente;
  si contiene locks debe ser deepcopyable. Esta solución aún no se ha escrito.

Manager previsto:

- Parámetros opcionales separados `selection_analysis=None`,
  `selected_region_id=None` al final de `generate_intelligent`, conservando API.
- Validar huella y máscara ANTES de backend/preflight pesado. Selección explícita
  gana al referente que infieran texto/rewriter/body/pose. Foto limpia al cliente;
  nunca overlay. Propuesta: helper nuevo `selected_region_edit.py` invocado temprano
  para evitar que las ramas automáticas sobrescriban la selección.
- Crop Kontext, composición limitada al soporte real y cabezas ajenas excluidas
  después del feather. Restaurar exterior tras cualquier acabado. Ante error de
  región/crop/integración conservar fuente con aviso, sin fallback global.
- Perillas/prompts en `editor_selection.json/edit`; receta nativa de Kontext en
  `model_configs.json/kontext`. Verificar tamaños mínimos del latente y evitar
  recortes de dimensión cero. No añadir recetas específicas por palabras.

## Verificación pendiente

- Tester no ha escrito nuevos probes del selector. Deben cubrir huella/tamaño/id,
  máscara real al clicar, dos sesiones aisladas, análisis tardío, nueva foto y
  reutilización, prioridad de selección y conservación del exterior tras acabado.
- Mantener las 16 pruebas previas de protección y contratos Android 13/9/8.
- Probar UI local SERVIDA y callbacks, y revisar visualmente ediciones reales
  completadas de persona y objeto. Solo construir Blocks no basta.
- Reviewer no ha aprobado el nuevo selector: aún no existe el código. El APPROVE
  previo corresponde únicamente al retiro de LongCat/protección focal anterior.

## Procesos y workspace

Al guardar este relevo no queda ningún preflight/inferencia del selector activo.
ComfyUI sí continúa en 8188 (worker PID22944, lanzador PID25964); cola running y
pending vacías a las 23:26. No UI en 7860/7861. MUSICIA está activo en otro proyecto:
comprobar GPU/colas antes de pruebas. No reiniciar ni purgarlo.

El workspace ya traía cambios del trabajo previo y otros ajenos; hay borrados de
dos PNG en generate y un JSON de .antigravitycli modificado que no son del selector.
No revertirlos, commitearlos ni limpiarlos. Los probes/artefactos nuevos pueden
estar sin seguimiento o bajo reglas de ignore: `git status --untracked-files=no`
no los enumera. Revisar archivos existentes además del diff antes de retomar.
