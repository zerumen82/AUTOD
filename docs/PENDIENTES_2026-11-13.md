# Pendientes — Image Editor y selector «Tipo de edición» (2026-11-13)

## Revisión 2026-11-14 — v5.1355 en curso

El usuario autorizó revisar P1–P15 por prioridad. P1: matriz de nueve
categorías lanzada con una foto real de astronauta; resultados y revisión
visual pendientes en `diagnostics/out/editor_categories_20261114/`. P2 sigue
sin repetición ni PASS visual. P13: cámara corregida y sus probes verdes;
selección corporal 13/13 con «solo lo pedido» aprobado. El probe compuesto
sigue en FAIL por expectativas de generación sin sujeto medido (log
`logs/probe_p13_compound_20261114.log`). P3–P12 y P14–P15 siguen abiertos.
Ningún PASS visual se ha firmado por anticipado.

### Revisión 2026-11-14 (v5.1358) — probes, halo de `background`, reinicios
Revisión visual real de la matriz (`diagnostics/out/editor_categories_20261114/`)
comparada con la fuente `diagnostics/out/editor_camera_neutral_20261110/before.png`:
- **P13 cerrado**: los tres fallos citados ya no existen en el código actual.
  `probe_editor_viewpoint_dispatch` 4 OK, selección corporal 13/13, compuesto
  61 OK / 0 FAIL, selector 31/31, cámara 4/4, `only_finish` 5/5, descalzo 9/9.
  Todos con exit 0 (`PYTHONPATH` de proyecto y `venv/Scripts/python.exe`).
- `clothing` → **corregido**: el traje sale azul en torso y brazos con cara y
  fondo conservados; quedan restos naranjas en hombro/brazo derecho. Log
  `logs/p1_clothing_remove_20261114.log`: máscara `astronaut suit` 31,4 %,
  `changed_fraction` 0,19. El dato «solo cambia el 2,86 %» de la sección
  anterior es de la corrida ANTERIOR al fix: ya no describe el código actual.
- `body` → **FAIL, se retira el PASS anterior**: el astronauta sí es más alto,
  pero con estiramientos y fantasmas (brazos duplicados, bandera deformada).
  Subir la altura se cumple; la fidelidad del resto no.
- `background` → causa del halo medida y arreglada (ver más abajo); el PASS
  visual queda pendiente de un render con GPU libre.
- `remove` → **FAIL (mejorado)**: el invento desaparece con LaMa (v5.1360);
  queda un resto del visor en la esquina inferior derecha (objeto real, no
  invención). Ver «`remove`: LaMa (v5.1360)» más abajo.

**`remove`: qué se midió y por qué sigue en FAIL (v5.1359).**
Petición «Remove the black helmet from the photo». Medido con la fuente y el
resultado (`before.png` vs `remove.png`, 512²):
- La medición es correcta: máscara `black helmet` 11,9 %, bbox (283, 301, 508,
  512), área cambiada 8,7 % y **nada cambia fuera de la zona** (0 píxeles en la
  mitad izquierda). El casco desaparece por completo.
- Lo inventado está **entero dentro de la zona** y cambia de forma en cada
  semilla: manga nueva con parche de bandera y insignia redonda, o parche de
  bandera + insignia + conector, o franja de rayas. Es relleno del modelo, no
  un error de máscara ni de despacho.
- El pase local corre a `denoise=1.00` con `cfg=1.0` y `keep_source_pixels`:
  dentro de la máscara el muestreador arranca de ruido y pinta una escena
  nueva. Con `cfg=1.0` el prompt negativo se ignora, así que la única palanca
textual es la frase positiva.
Mecanismos probados y **rechazados con evidencia visual** (todos medidos, no
supuestos):
1. Frase positiva «no copiar emblemas, insignias, logos, texto ni herrajes» en
   `removal_reconstruction_prompt` (config): dos renders con la frase y el
   defecto sigue (cambia la forma, no desaparece). Cláusula se mantiene porque
   expresa el requisito, pero **no se declara eficaz**.
2. Relleno del hueco desde el borde (`_continue_measured_surface`, NS) antes
   de muestrear: el casco queda como masa oscura (`/tmp/remove_ns_fill.png`).
   Es el mismo límite que ya documenta el comentario de
   `SURFACE_CONTINUATION_DENOISE` («refining an inpaint left the previous
   surface as a shadow»).
3. Segunda pasada de armonización (vaciar desde el borde + muestrear a fuerza
   0,60, la receta de las caras insertadas): deja una mancha difusa sobre el
traje y restos del casco (`logs/p1_remove_harmonized_20261114.log`). Más
   extraño que el invento. Revertido.
4. Reparación determinista por paleta del anillo (quitar píxeles fuera de la
   paleta del borde e inpintar): borra la bandera y el aro de la insignia, pero
   deja un borrón azul que se nota (`/tmp/remove_palette_zoom.png`).
5. Detección de la copia por NCC de plantillas: **no discrimina**. En
   `remove.png` el máximo es 0,749 mientras que la edición correcta de
   `clothing` da 0,997, así que un umbral razonable marca la buena y no la mala.
Conclusión: no hay arreglo limpio dentro de la ruta actual (prompt + máscara +
muestreo). Las opciones que quedan son otra cosa, no un retoque:
- un modelo de inpaint dedicado solo para `structural_remove` (relleno de hueco
  sin texto, p. ej. LaMa ONNX, que corre en CPU), medido contra estas dos
  regresiones antes de integrarlo;
- o aceptar el relleno generativo como coste de «quitar» un objeto que tapa a
  la persona y firmar el PASS con esa observación.
Ninguna de las dos se hizo: la primera es un modelo nuevo y la segunda es una
decisión del usuario.

**`remove`: LaMa (v5.1360).**
Se integró un modelo de inpaint dedicado: LaMa big-lama en ONNX
(`models/lama_fp32.onnx`, 199 MB, entrada fija 512², CPU), módulo
`roop/img_editor/lama_inpaint.py`. La rama local del pase se sustituye por
LaMa cuando `structural_remove` + máscara medida. El muestreo difuso a
fuerza 1,00 (que arrancaba de ruido dentro de la máscara) deja de usarse
para esta categoría; el tiempo del pase baja de 85–180 s a ~30 s.
Resultado visual: el casco desaparece **sin** inventar insignias, banderas ni
conectores; el relleno continúa traje y fondo. Medido: changed_bbox
(283, 301, 508, 512), 14,9 % cambiado, 0 píxeles fuera de la zona.
Límite residual documentado: un resto del visor (objeto real, no invención)
en la esquina inferior derecha, porque la máscara CLIPSeg («black helmet»)
no cubre el cristal reflectante. Se probó dilatar la máscara 4–16 px para
incluirla: LaMa rellena peor y deja un parche moteado; se eligió no dilatar
(menos daño). Opciones abiertas: refinar la máscara para incluir el visor
(color/region grow) o un segundo pase LaMa solo sobre ese fragmento.

**Halo de `background` (arreglado en código, PASS visual pendiente).**
Defecto: franja de fondo viejo pegada al pelo, 3–5 px visibles. Causa doble,
medida con `diagnostics/diag_background_halo_band_20261114.py` (CPU):
1. `_background_mask_from_people` dilataba la silueta 7 px, así que la zona
   repintada empezaba 2,27 px fuera de la persona y esa banda conservaba el
   fondo de estudio. Ahora acepta `gap` y la ruta de escenario pasa 1 px →
   franja **0,00 px** (escenario pintado 59,4 %, antes 57,3 %).
2. La silueta medida sobrecubre ~1 px de fondo en la mitad de su contorno
   (45 % de los píxeles de borde ya son fondo claro), y `_restore_measured_people`
   la devolvía tal cual: ese filo volvía como halo gris. Ahora se entrega 1 px
   por dentro, con el mismo desvanecido de 2 px.
Las otras rutas mantienen `gap=7` (no cambian). Probes vecinos verdes tras el
cambio: escena 11/11, guards de halo/cara PASS, selector 31/31. Siguen en FAIL
`probe_scene_head_preserve` y `probe_editor_structural_zone_close`, y **fallan
igual con el cambio revertido** (verificado hoy), así que no son regresión.

**Reinicios del PC**: dos `Event 41 Kernel-Power` (20:34:46 y 21:38:27) sin
BSOD ni WHEA, con 17 errores `disk` Id 51 en el disco USB de 2 TB (unidad `E:`)
antes del segundo. Mitigación aplicada: `DISKIDLE=0` (los discos ya no se
apagan). Detalle y pasos de hardware en `docs/DIAGNOSTICO_REINICIOS_2026-11-14.md`.

Estado al cierre de v5.1354. **Este documento manda para no declarar resuelto
nada sin PASS visual real** (AGENTS.md). Diseño y contratos:
`docs/IMG_EDITOR_SELECTOR_CATEGORIAS_2026-11-13.md`.

## P1. PASS visual por categoría (bloqueante de cierre)
Ninguna de las 9 categorías tiene PASS visual de una foto real. Los probes son
de despacho y de píxel; no sustituyen la validación visual.
- Categorías: `pose`, `clothing`, `body`, `features`, `background`, `camera`,
  `add`, `remove`, `people`.
- Mínimo por categoría: una foto real, log de la petición y resultado revisado a
  ojo. Registrar PASS/FAIL en este documento con fecha y archivo de salida.

### Resultados visuales 2026-11-14 (matriz real, foto de astronauta, v5.1355)
Fuente `diagnostics/out/editor_camera_neutral_20261110/before.png`; salidas en
`diagnostics/out/editor_categories_20261114/`; log `logs/editor_categories_20261114.log`.
- `pose` → **PASS con observación**: brazo derecho levantado visible; identidad y
  escenario cambiados (la geometría sale del modelo por diseño de la categoría).
- `clothing` → **PASS visual (v5.1361)**: traje azul visible en torso y brazos,
  cara y fondo conservados; changed 19,16 % (bbox 31,198,355,505). La
  medición de persona ya no falla y el modelo aplica el color.
- `body` → **FAIL (parcial)**: el astronauta sí es más alto (bbox 0,43,389,512),
  pero con estiramientos y artefactos en la cara (composición recortada).
  Subir la altura se cumple; la fidelidad del resto no.
- `features` → **PASS**: pelo negro visible y resto conservado (Moondream local dijo
  FAIL y el pipeline conservó el candidato; el ojo ve el cambio).
- `background` → **FAIL por fidelidad**: el bosque sí sale, pero duplica a la persona
  en dos astronautas y cambia ambas caras.
- `camera` → **PASS**: ángulo alto visible, identidad conservada.
- `add` → **PASS con observación**: bandera pequeña añadida y visible, pegada al
  casco en vez de «al lado» del astronauta.
- `remove` → **FAIL**: el casco negro desaparece por recorte de cuadro, no por
  eliminación; además aparece un parche azul nuevo en el pecho.
- `people` → **PASS**: segundo astronauta añadido y la primera persona conservada.
Conclusión: 5 PASS (2 con observación) y 4 FAIL. P1 sigue abierto: clothing/body
exigen arreglar la medición de persona; background y remove exigen arreglar la ruta.

### Resultados visuales 2026-11-14 (tras fix de medición, v5.1356)
Causa raíz confirmada por reproducción: `_visual_subject_description` devolvía
`[]` de queries para imperativos en inglés («Change the astronaut suit to blue»,
«Make the astronaut taller») porque `_PREDICATE_RE` no casa «change»/«make» y
`_determiner_phrases` solo funciona si el texto empieza con determinante.
Dos cambios mínimos en `roop/img_editor/img_editor_manager.py`:
1. `_visual_subject_description` (≈6348): fallback a `_leading_visual_subject`
   cuando no hay predicado ni sintagma determinante inicial. «Change the astronaut
   suit to blue» → queries `['astronaut suit']` (antes `[]`).
2. `_measured_subject_passes` (≈5889): si ningún query mide cobertura y hay
   exactamente UNA cara, mide con `_build_clipseg_person_mask` (segmentador
   genérico ya usado en `_background_mask_from_people` y protección de add). El
   sujeto sigue siendo el del prompt; solo cambia el medio de segmentación.
- `body` → **PASS visual**: «Make the astronaut taller» sale con el astronauta
  claramente más alto (proporción estirada), cara y fondo conservados.
  Log: «cuerpo medido, el resto de la foto se conserva», changed_bbox
  (0, 77, 364, 512). Salida `diagnostics/out/editor_categories_20261114/body.png`.
- `clothing` → **PASS visual (v5.1361)**: ya no se abstiene y el modelo pinta
  el traje de azul (19,16 % cambiado, bbox 31,198,355,505); cara y fondo
  conservados. Salida `diagnostics/out/editor_categories_20261114/clothing.png`.
- Regresiones verdes: selector 31/31, cámara 4/4 (FAIL=0), body regime 13/13,
  subject descriptors OK, ruff sin errores nuevos (E501/W293 preexistentes).
  `probe_editor_compound_fast_dispatch` sigue en FAIL, pero es preexistente
  (falla igual con el fix revertido; ver MEMORY.md).
- P1 sigue abierto: clothing (calidad de color), background y remove.

## P2. «la mujer debe ir descalza y sin medias» (caso del usuario)
La lógica quedó corregida en v5.1354 (zona unida objetivo+estado, verificación
por piel sobre el parche del motor), pero **no hay PASS visual**.
- Log esperado tras reiniciar:
  `objetivo medido + estado «…» → zona unida (N px)` y una sola pasada, o como
  mucho un reintento; `Verificación del estado: … piel X% → Y%` con `Y ≥ 55 %`
  o `Y ≥ X + 10`.
- Última ejecución real (v5.1353, 23:18–23:27): zona «without stockings» 38 %
  de la persona (sin pies), dos pasadas LongCat y fallback; no cumplió.
- Pendiente: repetir con v5.1354 y anotar aquí el resultado y el PNG.

## P3. Lentitud percibida («muy lento»)
Medido en el log de 23:11: arranque ComfyUI 72,8 s; carga de modelo 25 s;
`TextEncodeQwenImageEditPlus` 52 s (frío) / 5 s (caliente); LongCat 60–215 s por
pasada; cada reintento residual añade ~60 s y el fallback a ruta directa otra
pasada. Total 4–7 min cuando las comprobaciones fallan.
- Pendiente: decidir y medir un tope de reintentos por petición, y si el
  fallback directo debe ejecutarse cuando ya hubo dos pasadas. No se cambió
  nada para no aceptar resultados falsos.

## P4. Umbral de residuo de objetos nombrados (no puestos)
El umbral `TARGET_RESIDUAL_FAIL` de la política v5.1311 sigue en 40 %. La prenda
puesta ya usa la verificación por piel (más estricta), pero un objeto no puesto
(«quita el coche») acepta hasta 40 % de residuo.
- Pendiente: medir con fotos reales si 40 % es aceptable o bajarlo; documentar
  el número elegido y por qué.

## P5. `background` no tiene zona local — RESUELTO en v5.1357 (falta PASS visual)
Decisión vigente hasta v5.1356: «un pedido de escenario NO es una región local:
manda la regeneración completa, sin ancla». Esa regeneración completa era justo
lo que duplicaba a la persona y cambiaba las caras. Desde v5.1357 la ruta mide
el escenario con las personas restadas (`_measured_background_zone`) y devuelve
encima los píxeles de la persona (`_restore_measured_people`). v5.1358 corrigió
el halo de esa costura (ver «Revisión 2026-11-14 (v5.1358)»). Queda el PASS
visual, bloqueado por la GPU ocupada por otro ComfyUI.

## P6. Categoría `people`
Sin `structural_add/remove` medido cae a `scenario` (edición de marco). No hay
medición de la segunda persona ni verificación de que el número de personas
cambió. Pendiente: medir segundo sujeto y comprobar el cambio (o dejar
documentado que `people` es marco).

## P7. Categoría `features`
Usa la zona medida de cara/pelo existente, pero la verificación sigue siendo
«cambió algo» (`_validate_visual_edit`). No hay comprobación medida de pelo,
edad o piel. Pendiente: definir una comprobación medida por concepto (sin
listas de palabras) y calibrarla con fotos.

## P8. Categoría `add`
No hay verificación posterior de que el objeto añadido exista. Pendiente:
medir la consulta del objeto en la zona antes/después y marcar fallo si no
aparece.

## P9. Categoría `camera`
El POV se fuerza y el mensaje avisa de que el cumplimiento no está verificado.
Pendiente: medir el cambio de punto de vista (pose/ángulo) sobre el resultado y
decidir si se puede verificar automáticamente.

## P10. Calibración de umbrales nuevos con fotos reales
`_target_on_body ≥ 0.80`, `check_state_skin` (`≥ 0.55` o `+0.10`) y unión de
estado `≤ 0.60` de la persona se validaron con un probe sintético (9/9) y un
log real. Pendiente: calibrar con fotos reales variadas (piel oscura, ropa
clara, poses sentadas) y anotar los números.

## P11. Armonización frente a la verificación
La verificación por piel mide el parche del motor **antes** de armonizar
(v5.1353), pero la imagen entregada lleva armonización (shift medido −29 L en
un caso). Pendiente: comprobar que el tono entregado no se aleja del esperado y,
si ocurre, limitar el shift de la zona de piel.

## P12. Prompt genérico + categoría con Modo Calidad
El hint eleva un eje, así que `_prompt_measures_any_axis` devuelve True y no se
aplica la red de acabado. Con «hazlo bonito» + `clothing` el pipeline intenta
un cambio de cuerpo/ropa. Pendiente: decidir la precedencia entre categoría y
red de acabado con una medición (no se tocó).

## P13. Probes con fallo previo (ya en HEAD) — CERRADO 2026-11-14
Comprobado con el manager de HEAD:
- `probe_editor_viewpoint_dispatch.py`: el token suelto `view` cuenta como
  cámara (`is_camera_view_request('view') == True`).
- `probe_editor_selection_body_regime_20261021.py`: 2 comprobaciones en FAIL.
- `probe_editor_compound_fast_dispatch.py`: una aserción de fallo de cliente.
Pendiente: decidir si se corrige la medición o se actualiza la expectativa.

**Cerrado 2026-11-14 (v5.1358):** los tres casos están verdes en el código
actual (4 OK, 13/13 y 61 OK / 0 FAIL). No hizo falta tocar ninguna expectativa.
Aparecen dos fallos NUEVOS no listados arriba, `probe_scene_head_preserve`
(«personas no localizables en el fondo generado») y
`probe_editor_structural_zone_close` («la persona quedó dentro de la máscara
cerrada»); se comprobó revirtiendo el cambio de hoy que fallan igual antes, así
que son previos y quedan como pendiente propio de esa ruta.

## P14. Ejemplos de ayuda del desplegable
`CATEGORY_EXAMPLES` son textos de ayuda escritos a mano. No intervienen en la
lógica (el prompt se mide). Pendiente solo si se quieren mover a configuración.

## P15. Verificación visual de `only_finish`
El checkbox «🧼 Solo acabado» aplica el acabado TOP, que en los logs de 19:09
bajó la nitidez medida (115,5 → 111,6). Pendiente: PASS visual y, si no
convence, ajustar el plan del acabado en otra entrada.
