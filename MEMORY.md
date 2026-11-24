- **v5.1448 (2026-11-24) — Image Editor: el modelo ve la persona entera para repintar la cabeza (sin rastros).** El log de v5.1447 (21:54) mostraba que con la máscara parcial (solo cabeza+pelo) el modelo preservaba el pelo: 6.8% de cambio y 1.6% pegado. Causa: una máscara parcial hace que el modelo sea conservador. Ahora, para una edición de apariencia anclada a una cabeza, la generación usa la máscara de la persona entera (contexto completo) para que el modelo repinte con decisión, pero el pegado sigue restringido a la cabeza+pelo medidos (cuerpo intacto). Sin PASS visual; falta reiniciar y probar.
- **v5.1447 (2026-11-24) — Image Editor: la cabeza del agujero incluye el pelo medido (sin rastros, sin hardcode).** El usuario reportó que tras v5.1446 seguían quedando rastros de pelo. Causa: el agujero de cabeza (0,88×1,35) no cubría el pelo que sobresale (sienes/nuca); el pegado «cambio medido ∩ permitido» dejaba esa banda original. Añadido `_head_with_its_hair`: el pelo medido por CLIPSeg se une al agujero solo si pertenece a ESA cabeza (≥15 % del mechón sobre la elimpse, sin solapar otra cabeza ≥15 %) y las demás cabezas se restan. Se aplica en las dos rutas de cabeza (entrega por cabeza y cabeza-como-agujero), así que el modelo repinta todo el pelo de esa persona y el pegado ya no deja banda original. El helper es general (no depende de la palabra «calvo»). Verificación: `py_compile` OK.
- **v5.1446 (2026-11-24) — Image Editor: cabeza más amplia y verificación medida en la ruta rápida, sin hardcode.** El usuario rechazó la solución específica de «calvo» («NO QUIERO HARDCODING») y exigió un arreglo general. Revertida la zona de pelo específica de calvo y sustituida por mejoras generales: (1) la elipse de la cabeza en `_head_mask_from_faces` pasa de 0,78×1,05 a 0,88×1,35 con lóbulo posterior 0,50×0,55 (sienes y nuca ya entran en la zona de cualquier edición de apariencia anclada a la cabeza); (2) `_complete_hair_surface` amplía su alcance de `span*0.35` a `span*0.85`, así que el pelo de la cabeza se mide o se excluye completo en vez de solo el mechón a 70 px del pico; (3) la ruta rápida (la que terminaba en «cumplimiento visual no verificado») ahora ejecuta `_measure_requested_change` antes de entregar y avisa lo que no se cumplió. Verificación: `py_compile` y los probes de apariencia en verde. Sin PASS visual nuevo; los siguientes renders del usuario dirán si la banda de pelo desaparece sin hardcoding.
- **v5.1445 (2026-11-24) — Image Editor: el calvo volvía a dejar pelo y los zapatos se aceptaban sin quitarse.** El usuario reportó que «el hombre debe ser calvo» se dejó pelo y «la mujer debe ir sin zapatos» no quitó las botas. Comparado con la entrega del domingo 23/11 23:52 (`img_editor/edit_1795474328.png`, cabeza casi pelada), la única diferencia en el editor es el commit 9357685 (v5.1444, 24/11 11:07). Dos causas medidas: (1) **calvo** — `local_appearance_min_steps` subió de 20 a 28 en v5.1444; LongCat Turbo es destilado y el log del 24/11 muestra la respuesta medida sobre el sujeto cayendo de 40,5 % (primer pase) a 2,9 % y 2,0 % en las pasadas siguientes, con resto glossy en la coronilla en `edit_1795521434.png`. Además la medición de pelo residual (`_bald_hair_coverage`) devolvía None en silencio (máscara vacía, cuero cabelludo <0,5 %, sin pelo medido o sin pelo en el resultado), así que el log no decía que el chequeo no se hizo. (2) **zapatos** — la verificación midió 238686 px del objetivo antes y 28206 px después (12 %) y se declaró «Validación visual OK» con las botas puestas: `residual_failure_fraction` era 0,15 y la medición que fijó ese valor dice que una eliminación correcta deja 1,5 % y un objeto presente mide 91,8 %, luego un 12 % no puede aprobarse. Fix: pasos de apariencia local vuelven a 20 (el valor del domingo), se mantiene `local_appearance_max_work_side=1536` (guarda del stall del codificador Qwen VL en recortes 3480×2901) y el umbral de residual del objetivo baja a 0,05 (3,3× el residuo de una eliminación correcta, 18× por debajo del objeto presente). Se añade traza a los cuatro caminos en que la medición de pelo no se hace. **Sin PASS visual nuevo**: no se ha vuelto a renderizar con ComfyUI; el siguiente paso es repetir los dos prompts con la app reiniciada. Documento: `docs/IMG_EDITOR_PELO_ZAPATOS_2026-11-24.md`.
- **v5.1444 (2026-11-24) — Image Editor: el calvo ya no edita a otra persona; Turbo con más pasos y codificación acotada.** El usuario reportó que el último «el hombre debe ser calvo» tardaba y editaba a otra persona. Causa raíz medida: con 2 caras detectadas, `_measured_subject_passes` une las máscaras CLIPSeg «Man»/«el hombre» en un sujeto del 31–33 % que cubre a ambas personas, y el fallback `_head_zone_for_subject` intersectaba ese sujeto con **la unión de todas las cabezas** (`_head_mask_from_faces(image, faces)`), así que el agujero de generación y el pegado incluían la cabeza ajena. El buen camino `_appearance_head_pass` (mide qué cara ilumina la frase, regla 1,25×) solo se llamaba cuando no había máscara de sujeto y nunca se usaba aquí. Fix general (sin listas de palabras ni géneros): `_head_zone_for_subject` elige UNA cabeza con `_appearance_head_pass` y, si la frase no separa, con `_leading_face` sobre la máscara medida; desempate por `det_score` cuando la máscara es plana; recorta las demás cabezas antes de enviar la zona al sampler y al pegado. Acepta `raw_prompt`/`translated_prompt` desde los dos call sites. Fase 2 (mantiene **LongCat Turbo**, no cambia de motor): pasos de apariencia local y tope del lado largo de la imagen de trabajo pasan a `config/editor_selection.json` (`local_appearance_min_steps=28`, `local_appearance_max_work_side=1536`, con `_doc_local_appearance_quality`). El stall de 5 min en Qwen VL del 24/11 venía de un recorte 3480×2901; el cliente ya muestrea a ~1 MP y `_integrate_local_generation` reescala al recorte original con compensación de nitidez, así que acotar antes no cambia la zona. Se aplicó en las DOS rutas de generación (había un segundo bloque con `appearance_sample` fijo en 20 que el primer parche no tocaba).
   - Verificación (sin reiniciar la UI, sin ComfyUI real): `diagnostics/probe_editor_head_zone_single_person_20261124.py` PASS — reproduce la unión antigua con 38 396 px de la otra cara dentro y la nueva deja 0 px, conserva 33 418 px de la cara elegida; `probe_editor_local_appearance_quality_20261124.py` PASS — config 28/1536, recorte 3480×2901 → 1536×1280, cobertura de máscara 14,1449 %→14,1450 %, pegado con generado pequeño 40 000 px dentro y 0 fuera; `probe_editor_local_appearance_downscale_path_20261124.py` PASS — fuerza tope 200 en la ruta real, primer pase 200 px, 28 pasos en los 3 generate; `probe_editor_bald_backhair_20261016.py` PASS con `steps=28` y log «Cabeza del sujeto medida … (frase medida); las demás cabezas quedan fuera». `py_compile`, JSON válido y `git diff --check` PASS; Ruff sin errores nuevos (quedan 3 F821 `_LONGCAT_FULL_VERSION` preexistentes en HEAD). `probe_editor_versatility_integrity_20261105.py` sigue con 2 FAIL de `RemovalContracts`/LaMa, preexistentes y ajenos a la función tocada. **Sin PASS visual con las fotos reales del usuario**: falta reiniciar la app y repetir el calvo con dos personas; la lentitud y la obediencia solo están medidas con probes.
- **v5.1443 (2026-11-24) — Image Editor: corrige JSON inválido y tono entregado no medido.** `config/quality_mode.json` usaba booleans estilo Python (`True`) que rompían `json.loads`, dejando el acabado ultra-realista sin cargar; se normalizan a `true`. En `selected_region_edit.py`, `_skin_pixels(harmonized)` recibía una imagen PIL en lugar de array, provocando «Tono entregado no medido: Image object is not subscriptable»; se convierte a `np.asarray`. Versionado: v5.1443 en `run.py`, `roop/core.py`, `AGENTS.md`, `spec/00-PRINCIPIOS.md` y esta entrada.
- **v5.1442 (2026-11-23) — Image Editor: corrige error de configuración `editor_ui.automatic_appearance_engine`.** `config/generation_prompt.json` tenía `"automatic_appearance_engine": "longcat_full"` (v5.1441) pero `_automatic_appearance_engine()` en `roop/img_editor/img_editor_manager.py` solo acepta `"kontext"` o `"longcat"`, por lo que la ruta de apariencia pura lanzaba `ValueError: editor_ui.automatic_appearance_engine debe ser kontext o longcat`. El valor pasa a `"longcat"` (decisión del usuario: **LongCat, no Kontext**); `get_editor_ui_config()["automatic_appearance_engine"]` resuelve a `"longcat"` y pasa el filtro `in ("kontext", "longcat")`. Versionado: v5.1442 en `run.py`, `roop/core.py`, `AGENTS.md`, `spec/00-PRINCIPIOS.md` y esta entrada.
- **v5.1439 (2026-11-23) — Mejorar fotos y Image Editor: presets realineados, acabado ultra realista configurado.** Presets de mejorar fotos retuneados (Retrato: denoise 16/sharpen 12, Realista: denoise 14/sharpen 14/upscale 1x, Ultra realista: denoise 16/sharpen 26/upscale 2x/face_model GFPGAN, Nocturna: denoise 48/sharpen 20, Paisaje: denoise 30/sharpen 26). Image Editor now aplica realismo post-edit desde `config/quality_mode.json` con sharpen ajustado en rangos [1.05, 1.10], detail_boost local ligero y pulido generativo moderado cuando mag >= 0.48. Presets alineados con topes de pila `_STACK_CAPS`. Config `quality_mode.json` añade sección `realism` con `preserve_skin_texture`, `sharpen_floor/ceiling`, `detail_boost_blend` rango, y `enable_polish_generative`. Probes: `probe_enhance_realism_20261123.py` PASS técnico; pending PASS visual con lote real del usuario. Código: `top_tier_finish.py` integra el nuevo acabado sin romper ruta clásica. Versionado: v5.1439 en `run.py`, `roop/core.py`, `AGENTS.md`.

- **v5.1441 (2026-11-23) — Image Editor: exactitud corporal con LongCat Full.** Kontext dev Q4_K_M tardaba ~5 min/paso en RTX 3060 Ti 8 GB (~60–100 min por edición); LongCat Turbo (8 pasos) era rápido pero menos exacto en cambios corporales complejos. Se cambió `editor_ui.automatic_appearance_engine` de "longcat" a "longcat_full" en `config/generation_prompt.json` y el router de body transforms (uncover/cover/change) ahora usa **LongCat Full** (35 pasos, cfg 2.5, Qwen-VL con visión) en lugar de Turbo/Kontext. LongCat Full sigue siendo mucho más rápido que Kontext en 8 GB y es más exacto en edición corporal, ropa y pose. Modelo en disco: `LongCat-Image-Edit-Q4_K_S.gguf`. Código: `_route_edit_engine` en `img_editor_manager.py` usa `_LONGCAT_FULL_VERSION`. Versión: v5.1441 en `run.py`, `roop/core.py`, `AGENTS.md`, `spec/00-PRINCIPIOS.md`. (Entrada v5.1440 de improve-fotos conservada más arriba.)
  - Fix aplicado en `ui/tabs/enhance/logic.py`: `apply_upscale_onnx(..., resample_mix=0.25)` mezcla 75 % ESRGAN + 25 % Lanczos y devuelve al modelo la luz/color a gran escala del escalado limpio (sin velo ni tinte; ojo: `bgr`/`out_rgb` van en órdenes distintos y sin convertir salía la piel azul); `apply_recover_faces_onnx(..., tone_match=True)` conserva la luz del original dentro de la cara y solo toma del modelo la estructura (en la muestra no había blanqueo medible: media 133.0 → 132.9, p99.5 230 → 230); sharpen × 0.6 con upscale; `grain_size` por defecto 2. Las 15 recetas se pasaron completas con su upscale real: ninguna revierte pasos, correlación ≥ 0.92, recortes +0.0 a +2.8 pp (Paisaje es la más agresiva porque sube luces y escala 2x).
  - Resultado de la pila completa a 2x (A anterior → B nueva): correlación de textura 0.73 → 0.79, banda 1.96 → 1.71, recorte 16.20 % → 14.98 % (original 14.79 %), sin tinte. A 1x: 0.93 → 0.95. `py_compile` y Ruff PASS; `probe_enhance_realism_20261123.py` sigue PASS con A 0.95 / B 0.90 / C 0.99 / D 0.98 / E 0.97. Hojas: `diagnostics/out/enhance_audit_20261123/{skin_stages,face_orders,upscale_mix,upscale_tone,presets_sheet}.png`. **Sin PASS visual con las fotos reales del usuario** (pendiente de reiniciar la app y repetir un lote).
- **v5.1437 (2026-11-23) — Mejorar fotos: las mejoras salían «muy muñecas» (piel de cera).** El usuario reportó que sus últimos lotes con los presets salían poco realistas. Los logs vivos (`logs/console_run_20261123_0103.log`, líneas `[NATIVE-DROP] enhance:` de `console_diag_drop_20261123_0052/0100.log`) muestran la receta con la que se abrió la pestaña: **denoise 65 · sharpen 40 · caras GFPGAN 80 · upscale 2x · grano 20/tamaño 10**, y las salidas de `enhance/*_dn-sh-rf-up2x-db-co.png` (17/11 23:50) llevan además `db`+`co`, compatibles con esa base **más un preset acumulado** (v5.1417 suma denoise/sharpen/grano). Causas medidas en `ui/tabs/enhance/logic.py`: (1) el denoise era un bilateral único (d=14, sigma=36 a 65) que borraba el poro, el pelo fino y el humo, y dejaba la piel lisa; (2) el unsharp llegaba a amount 1.5 sin umbral ni recorte de halo, así que amplificaba el ruido de la piel aplanada y creaba bordes «crujientes»; (3) la recuperación de caras repintaba la piel con la textura genérica del modelo (máscara de porcelana); (4) el grano por defecto era de tamaño 10 (manchas, no grano); (5) la protección de piel de `apply_vibrance_cv2` comparaba `h >= 330` cuando OpenCV usa H 0-180, así que la piel recibía el 100% de la saturación.
  - Fix: denoise en LAB (croma suavizado fuerte + luminancia moderada con reinyección del 55→32% del detalle que el filtro se lleva), sharpen con umbral suave (descarta el microdetalle de zonas planas) y recorte de halo (amount máx. 1.1), reinyección del high-pass original dentro de la máscara facial (60%) para devolver el poro, grano ponderado por medios tonos, rango de piel corregido y los 15 presets retocados (caras: recover ≤ 0.5, denoise ≤ 60, sharpen ≤ 35, grain_size 2-4; Ultra realista pasa de CodeFormer a GFPGAN y de 30/45/75 a 22/22/40). Defaults de `config/enhance/enhance.json` y `state.py`: 35/20/0.40/6/2 (antes 65/40/0.80/20/10). `events._STACK_CAPS` satura la acumulación (denoise 60, sharpen 45, deband 60, grano 30, recover 0.8) y el aviso añade «tope de pila en …»; los sliders manuales siguen llegando a 100.
  - Verificación: `diagnostics/probe_enhance_realism_20261123.py` **PASS** — 15/15 presets válidos y sin recetas de cera, hojas `diagnostics/out/enhance_realism_20261123/{skin_sheet,face_sheet,pore_zoom}.png`. Revisión visual: A (65/40/80/20) y B (95/85/100/30, la pila probable) dejan la mejilla lisa con moteado y halo en el cuello; C (Ultra realista nueva) y D (defaults) conservan poro y no tienen halo; E (pila topada 60/45) ya no hace halo y mantiene algo de textura. Correlación de la banda de poro con el original: A 0.95 / B 0.90 → C 0.99 / D 0.98. `py_compile` y Ruff de los archivos tocados PASS; `build_enhance_ui()` construye 48 componentes con defaults 35/20/40/6/2. **Sin PASS visual con las fotos reales del usuario**: hace falta reiniciar la app (los defaults se leen al construir la pestaña) y repetir un lote.
- **v5.1436 (2026-11-23) — Drop nativo: el evento reenviado perdía el `dataTransfer` y no cargaba ninguna foto.** El usuario reportó que al arrastrar fotos sobre la franja de «Mejorar fotos» no se cargaba nada. Causa raíz medida con `diagnostics/probe_native_drop_serialization_20261123.py`: `roop/core.py` reenviaba al receptor oculto un `new Event('drop')` con `Object.defineProperty(evt, 'dataTransfer', …)`, y esa propiedad **no es enumerable** por defecto; pywebview serializa el evento con `for (var key in obj)` (venv/Lib/site-packages/webview/js/api.js), así que Python recibía el evento **sin `dataTransfer`** y no había rutas que encolar. El probe captura el JSON exacto que va al puente: variante A (código anterior) → `has_data_transfer=false`; variante B (misma propiedad enumerable) y C (`new DragEvent` con el `DataTransfer` real) → llegan los nombres de archivo. Fix: se reenvía un `DragEvent('drop', {dataTransfer: dt})` con el `DataTransfer` original (File reales del disco, que es lo que WebView2 exige para `postMessageWithAdditionalObjects`) y queda un fallback con `enumerable: true`. Afecta por igual a «Mejorar fotos» y a las zonas de origen/destino de FaceSwap. Además, el handler Python distingue ahora «evento sin dataTransfer» de «drop sin rutas nativas» para que la próxima vez quede traza. Verificado en la app viva: `[NATIVE-DROP] enhance: 66 ruta(s) recibidas` → `cargando 66 foto(s) por ruta` → `aceptadas 66/66` con «66 foto(s) en cola» y vista previa de la primera.
  - **Cuadro vacío retirado (petición del usuario, mismo turno).** Aun funcionando el arrastre, el usuario vio «no carga nada» porque el cuadro de debajo de «Importar» seguía vacío: el drop nativo pasa rutas y no sube archivos, así que `gr.Files` nunca pinta nada. Se elimina visualmente ese cuadro: la zona de arrastre pasa a ser la línea de cola, `with gr.Row(elem_id="enhance_input_files")`, cuyo texto invita a arrastrar y muestra «N foto(s) en cola»; `gr.Files` queda `visible=False` (sin etiqueta ni elem_id) para no romper los eventos que lo usan como contrato. Se retira el CSS de la franja de 64 px y queda solo el resaltado de la fila. Comprobado en el `/config` servido: `elem_id="enhance_input_files"` aparecía como `row` visible y el `file` de Mejorar fotos quedaba `visible=False`.
  - **El `elem_id` de una fila de layout NO llega al DOM (causa del «no coge nada»).** Con la app viva y CDP (`AUTOAUTO_DIAG_CDP_PORT`, `diagnostics/probe_native_drop_cdp_20261123.py`), `document.getElementById('enhance_input_files')` devolvía `null` aunque el `/config` declaraba ese `elem_id` en el `row`: Gradio no lo renderiza en componentes de layout, así que la zona de arrastre no existía y ningún drop se interceptaba (el log no registraba ni un evento). El id se movió al `gr.Markdown` de la línea de cola, que sí lo honra. Además, Gradio solo monta la pestaña activa: hasta abrir «Mejorar fotos» no hay zona en el DOM.
  - **Zona final: toda la pestaña.** Para no exigir acertar en una franja de 48 px, `zoneFor` acepta `#enhance_input_files`, `#srcfiles`, `#destfiles` y `.img-enhance-container` (contenedor real de la pestaña, comprobado en el DOM), y la cola se decide por id (`srcfiles`→origen, `destfiles`→destino, resto→enhance). Verificado con drops reales por CDP (archivos del disco, `Input.dispatchDragEvent`): soltando en el panel `1/1` y `3/3` en cola con vista previa; soltando en el contenedor de la pestaña `1/1`; trazas `enhance: N ruta(s) recibidas` → `aceptadas N/N`. `py_compile`, Ruff y `node --check` del JS extraído PASS. El hook CDP queda apagado salvo que se exporte `AUTOAUTO_DIAG_CDP_PORT`. **Sin PASS visual del usuario**: falta que lo confirme con su ratón sobre la pestaña abierta.
- **v5.1435 (2026-11-22) — Mejorar fotos: zona de arrastre compacta.** El cuadro de archivos (206 px) pasa a una franja de 64 px (70 px de bloque) con borde discontinuo, icono pequeño y texto en línea; se mantiene el mismo `gr.Files` como zona de drop nativo y fallback de clic, con `elem_id` intacto. Medido con Selenium: 206→70 px, contenido centrado sin recorte. Sin arrastre físico nuevo sobre la franja; el enlace de zonas no cambia.
- **v5.1434 (2026-11-22) — Drop de Mejorar fotos: aviso visible y traza de cola.** El usuario confirma que el arrastre ya funciona: el drop real de 66 fotos en la app v5.1433 quedó en 66/66 en cola con vista previa (`logs/verify_drop_v2_20261122.log`), y en Chrome real (Selenium) se comprobó que el timer aplica resumen, desplegable de foto y antes/después, además del toast `gr.Info`. El cuadro de carga (`gr.Files`) sigue vacío a propósito porque no se suben archivos; rellenarlo con rutas crudas deja el evento en «processing» y bloquea el resto de salidas, así que se descarta. Se añaden trazas de rutas recibidas/aceptadas y del estado de la vista previa, y aviso `gr.Warning` si el arrastre no trae formatos compatibles. Sin PASS visual de calidad (no aplica al drop).
- **v5.1433 (2026-11-22) — Drag&drop de Mejorar fotos y FaceSwap: rutas nativas, sin subir archivos.** El drop de v5.1428 inyectaba un listener que leía cada archivo con `arrayBuffer()` y lo subía en serie por `fetch('/upload_native')`, anulando el drop de Gradio; con cientos de fotos bloqueaba la UI y en FaceSwap además nadie consumía las rutas. Se sustituye por la vía nativa de pywebview: un listener `drop` por zona (Mejorar fotos, origen y destino de FaceSwap) sobre receptores ocultos, interceptación en fase de captura solo dentro de esas zonas (Generate, Animate e Image Editor conservan el drop de Gradio) y colas por destino (`NATIVE_DROP_QUEUES`). Enhance las vacía con su timer; FaceSwap añade timers para origen y destino que llaman a la carga normal por ruta. Se elimina `/upload_native` y el `js_api` de drop. `py_compile` y Ruff de los archivos tocados sin errores nuevos. Sin PASS visual.
- **v5.1428 (2026-11-22) — Drop nativo (pywebview) para lotes grandes.** Captura de rutas completas sin subir File objects por el renderer; Enhance integra cola por rutas mediante timer de polling. Sin PASS visual aún.
- **v5.1427 (2026-11-22) — Consolidación de cambios pendientes.** Se cierran los cambios sin commitear: FaceSwap (tracking/reacquire parcial y append de bloques en origen/destino), Mejorar fotos (importador y firmas), capturer/face_util (clamping de imágenes gigantes y PIL sin límite de píxeles) y limpieza de lint (E402/F401) en los módulos tocados. Ruff y py_compile verdes. Sin PASS visual nuevo.
- **v5.1426 (2026-11-22) — Mejorar fotos: límite drag&drop + selector nativo.** Drag&drop limitado a 30 archivos con aviso; botón «Buscar fotos en el PC» usa diálogo nativo Windows y pasa rutas sin saturar WebView2. Sin PASS visual.
- **v5.1425 (2026-11-22) — Mejorar fotos: heartbeat en lote.** Generador rinde cada 2 s aunque no haya progreso, para evitar bloqueo de UI en pasos largos (recuperación CodeFormer ~16 s). Sin PASS visual.
- **v5.1424 (2026-11-22) — Mejorar fotos: preview ligero y recuperación responsive.** La vista previa ya no ejecuta recuperación de caras (más rápida) y la recuperación por lotes reporta progreso por cara/cancelable. Sin PASS visual.
- **v5.1423 (2026-11-22) — Mejorar fotos: progreso por foto.** La barra muestra «Foto X/Y» y la fase actual, evitando la sensación de cuelgue en lotes grandes (Ultra realista puede tardar ~1 min/foto). Sin PASS visual.
- **v5.1422 (2026-11-22) — Mejorar fotos: alivio de cuelgue en lotes.** Se limitan las actualizaciones de la barra de progreso a cambios reales de porcentaje/fase para no saturar la UI con 70+ fotos. Sin PASS visual.
- **v5.1421 (2026-11-22) — Mejorar fotos: carpeta de destino persistente.** La carpeta de salida elegida se guarda entre sesiones (config/enhance/last_output_dir.txt) y se restaura al abrir. Sin PASS visual.
- **v5.1420 (2026-11-22) — Mejorar fotos: corrección de acumulación de presets.** La pila ahora conserva y muestra todos los presets añadidos (no solo el último) y se mantiene al cambiar de foto. Sin PASS visual.
- **v5.1419 (2026-11-22) — Mejorar fotos: pila visual de presets.** Los presets acumulados se muestran como iconos justo debajo de la vista previa (antes/después), con botón para limpiar la pila. Sin PASS visual.
- **v5.1418 (2026-11-22) — Mejorar fotos: barra de progreso realista.** El progreso ahora se calcula por imagen y por paso (denoise, sharpen, etc.) en lugar de saltos por foto; la UI muestra porcentaje continuo. Sin PASS visual.
- **v5.1417 (2026-11-22) — Mejorar fotos: presets acumulativos.** Los botones de preset ya no sustituyen la receta: se acumulan sobre los ajustes actuales (denoise/sharpen/deband/grano suman; exposición/sombras/luces/vibrance se multiplican; upscale y recuperación de caras toman el máximo; motor de caras y temperatura del último preset). La UI muestra una «Pila activa» y un botón «Limpiar pila»; el preview se re-renderiza con la receta acumulada. Funciona también con «Seleccionar todas». Sin PASS visual.
- **v5.1416 (2026-11-22) — Prueba definitiva: el bloqueo es por el volumen de archivos, no por «Mejorar fotos».** El usuario dio la carpeta real `D:\PROJECTS\uee\tlk\IA\newIMGA\FALTAeXALAR`: **441 imágenes, 2,04 GB**, 33 archivos >20 MB (máximo 46 MB). Con CDP se adjuntaron esos mismos archivos al input de **FaceSwap**: la conexión CDP se cortó y la ventana se cerró **igual** que en «Mejorar fotos». Con 400 imágenes de ~300 KB (117 MB) ambas pestañas funcionan sin errores y con el navegador respondiendo. Conclusión: WebView2 no puede gestionar un drag&drop de 2 GB en ninguna pestaña; no es un defecto del código de mejora. Se retiran las marcas de diagnóstico (script del título y puerto CDP 9222). Para ese lote hace falta carga por ruta (selector nativo) o dividirlo en tandas; ambas opciones fueron rechazadas por el usuario. Sin PASS visual.

- **v5.1415 (2026-11-22) — Drag&drop de 400 fotos confirmado en navegador real; preview de la primera restaurado.** Tras el `storage_path` único (v5.1412) se habilitó CDP (`webview.settings['REMOTE_DEBUGGING_PORT']=9222`, v5.1414) y se inspeccionó la página en vivo: adjuntar 400 archivos al input real de «Mejorar fotos» dejó el renderer respondiendo en 0,0 s durante 25 s y la cola mostró «400 foto(s) en cola», sin cuelgue. El usuario vio que las fotos cargaban pero no aparecía el preview: era la omisión deliberada de `make_pair_on_import` para lotes >60. v5.1415 renderiza **solo la primera foto** (antes/después) y mantiene `gr.skip()` en los 13 sliders. Probe de 400 fotos con subida real PASS, Ruff, diff y build PyInstaller PASS. Nota: v5.1413 añadió marcas de diagnóstico en `document.title` (el título nativo no las refleja) y v5.1414 dejó el puerto CDP 9222 activo; retirar ambos cuando no hagan falta.

- **v5.1412 (2026-11-22) — CAUSA RAÍZ del bloqueo al subir fotos: WebView2 reutilizaba un navegador viejo atascado.** Con la app abierta y recién colgada, el log de la sesión activa terminaba en el arranque (sin errores) y **todos** los procesos `msedgewebview2` eran de las 00:20–00:21 aunque la app actual era de las 01:21; dos renderers viejos llevaban 1123 s y 607 s de CPU girando. `roop/core.py` llamaba `webview.start(debug=False, func=None)` sin `storage_path`, así que todas las ventanas compartían el mismo user-data-dir y heredaban el proceso atascado. Fix: `storage_path=os.path.join(tempfile.gettempdir(), f"autoauto_webview_{os.getpid()}")` → proceso de navegador nuevo en cada lanzamiento. Se cerraron los WebView2 viejos y se reinició: app **v5.1412** (PID 7160) con WebView2 fresco (01:24:15) y sin renderers girando. Las versiones anteriores mostraban el mismo síntoma por esta causa; los probes de Gradio pasaban porque el backend nunca tuvo la culpa. `py_compile`, Ruff, diff y build PyInstaller PASS. Falta el PASS visual con el drag&drop real.

- **v5.1411 (2026-11-22) — Mejorar fotos: solo drag&drop, sin botón nativo ni campo de ruta.** El usuario **rechazó explícitamente** el botón «Buscar fotos en el PC» y el campo «Ruta directa»: exige que funcione drag&drop como en el resto de la app. Se eliminan ambos componentes y todo su código (`_DIRECT_PATHS`, `_expand_direct_paths`, `import json`, handlers de navegador). Optimización aplicada al propio drag&drop: `refresh_queue` guarda las rutas resueltas en `_LAST_QUEUE_PATHS` (en el servidor) y el `.then(make_pair_on_import)` solo recibe la foto seleccionada más los sliders, **no los cientos de FileData del upload**; se descartó `gr.State` porque no se expone en la API. Se mantienen la omisión de preview en lotes grandes y `gr.skip()`. Verificación: probe de 400 fotos con **subida real** PASS (13,5 s, cola 400, navegación, lote 400/400), probes select-all y UI PASS; `py_compile`, Ruff, diff y build PyInstaller PASS. **Sin PASS visual**: si reaparece el bloqueo, el siguiente paso es una traza del navegador, no otro parche a ciegas.

- **v5.1410 (2026-11-22) — Mejorar fotos: botón nativo «Buscar fotos en el PC» para lotes grandes.** Tras el quinto bloqueo con drag&drop de 400 fotos, el usuario aclaró que en FaceSwap **no son las mismas fotos**: allí carga otras imágenes, así que el éxito de FaceSwap no demostraba que WebView2 aguante estos originales. Se implementa la vía acordado en v5.1409: botón «Buscar fotos en el PC» con `tkinter.filedialog.askopenfilenames` que devuelve rutas; `_expand_direct_paths` acepta ahora una lista JSON de rutas (además de carpeta/archivo), el handler escribe esa lista en el campo «Ruta directa» y dispara la cola normal. Los 400 archivos **no pasan por el navegador**. El drag&drop sigue disponible para lotes pequeños. Verificación: probe de 400 fotos PASS (cola desde JSON de 400 rutas, preview omitido, navegación y lote 400/400); `py_compile`, Ruff, diff y build PyInstaller PASS. Sin PASS visual con las fotos reales del usuario.

- **v5.1409 (2026-11-22) — Mejorar fotos: optimización del renderer tras el cuarto bloqueo con 400 fotos.** El bloqueo volvió en v5.1408: servidor vivo, log limpio y **ninguna foto nueva en el cache de Gradio** → se cuelga en WebView2 antes de enviar. Tras tres intentos (omitir preview, alinear con `.upload()` de FaceSwap, etc.) el usuario eligió **mantener drag&drop y optimizar**, no pasar al selector nativo. Cambios: `make_pair_on_import` devuelve `gr.skip()` en lugar de `None`/`gr.update()` para las dos imágenes y los 13 sliders (el navegador no repinta esos componentes tras la subida), y CSS `content-visibility: auto` + `contain-intrinsic-size: 28px` en las filas del importador (WebView2 no maqueta las cientos de filas fuera de pantalla). Verificación: probe de 400 fotos PASS (adaptado: `gr.skip()` llega por API como `{'__type__': 'update'}`), probe select-all PASS, Ruff, `py_compile`, diff y build PyInstaller PASS. **Sin PASS visual**: si se cuelga otra vez, el siguiente paso acordado es el botón nativo «Buscar fotos en el PC» que carga por ruta sin subida web. Había dos ventanas apuntando al mismo servidor; se cerraron para dejar una sola.

- **v5.1408 (2026-11-22) — FaceSwap: subida por bloques acumulativa en origen y destino.** El usuario pidió subir un bloque de imágenes en «Caras Origen» y en «Caras de Destino» y después otro, sumándose y procesándose las caras, sin romper el flujo. Antes: `on_src_changed` limpiaba y reextraía todo el banco en cada subida (perdiendo la selección); `_load_target_files` limpiaba la cola de destino, así que el segundo bloque reemplazaba al primero. Cambios en `ui/tabs/faceswap/events.py`: registros `_LOADED_SRC_PATHS` / `_LOADED_TARGET_PATHS` con dedup por ruta normalizada; origen extrae solo archivos nuevos y **añade sus caras al banco** `INPUT_FACESETS` (disponibles para el swap); destino añade solo rutas nuevas a `list_files_process` conservando selección, caras elegidas y referencias; «Buscar en el PC» y la ruta directa también pasan a modo añadir; los botones de limpiar vacían los registros.
  - Verificación: `diagnostics/probe_faceswap_append_blocks_20261122.py` PASS — origen 1→2 caras sin duplicar, destino 1→2 archivos sin duplicar, limpieza y recarga correctas. `probe_faceswap_upload_event.py`, `probe_faceswap_target_load.py` y `probe_faceswap_clear_ui.py` PASS sin regresión; Ruff, `py_compile`, diff y build PyInstaller PASS. Falta el PASS visual con las fotos reales del usuario.

- **v5.1407 (2026-11-22) — Mejorar fotos: drag&drop de lotes grandes como FaceSwap.** Segundo bloqueo al subir 400 fotos por drag&drop en la pestaña «Archivos»: el servidor seguía vivo (PID 11272) y en el cache de Gradio no aparecía ninguna foto nueva, así que se colgaba en el navegador **antes de enviarlas**. El usuario confirmó que en FaceSwap carga todas las que quiere con drag&drop. Diferencia encontrada en el cableado: FaceSwap usa `gr.Files.upload()` (dispara una vez al terminar) y «Mejorar fotos» usaba `.change()` (puede dispararse por tandas con drag&drop masivo). Cambio: `input_files` e `input_folder` usan `.upload(..., show_progress="full")` con el mismo `.then(make_pair_on_import)`. Verificado que los triggers del importador son solo `upload`, como FaceSwap; probes de 400 fotos y select-all PASS; Ruff, `py_compile`, diff y build PyInstaller PASS.
  - Nota de limpieza: con autorización se cerraron WebView2 huérfanos del 20/11 (dos giraban ~14 000 s de CPU). Al cerrar la ventana «Seleccionar AutoAuto» se apagó la app completa. Falta el PASS visual con el drag&drop real del usuario.

- **v5.1406 (2026-11-22) — Mejorar fotos: importador igual que FaceSwap.** El usuario señaló que en FaceSwap puede cargar todas las fotos que quiera. Comprobado en código: FaceSwap usa el mismo `gr.Files(file_count="multiple")` que «Mejorar fotos»; el bloqueo venía del preview automático y del desplegable de 400 entradas (corregidos en v5.1405). Prueba de **subida real de 400 archivos**: cola 400 en 14,5 s y preview omitido, sin error; la «Ruta directa» nunca fue obligatoria. A petición del usuario se retira el CSS que ocultaba la lista de nombres y se pone la altura del importador a **200**, igual que FaceSwap; el campo «Ruta directa» queda como opción extra. `gr.Files` ya no filtra por `file_types` (el filtro de imágenes sigue en `_queue_paths` con el registro de PIL).
  - Verificación: `probe_enhance_large_batch_20261121.py` PASS (400 fotos, navegación más allá del tope y lote 400/400); Ruff, `py_compile`, `git -c core.whitespace=cr-at-eol diff --check` y build PyInstaller PASS. La UI caída sigue sin reiniciarse por decisión del usuario.

- **v5.1405 (2026-11-22) — Mejorar fotos: lotes grandes sin congelar la UI.** El usuario reportó que al subir más de 400 fotos la UI de Gradio se quedó en blanco. Diagnóstico: el proceso ya no escuchaba en 7861; en servidor aislado el backend aguanta 400 fotos (cola 17 s, vista previa 21 s), así que el fallo está en el frontend de pywebview por volumen de archivos y render automático al importar. Cambios: umbrales `large_batch` en `config/enhance/enhance.json` (`auto_preview_max_files=60`, `max_visible_choices=100`); `refresh_queue` acepta `direct_text` y expande una carpeta local sin subir archivos (`_DIRECT_PATHS`); con colas grandes `make_pair_on_import` omite el preview y avisa; el desplegable se limita y se oculta la tabla de nombres del importador por CSS.
  - **Intento fallido anotado:** limitar `choices` sin `allow_custom_value` hacía que Gradio rechazara la foto 151/400 («value is not in the list of choices»), rompiendo Anterior/Siguiente. Corregido con `allow_custom_value=True` en `preview_photo`.
  - Verificación: `diagnostics/probe_enhance_large_batch_20261121.py` PASS — 400 fotos por ruta directa, 100 opciones visibles, preview omitido, foto 151 accesible y lote 400/400 en la carpeta elegida. `probe_enhance_select_all_20261121.py` y `probe_enhance_ui_20261117.py` PASS sin regresión (actualizados para el nuevo parámetro `direct_text`). `py_compile`, Ruff, `git -c core.whitespace=cr-at-eol diff --check` y build PyInstaller PASS. La UI caída **no se reinició** (decisión del usuario); no se declara PASS visual en pywebview con el lote real.

- **v5.1404 (2026-11-21) — Mejorar fotos: carpeta de destino elegible.** El usuario pidió poder seleccionar otra carpeta de destino. El campo «Carpeta de destino» ahora es editable y hay un botón «Examinar…» que abre el diálogo nativo de Windows (mismo patrón que el selector de FaceSwap); también se puede pegar una ruta. `events.py` crea la carpeta si no existe, expande variables de entorno y `~`, y cae a `enhance/` con aviso si la ruta es inválida; «Abrir carpeta» y `do_batch` usan la ruta elegida (parámetro `output_path`, opcional para no romper probes antiguos). `logic.py` intacto: `process_batch` ya recibía `output_dir`.
  - Verificación: `diagnostics/probe_enhance_select_all_20261121.py` PASS — contrato 45/45 y 64 eventos, botón presente en la página servida, lote en carpeta nueva (creada y usada) y ruta inválida con fallback avisado; se mantienen las comprobaciones OFF/ON, 15 presets, modo carpeta y sRGB/254. `py_compile`, Ruff, `git -c core.whitespace=cr-at-eol diff --check` y build PyInstaller PASS. La UI activa no se interrumpió; el diálogo nativo queda pendiente de comprobación visual al abrir la versión nueva.

- **v5.1403 (2026-11-21) — Mejorar fotos: «Seleccionar todas», presets revisados, modo carpeta y exportación sRGB/254 PPP.** El usuario pidió una casilla junto a la subida de fotos: activa, un mismo preset/ajustes se aplica a TODAS; desactivada, se conserva el ajuste por foto. Las decisiones aprobadas fueron que «Seleccionar todas» gana sobre el modo automático, que sRGB/254 PPP sea solo de «Mejorar fotos» y que pulsar un preset actualice únicamente el preview «DESPUÉS» (el «ANTES» sigue siendo la foto original).
  - Implementación: `ui/tabs/enhance/ui.py` añade `select_all` al contrato; `events.py` propaga la receta a todas las fotos al activar la casilla o aplicar un preset, `do_batch` pasa `settings_by_path=None` y desactiva autopilot cuando está activa, y los botones de preset solo repintan `preview_after`. El modo carpeta descarta no-imágenes con `Image.registered_extensions()`. `logic.py` guarda PNG/JPG con ICC sRGB y `dpi=(254,254)` leídos de `config/enhance/enhance.json` (`export.dpi`, `export.color_profile`), sin ficheros ICC externos ni red.
  - Revisión pedida: 15/15 presets válidos con receta e icono, «Blanco y negro» con `vibrance=0` correcto; el modo carpeta enumera y deduplica el payload de directorio. Pendientes P1–P12/P14–P15 siguen abiertos y P13 solo técnico; no se declara ningún PASS visual nuevo.
  - Verificación: `diagnostics/probe_enhance_select_all_20261121.py` PASS — contrato 44/44 y 63 eventos, lote OFF con dos recetas distintas frente a ON con dos salidas byte-idénticas y resumen marcado, PNG/JPG con 254 PPP y perfil `sRGB built-in`. Probe anterior `probe_enhance_ui_20261117.py` PASS sin regresión; la página servida muestra la etiqueta «Seleccionar todas». `py_compile`, Ruff de los Python tocados, `git -c core.whitespace=cr-at-eol diff --check` y build PyInstaller (`AutoAuto.spec`, `dist/`) PASS. La UI activa (PID 4720, v5.1402) no se interrumpió: pruebas aisladas en 7862/7863/7864. Queda el juicio visual del usuario al abrir la versión nueva.

- **v5.1402 (2026-11-20) — FaceSwap: bloqueo de readquisición a otra persona y retención de logs 30 días.** El usuario eligió contrastar una identidad fija en saltos dudosos y conservar logs 30 días. En `ProcessMgr._unverified_reacquire_jump`, si no hay corte visual reciente y el salto supera `max(reacquire_far_jump_distance_floor, reacquire_suspect_jump_distance_scale × extensión)`, se exige a la identidad fijada por el primer swap real una similitud ≥ `reacquire_suspect_jump_min_bank_similarity` (config: 120 px, 1,1 y 0,10). Antes del primer swap, se usa la primera referencia UI. El gate anterior para salto lejano ≥1,3 y score de detector ≥0,25 permanece. Fijar la cara intercambiada evita que la max-sim del banco de perfiles (0,370 al hombre en la reproducción), el orden del banco o el embedding adaptable (0,33→0,99) conviertan a otra persona en destino.
  - **Reproducción FAIL→PASS:** `diagnostics/probe_faceswap_recent_reacquire_20261120.py` usa detecciones reales y geometría registrada del lote. Antes del arreglo, aceptaba el hombre f1699 y `_apply_reacquired_face` lo fijaba; después rechaza ambas rutas. Conserva los retornos reales de la mujer f1329 (sim fija 0,143) y f1729 (0,265), y la cara ocluida del segundo vídeo f1661 (0,066 pero salto cercano). El banco UI original se perdió: el probe reconstruye f0 y añade una referencia de perfil f1631 para comprobar incluso el caso de max-sim engañosa. Una segunda prueba invierte el banco: con el perfil en primer lugar, el gate anterior volvió a aceptar al hombre (FAIL); al congelar el embedding de la primera cara intercambiada, el probe vuelve a PASS en ambos órdenes.
  - **Render integrado real y PASS acotado del tester:** `diagnostics/render_faceswap_recent_20261120.py` produjo `video5_1640_1736_v51402_refs3.mp4` con fuente real #79, banco reconstruido f0/f1329/f1631 y 96 fotogramas H.264/3,20 s. Comparativa `video5_1640_1736_v51402_refs3_samples.jpg` y `..._pixels.json`: en el temporal anterior el ROI del hombre tenía 4457–11565 píxeles cambiados (>20) en f1698–1717; en el nuevo son 0 en f1698–1722, y la mujer vuelve a cambiar en f1729/1730 (20832/24315 píxeles en su ROI). El tester independiente dio PASS visual/técnico **solo del tramo**. El primer render con banco de dos referencias generó MP4, pero el wrapper falló al leer el log ya cerrado y no recuperó la mujer; el wrapper se corrigió y se repitió con tres referencias. El retorno femenino aún muestra suavizado/artefactos; calidad visual y lote completo siguen pendientes.
  - **Logs y verificación:** `process_log_retention_days=30` en configuración; vídeo y lotes de fotos comparten limpieza por antigüedad de nombres `process_YYYYMMDD_HHMMSS.log`, sin seguir enlaces ni borrar el log activo. Prueba aislada: borra 31 días, conserva 29 días/activo/otro tipo; un valor enorme inválido no borra nada. El encabezado del render dice `v5.1402 (2026-11-20)`; los 11 logs del 20/11 seguían presentes tras dos renders. `probe_faceswap_tracking_20261119.py` PASS, Ruff de Python tocados PASS, `py_compile` PASS, JSON válido y build PyInstaller fresco PASS (`diagnostics/out/faceswap_latest_20261120/pyinstaller_dist/AutoAuto.exe`, 6 448 098 bytes). Ruff global aún informa 433 errores anteriores; no se declara lint global verde. Reviewer pendiente de veredicto final.

- **2026-11-20 — Auditoría del lote FaceSwap más reciente (en curso; sin PASS del lote).** Los logs `process_20261120_011957.log` a `process_20261120_021602.log` corresponden a 11 vídeos con 698 facesets origen. Los cuatro primeros MP4 están completos y `ffprobe` confirma 1784/1884/1204/170 fotogramas; el quinto quedó en 1736/3192, sin MP4 final, con un `.temp.avi` FFV1 decodificable secuencialmente. No hay proceso AutoAuto/FFmpeg activo. Las comparativas reales original/salida están en `diagnostics/out/faceswap_latest_20261120/`; son muestras, no PASS visual de los vídeos completos. En los cuatro logs completos: 1206/1749/1190/163 `Success`, 2/4/0/0 `Lost`; el segundo tiene 46 `RE-ENGANCHE`, concentrados en tramos de oclusión. No se registran excepciones de procesamiento.
  - **Regresión visual confirmada en el quinto vídeo:** `Bisexualoving_20260117...` selecciona inicialmente la mujer (bbox f1=(288,240,430,400)). El `.temp.avi` cambia el rostro del hombre en f1699–1700 (comparativa `video5_partial_anomalies.jpg`), tras `RE-BUSQUEDA closeup frame 1699 (sim=0.33, bbox=(614,188,772,406))`; f1700 pasa a sim 0.99 de esa cara. La última bbox femenina con `Success` era f1664=(44,165,411,481): salto de centros 466,2 px y extensión 367 px; el límite actual `1.3 × 367=477,1` lo clasifica como cercano. El detector real da score 0,795 al hombre. Con la referencia reconstruida de la mujer en f0, similitudes: hombre f1699=0,028; retorno femenino f1329=0,143 y f1729=0,265. Reducir el límite geométrico a 1,2 también impediría el retorno válido f1329 (salto/escala 1,271), por lo que no se aplica como arreglo ciego. La referencia UI original no está guardada; estas similitudes son de caras reales re-detectadas, no del banco UI. El usuario aprobó el control de identidad fija; v5.1402 lo implementa y lo verifica en un tramo. Falta el PASS visual del lote completo.
  - El banner dentro de estos logs dice `v5.1202`, aunque la UI y `logs/active_server.json` dicen `v5.1401`; hay una cadena de versión antigua en el registro de proceso. Se anota como deuda de trazabilidad, sin inferir que el motor ejecutado fuese v5.1202.

- **v5.1401 (2026-11-20) — FaceSwap vídeo 5: suelo de confianza para readquisición lejana.** El usuario confirmó que no conserva el banco destino del lote. Se localizó fuera del repo la fuente real #79, `[003856].jpg`. `diagnostics/probe_faceswap_video5_occlusion_20261119.py` permite medir detecciones, similitud y alfa de composición en los fotogramas originales, con render directo forzado o integrado. El escaneo de 18 fotogramas (`video5_ref_scan_280.json`) obtiene similitud 0,069 entre las detecciones de f280/f431; sus máximos entre cualquier pareja de caras **no prueban identidad**.
  - En el render **forzado, sin tracking ni identidad verificada**, f425–435 con la fuente real (`video5_diag_425_436_real_source_direct_v2.json/.mp4`), f431 cambia 6177 píxeles (>12) y la alfa media antes de cualquier paste oral es 0,245 dentro de la parte visible de la bbox; no hubo paste oral. La detección tiene confianza 0,188, similitud 0,221 con la referencia reconstruida f425, y el gate vigente `_recovery_face_clipped` la rechaza. Reproduce un parche pequeño si se fuerza la composición, **no** demuestra que v5.1400 lo entregue por la ruta integrada.
  - El render integrado f280–435 con una sola referencia reconstruida f280 solo hace 21 llamadas al swap y pierde el destino mucho antes de f431; otro tramo f550–553 con referencia f610 tampoco ancla al comienzo. La referencia reconstruida no equivale al banco UI original. El MP4 entregado anterior cambia 5103 píxeles en f431 frente al original (vecinos: 568 y 66), pero precede a v5.1400. No se modificaron originales ni entregas. El reviewer dio **APPROVE técnico** del probe corregido; el tester dio **PASS diagnóstico acotado** y ningún PASS visual del caso 5. Pendiente: preparar un banco destino nuevo verificado visualmente y repetir render integrado de la versión actual.
  - **Causa y corrección aprobada por el usuario:** el componente conectado del cambio en el MP4 entregado f431 ocupa 5097/5103 píxeles en x=105–183, y=282–361, sobre el fondo; `video5_f431_false_region_compare.png` lo confirma visualmente. El log original `logs/ui_start_20261118_235534.log` (línea 31192) registra `RE-BUSQUEDA closeup frame 432 (sim=0.56, bbox=(23,258,191,400))`. En el frame original 431, el detector devuelve esa misma bbox con `det_score=0.12364`; a 1024 también aparece una bbox próxima con score 0,121. Antes del arreglo, `_unverified_reacquire_jump` aceptaba ese salto con la bbox anterior del log y similitud 0,56. `config/model_configs.json` fija `reacquire_far_jump_min_detection_score=0.25`; `ProcessMgr.py` exige ese suelo configurable únicamente a readquisiciones lejanas sin corte visual reciente. El retorno real de f811 con corte visual y score 0,442 conserva su ruta.
  - **Verificación técnica de v5.1401:** `diagnostics/probe_faceswap_tracking_20261119.py` pasó de FAIL antes del arreglo a PASS con la detección real f431 y similitud 0,56 reproducida desde el log: `far_background_rejected=true` y `_apply_reacquired_face` devuelve `None` antes de cambiar el destino. También pasan los controles de candidato cercano de baja confianza, score inválido y retorno tras corte visual. Ruff de los Python tocados, `py_compile` y `git diff --check` con CRLF permitido PASS; build PyInstaller fresco tras el arreglo PASS (`build_v51401_review_fix.log`, EXE de 6 447 945 bytes). El reviewer emitió **APPROVE técnico** y el tester **PASS técnico acotado** para el cambio final. El lint global conserva los errores anteriores registrados en `spec/03`; no se declara PASS visual del vídeo 5 sin un banco destino representativo y render integrado nuevo.

- **v5.1400 (2026-11-19) — FaceSwap: continuidad de la cara parcial y revisión Animate Image.** El tester encontró dos fotogramas originales intercalados (f2112/f2114) en el primer render parcial, por lo que v5.1399 no alcanzó PASS visual de ese tramo. La traza real mostró el mismo destino con similitud al banco 0,728/0,713, score local 0,956/0,964 y geometría coherente; el gate lo rechazaba solo por visibilidad 53,9 %/54,5 % frente al mínimo 55 %. `config/model_configs.json` ajusta la visibilidad mínima a 0,53 y el desbordamiento máximo por eje a 0,47; conserva 4/5 landmarks dentro, ojo/nariz/boca, geometría, identidad ≥0,55 y cercanía al último swap real. `diagnostics/probe_faceswap_tracking_20261119.py` añade las detecciones reales f2112/f2114 al contrato y devuelve PASS, junto con el rechazo de identidad ajena/NaN/salto falso y el retorno tras corte visual real.
  - **Evidencia visual acotada:** `diagnostics/probe_faceswap_render_20261119.py partial --tag _fixed` produjo `partial_2109_2142_fixed.mp4` con 33/33 `Success` y 0 `Lost`. `diagnostics/measure_faceswap_partial_flicker_20261119.py` compara cada frame con el vídeo original: f2112 6→3948 y f2114 10→3201 píxeles cambiados (>12) dentro de ROI facial 220×260; los 33 frames superan 100. La comparativa `diagnostics/out/faceswap_tracking_20261119/samples_partial_flicker_fix.jpg` muestra continuidad sin los dos flashes originales. Tras el suelo de setup se repitió el render: `partial_2109_2142_strict_setup.mp4` volvió a dar 33 `Success`/0 `Lost` y es byte idéntico a `_fixed` (SHA256 EB67CA5A27498FFB29C448E14318D9EDDE205E5E4E36994454DFECC656300A16). El render ampliado `partial_2109_2304_extended.mp4` cubre f2109–2303: 195 `Success`/0 `Lost`, 195/195 frames con cambio facial continuo (mínimo 2230 píxeles >12 en ROI 320×380); `partial_extended_samples.jpg` muestra ocho muestras alineadas desde borde hasta f2303. Tester: PASS visual acotado de este tramo. Son referencias destino reconstruidas de frames reales; el banco UI original no se conservó. Queda pendiente PASS visual del lote completo, especialmente vídeos 4/5/7 y el tramo anterior a f2109.
  - **Auditoría adicional del lote:** `video4_gap_audit.jpg` compara originales/resultados en f405–800: gran parte del hueco 402–793 coincide con la cara elegida tapada por otra persona o de espaldas, por lo que no se atribuye entero al tracker. `video5_flicker_audit.jpg` y `video5_pixel_audit.json` muestran alternancia de swap en f190/210/280 con fotogramas originales f200/248/290 cuando un objeto tapa el rostro o la cara sale del encuadre. La auditoría completa de los 1421 frames de ese caso (`video5_full_pixel_audit.json`) registra 24 episodios de cambio de solo 1–2 frames al usar un umbral diagnóstico de 5000 píxeles; las comparativas `video5_pops_early.jpg` y `video5_pops_late.jpg` confirman microcambios aislados de nariz/ojos en f431/f552/f1096 que aún pueden causar destellos. No se fuerza una cara sobre la oclusión sin verificar identidad/geometría; la continuidad de vídeo completo, la cobertura de máscaras pequeñas y los 17 resets del caso 5 siguen abiertos.
  - **Caso 7, anclaje inicial:** la reproducción iniciada en f800 reveló que `_setup_selected_faces_frame_for_video` podía aceptar la única cara visible aunque no coincidiese con la elegida. En f808 la similitud real con la referencia inicial es 0,0847; `ProcessMgr.py` elimina el fallback `auto-única-cara` cuando existe selección explícita. El reviewer detectó que un ajuste UI laxo podía bajar el umbral a cero y aceptar la misma cara desde la readquisición sin ancla; se añadió `selected_target_setup_min_similarity=0.35` en configuración como suelo común al setup, readquisición y aplicación final hasta el primer swap real. `probe_faceswap_tracking_20261119.py` comprueba con f808 que no crea ancla ni readquiere bajo distancia UI 0,9, mientras f1 sí puede anclarse después; conserva PASS en salto falso, cara parcial y retorno con corte medido. El render integrado `return_800_815_loose_unseeded.mp4` y su log muestran que f808 conserva la escena original sin `SETUP` ni `Success` aunque sea la única cara. Los renders cortos f800–831 con banco reconstruido f0/f1 no reproducen la similitud 0,33 que registró el banco UI en f811: no son PASS visual del retorno y quedan solo como diagnóstico de la ruta de setup. El log original sí muestra recuperación en f811 y Success en f812/813.
  - **Validación:** probe con detecciones reales `report.json` PASS; Ruff de los Python tocados, `py_compile` y `git diff --check` con CRLF permitido PASS; `PyInstaller --noconfirm AutoAuto.spec` final PASS (`build_v51400_strict_setup.log`). El reviewer pidió corregir el bypass por distancia UI laxa y tras la corrección dio APPROVE técnico; el tester emitió PASS visual acotado del caso 3/caso 6 y PASS independiente para el anclaje inicial seguro del caso 7. Ninguno certifica los siete vídeos completos.
  - **Animate Image:** se revisó el último `animate/video_1795018500.mp4` (18/11 17:32): H.264 960×544, 97 frames/25 fps, 3,88 s y audio AAC. La hoja `diagnostics/out/animate_review_20261119/latest_frames.jpg` confirma movimiento sutil de cabezas/hombros y encuadre, pero no muestra el cumplimiento de la acción completa solicitada. El log `ui_start_20261118_171043.log` confirma traducción local, envío LTX-2 a ComfyUI y archivo final; la ruta de prompt no se perdió. El primer frame conserva la foto de entrada; `gr.Video` tiene `autoplay=True, loop=True` desde v5.1398, pero la reproducción real con audio en pywebview sigue sin verificar porque la UI registrada ya no está activa. No se declara Animate resuelto.

- **v5.1399 (2026-11-19) — FaceSwap vídeo: salto falso y continuidad de cara parcial.** El usuario autorizó continuar y eligió permitir swap parcial **solo con identidad y geometría verificadas**. En el vídeo 3, el primer anclaje falso del lote aparece en f270 (sim=0,27), pero el render de reproducción reveló una entrada anterior equivalente en f249: el embedding temporal daba 0,794 frente a solo 0,159 contra el banco fijo del destino; el hard cut de f196 había borrado `_last_swap_bbox`. `ProcessMgr.py` conserva ahora una ancla separada actualizada solo tras un swap real (no por ghost), aplica a seguimiento local y a todas las recuperaciones el gate de salto lejano con banco fijo, y mantiene un escape limitado a 3 frames tras **corte visual medido**, con sim de banco ≥0,30, det_score ≥0,40 y kps coherentes (retorno real de la cara elegida en f811 del vídeo 7). La cara parcial exige fracción visible ≥0,55, 4/5 landmarks dentro del frame con ojos/nariz/boca visibles, geometría coherente, sim de identidad ≥0,55 y proximidad al último swap real; los valores nuevos viven en `config/model_configs.json`. Los mínimos de identidad no se anulan por un ajuste laxo de distancia de la UI; un embedding NaN se rechaza. Se corrigió además el `NameError` de lint `eye_rel_c` → `_eye_rel_c` en un log de recuperación de foto.
  - **Evidencia real acotada:** `diagnostics/probe_faceswap_tracking_20261119.py` → PASS con detecciones InsightFace reales de f270/f2121/f811, hard cut reproducido, corte visual medido (f196 ya lejano para el falso; f811 reciente para retorno), identidad equivocada y NaN rechazados. `diagnostics/probe_faceswap_render_20261119.py false` renderizó f185–310: tras el arreglo no hay `Success` en la región falsa después de f28 del segmento. En recortes de f249/270/300 los píxeles cambiados (>12) bajan de 2001/2718/1434 a 14/33/100, mientras la cara válida f190 queda en 2308 antes y después; comparativa visual `diagnostics/out/faceswap_tracking_20261119/samples_false_before_after.jpg`. Render parcial f2109–2142: 31 `Success`, 0 `Lost`; muestras f2115/2121/2130/2140 cambian 3714/3456/4168/4732 píxeles dentro de la región de cara, comparativa visual `samples_partial.jpg`. `ruff check` de los Python tocados y `py_compile` PASS; `PyInstaller --noconfirm AutoAuto.spec` PASS (`build_final.log`). Primer review detectó el reset omitido y pidió cambios; se corrigió. Review final y tester aún pendientes en esta entrada.
  - **Límites/deuda:** son tramos de diagnóstico con referencias destino reconstruidas de fotogramas reales; el banco de tres referencias de la UI del lote original no se conservó. El vídeo 7 solo tiene prueba de retorno con detección real + similitud registrada, no render completo; los vídeos 4/5/7 y el lote completo aún requieren PASS visual. La rotación de logs eliminó el original `process_20261119_015653.log` durante las pruebas; se preservaron sus bboxes/similitud en `diagnostics/faceswap_tracking_20261119_baseline.json` y los logs posteriores en `diagnostics/out/faceswap_tracking_20261119/`. Los originales y MP4 entregados no se modificaron. El banner interno de ProcessMgr continúa en v5.1202 (deuda previa de trazabilidad).

- **2026-11-19 — Revisión de los siete FaceSwap de vídeo del lote 01:25–03:31 (sin cambio de código ni versión).** Se localizaron los siete MP4 de `output/` y se analizaron sus logs `logs/process_20261119_012552.log` a `logs/process_20261119_030141.log`; se compararon visualmente muestras de cinco vídeos con los originales de `.autodeep_temp/` y se leyó la ruta de tracking de `roop/ProcessMgr.py`. Resultado: **FAIL visual de tracking/continuidad**, no cierre de FaceSwap. Caso 3 (`015653`, 25 fps): el primer anclaje falso ocurre en frame 270 (`RE-BUSQUEDA closeup`, sim=0,27, bbox=(653,223,900,497)); tras otros `Lost`, el frame 300 vuelve a la misma región con sim=0,46. Las muestras 270 y 300–304 muestran rasgos faciales pintados sobre una zona sin cara. Caso 6 (`023506`, 29,97 fps): tramo 1754–2302 sin `Success` (549 frames); la cara seleccionada se ve en las muestras 2115, 2121, 2141, 2201 y 2295, pero el gate `cortada` rechaza bboxes que salen por el borde izquierdo aun con `track_sim=0,55–0,85`; en 2302 se recupera y en 2303 reaparece el swap de forma brusca. Caso 4 (`020959`, 30 fps): 402–793 sin `Success` (392 frames), 12 avisos `SIN IDENTIDAD CONFIRMADA`; muestras 405/600 se entregan originales en escena multicara. Caso 5 (`022453`, 30 fps): 90 logs `Lost`, 86 recuperaciones y 17 resets de tracking por salto de bbox (`face_cut=False`) en 1421 frames; requiere auditoría visual detallada de la cara elegida. Caso 7 (`030141`, 30 fps): 182–811 sin `Success` (630 frames) y 30 avisos de identidad sin confirmar; la muestra 400 no muestra a la persona del inicio, así que ese tramo no se imputa entero como defecto. Caso 2 (`014349`): 939 `Success` de 942 frames; continuidad básica del tracker en ese clip. Los recuentos `Success` son líneas del log, no una medida exacta de frames intercambiados. Evidencia visual temporal: `%TEMP%/autoauto_faceswap_review_20261119/faces_015653_299.jpg`, `border_023506.jpg` y comparativas originales/resultado; originales y MP4 permanecen intactos. La fuente #79 queda fijada en cada vídeo; el log de proceso aún imprime banner `v5.1202` aunque la app indica v5.1398, deuda de trazabilidad. No se ejecutó lint/build: solo diagnóstico de salidas y lectura de código. Priorizar reproducción acotada de falsa cara y cara parcialmente fuera de cuadro, manteniendo los gates que impiden cambiar a otra persona.

- **v5.1398 (2026-11-18) — Animate Image: reproducción automática del vídeo.** El usuario reportó que el resultado «no hace un video... es la misma foto más grande». El fichero generado tiene 97 fotogramas y movimiento medido; el primer fotograma es la foto de entrada (anclaje strength=1,0), así que con la reproducción manual el componente mostraba una foto estática. Se activa `autoplay=True` y `loop=True` en `gr.Video` de `ui/tabs/animate/ui.py` para que el vídeo se reproduzca al terminar. Pendiente: confirmación visual en pywebview (política de autoplay con audio).

- **2026-11-18 — README de portada para GitHub (documentación, sin cambio de versión).** El README raíz que describía únicamente la validación v5.1334 se sustituyó por una portada de AutoAuto v5.1397: seis áreas funcionales, arquitectura local, entorno probado, arranque sobre instalación ya preparada y enlaces a especificaciones/pendientes. Se comprobó que todos sus enlaces relativos existen. GitHub tenía `main` como rama predeterminada, pero el código AutoAuto está en `experiment/klein-9b`; el usuario eligió convertir esta última en predeterminada para que portada y código visible coincidan. La publicación se comprueba contra el estado remoto, no por la existencia de este documento.

- **v5.1397 (2026-11-18) — Mejorar fotos: presets sin duplicar, navegación y malla concurrente.**
  - El usuario observó presets duplicados, navegación incómoda y parpadeo al cargar el ejemplo; aportó un log con `Packet timestamp mismatch` y fallback continuo a elipse. Causa comprobada en código: los 15 presets de fábrica estaban en botones y dropdown; `preview_photo.change` y `slider.change` también reaccionaban a actualizaciones programáticas, duplicando renders; `FaceMesh.process()` se compartía entre hilos con bloqueo solo durante la creación.
  - `ui/tabs/enhance/ui.py` conserva los 15 botones de fábrica y muestra en «Mis presets» solo recetas personalizadas; el dropdown admite valores antiguos de API para no romper el lote. Se añadieron Anterior/Siguiente, contador y etiquetas numeradas que distinguen dos fotos con el mismo nombre. `events.py` actualiza la posición y los límites de navegación, usa `.input` para selector y ajustes (solo intervención humana), mantiene un único render tras importar/aplicar preset y usa progreso mínimo. El botón de ejemplo comunica si faltan imágenes. El reset conserva su render explícito. Las rutas equivalentes se comparan normalizadas.
  - `roop/img_editor/top_tier_finish.py` serializa todas las llamadas al mismo FaceMesh en las rutas de máscara y landmarks. Prueba real `diagnostics/probe_face_mesh_concurrency_20261118.py`: 24 máscaras + 24 lecturas de 468 landmarks de una foto facial en 8 hilos, **PASS**, sin error de grafo. La sesión antigua ya se había cerrado según `logs/ui_start_20261118_142735.log`; no se interrumpió un render.
  - `diagnostics/probe_enhance_ui_20261117.py` **PASS** sobre servidor aislado 7862: HTTP 200, `Cargar ejemplo` entrega 2 fotos locales, selector sin listeners `change` redundantes, rutas duplicadas en nombre distinguibles, anterior/siguiente y contador correctos, antes/después reactivo y lote 1/1; la API antigua `preset_name="Retrato"` sigue aceptada. `diagnostics/probe_enhance_presets_20261118.py` **PASS** con recetas/iconos y renders reales. PyInstaller `AutoAuto.spec` exit 0 (reutilizó el exe del launcher porque no cambió); `compileall`, Ruff de los archivos tocados y `git diff --check` exit 0. `ruff check .` sigue rojo con 433 errores anteriores repartidos por otros módulos; no se amplió el alcance a esa limpieza. sd-reviewer **APPROVE técnico** y sd-tester **PASS técnico acotado**. La desaparición del parpadeo en pywebview y la calidad visual de las fotos del usuario requieren comprobación al abrir la nueva versión; no se declara PASS visual por el probe de API.

- **v5.1396 (2026-11-18) — Mejorar fotos, presets y auditoría de arranque/Animate.**
  - El usuario aprobó el plan completo y corrigió la preferencia de servidor a `share=false`; `config.yaml` queda con túnel público desactivado (se conserva `server_name=0.0.0.0` para acceso LAN/Android). La sesión original PID 23924/7732 ya no existía al comprobarla; no se mató un render ajeno.
  - UI: comparador arriba, importación y galería compacta de 15 presets con SVG locales, ajustes manuales en acordeón cerrado y destino después de los presets. La captura servida `diagnostics/out/enhance_presets_20261118/ui_final.png` mostró iconos y acordeón; se corrigió un scrollbar horizontal detectado en la primera captura. `diagnostics/probe_enhance_ui_20261117.py` revalidado por sd-tester: 40 claves, 57 eventos, HTTP 200, API antes/después reactiva y lote 1/1.
  - Las recetas salen de `config/enhance/presets/*.json`; `enhance.json` conserva orden y límites, sin copia de valores. Cinco presets nuevos se materializaron en JSON. Guardar un preset personalizado ya escribe en la carpeta correcta, lo añade al selector, valida rango/nombre y no sobrescribe otro. «Blanco y negro» dejó de convertir `vibrance=0` en 100 y conserva grano gris. `diagnostics/probe_enhance_presets_20261118.py`: 15/15 recetas/iconos válidos, guardado real aislado y cinco renders en dos fotos (22 s); hojas `office_sheet.png` y `editor_camera_neutral_20261110_sheet.png`. Juicio visual del usuario pendiente; escenas diurnas no validan «Noche clara».
  - Log de arranque: `comfyui.db` y `.bkp` tenían DACL vacía; SQLite directo falló con `Permission denied`. Se restauró herencia de permisos sin alterar datos, `PRAGMA integrity_check` = `ok`; arranque fresco de ComfyUI emitió `Context impl SQLiteImpl`, respondió `/system_stats` y `/queue` con cola 0/0, sin error de base. Los avisos cu128/cu130 y Triton son de aceleración opcional; no se instalaron paquetes. El log LTX-2 terminó en 425 s sin resultado final, dentro del orden de magnitud del benchmark previo de 748 s en frío; no se atribuye fallo a `clip missing`/`unet missing` sin salida completa. Se corrigió el reporte falso CFG 5,5/FPS 12: el manager ahora informa 1,0/25, igual que el grafo; probe de wiring PASS con parámetros explícitos de UI.
  - Build PyInstaller (`venv/Scripts/python.exe -m PyInstaller --noconfirm AutoAuto.spec`) exit 0; `ruff check`, `compileall`, `git diff --check` y los probes UI/presets/LTX verdes; sd-tester **PASS técnico** y sd-reviewer **APPROVE técnico acotado a v5.1396**. `docs/PENDIENTES_2026-11-13.md` conserva P1–P12/P14–P15 abiertos; P13 solo técnico. Deuda de licencia de CodeFormer en «Ultra realista» anotada en `spec/03` sin cambio de modelo. No hay PASS visual general ni prueba completa nueva de LTX-2 o AutoAuto con `share=false` (la configuración y la UI aislada sí se comprobaron).

- **v5.1395 (2026-11-17) - Enhance: ajustes por foto (preset por imagen).**
  - `ui/tabs/enhance/events.py`: mapa `_PER_IMAGE_SETTINGS` (ruta→settings); `make_pair` carga/guarda ajustes por foto y sincroniza sliders; `make_after` guarda; `apply_preset` aplica al foto seleccionada y re-renderiza; `process_batch` usa settings por foto; `refresh_queue` limpia el mapa.
  - Verificación: `py_compile` OK y probe Enhance PASS.

- **v5.1394 (2026-11-17) - Enhance presets + Generate: más opciones de iluminación/planos/color.**
  - `ui/tabs/enhance/events.py`: 6 presets nuevos (Blanco y negro, HDR, Selfie, Vintage, Noche clara).
  - `ui/tabs/enhance/ui.py`: dropdown de presets ampliado.
  - `roop/img_editor/gen_prompt_modifiers.py`: ampliados iluminación (+5), planos (+5), color (+5), piel (+2) e imagen (+3).
  - Verificación: `py_compile` OK.

- **v5.1393 (2026-11-17) - Enhance: presets antes que sliders; tamaños optimizados.**
  - `ui/tabs/enhance/ui.py`: bloque de presets movido antes de los sliders; previews a 420 px, importación a 90 px, paddings reducidos (contenedor 8, panel 8, vista 2) para mejor uso de pantalla.
  - Verificación: `py_compile` OK y probe Enhance PASS.

- **v5.1392 (2026-11-17) - Enhance: previews arriba, sliders en horizontal ancho completo.**
  - `ui/tabs/enhance/ui.py`: se elimina la columna lateral de sliders; previews ANTES/DESPUÉS ocupan todo el ancho arriba y el panel de control (importar/destino/mejoras) queda debajo, con los sliders en filas horizontales como en el resto de pestañas.
  - Verificación: `py_compile` OK y probe Enhance PASS.

- **v5.1391 (2026-11-17) - UI: Mejorar fotos junto a Animate Image; sliders al lado de las previews.**
  - `ui/main.py`: pestaña «Mejorar fotos» movida para quedar entre «Animate Image» y «Settings» (en ambas funciones `create_ui`).
  - `ui/tabs/enhance/ui.py`: previews ANTES/DESPUÉS a 560 px con marco reducido, para que los sliders queden al lado y no debajo.
  - Verificación: `py_compile` OK.

- **v5.1390 (2026-11-17) - Enhance: antes/después más grandes con menos marco.**
  - `ui/tabs/enhance/ui.py`: altura de las dos imágenes ANTES/DESPUÉS subida de 420 a 560 px; padding del marco `.enhance-view` reducido de 8 a 4 px y borde suavizado; título con menos separación.
  - Verificación: `py_compile` OK.

- **v5.1389 (2026-11-17) - Enhance: pestaña más profesional, mejor uso del espacio.**
  - `ui/tabs/enhance/ui.py`: importación compacta con Tabs (Archivos/Carpeta) en vez de dos inputs apilados de 150px; botón «↺ Reset sliders» para volver a defaults.
  - `ui/tabs/enhance/events.py`: handler `reset_sliders` que lee `config/enhance/enhance.json` y restablece los 13 controles + status.
  - Verificación: `py_compile` OK, pestaña Enhance construye OK con Gradio 6.32.0.

- **v5.1388 (2026-11-17) - Animate: caché de modelos y prompts para evitar saturación de CPU.**
  - `roop/animate/animate_manager.py`: `_check_models` ahora cachea el resultado del walk del filesystem (TTL 60 s) en `self._models_cache`; antes recorría todo el directorio de modelos en cada llamada (UI valida modelos antes de cada animación).
  - `rewrite_prompt` cachea el resultado de NLP + traducción en `self._rewrite_cache` (máx 50 entradas); antes re-ejecutaba el analizador semántico y la traducción en cada animación aunque el prompt no cambiara.
  - Verificación: `py_compile` OK, probe LTX-2 animate wiring PASS 14/14.

- **v5.1387 (2026-11-17) - FaceSwap: botón «Usar resultado como origen».**
  - `ui/tabs/faceswap/ui.py`: nuevo botón `bt_use_result_as_source` (🔄) junto a `bt_open_output`.
  - `ui/tabs/faceswap/events.py`: handler `_use_result_as_source` busca el archivo más reciente en `roop.globals.output_path` y lo carga como source mediante `_safe_on_src_changed`. Permite encadenar swaps sin salir de la pestaña.
  - Verificación: pestaña FaceSwap construye OK con Gradio 6.32.0.

- **v5.1386 (2026-11-17) - Fix livecam tab: referencias None a ui.globals eliminadas.**
  - `ui/tabs/livecam_tab.py`: `bt_start.click` referenciaba `ui.globals.ui_selected_enhancer` y `ui.globals.ui_blend_ratio` (siempre None) como inputs → `AttributeError: 'NoneType' object has no attribute '_id'` al construir la pestaña. Eliminadas ambas referencias; `start_cam` lee enhancer/blend de `roop.globals` (configurados en FaceSwap). La pestaña ahora construye OK.

- **v5.1385 (2026-11-17) - Upgrade Gradio 3.41.2 → 6.32.0, re-activación de concurrency_limit/trigger_mode y CUDA en Image Editor.**
  - `pip install --upgrade gradio` → 6.32.0 (con gradio-client 2.7.2). Sin internet en el turno anterior; ahora conectado.
  - `ui/tabs/enhance/events.py`: re-añadidos `concurrency_limit=1` y `trigger_mode="always_last"` en los 5 listeners (input_files/input_folder change→then, preview_photo.change, btn_preview.click, slider.change, face_model.change). Gradio 6.x los soporta y evitan renders encolados al mover sliders.
  - `ui/tabs/faceswap/ui.py`: dos sliders con min=max=1 (preview_frame_num, face_selector_slider) rechazados por Gradio 6.x («Slider minimum must be less than maximum»). Cambiados a max=2 como placeholder (el máximo real se actualiza dinámicamente con `gr.update(maximum=...)` al cargar vídeo).
  - `ui/tabs/faceswap/events.py`: `gr.update(maximum=1, value=1)` → `maximum=2` (dos ocurrencias) para el mismo motivo.
  - `roop/img_editor/img_editor_manager.py`: 4 llamadas a `get_face_analyser_cpu()` (detección de caras en rutas de edición) sustituidas por `get_face_analyser()` (CUDA, con caché compartida con FaceSwap). Antes corrían en CPU (lento) aunque la GPU estaba disponible; ahora usan CUDAExecutionProvider. Las rutas de fallback CPU en `roop/face_util.py` (líneas 805, 1386) se conservan como fallback cuando CUDA falla.
  - Verificación: las 5 pestañas (faceswap, enhance, gen, img_editor, animate) construyen OK dentro de `gr.Blocks` con Gradio 6.32.0. Probe enhance UI PASS build + API + vista previa real.

- **v5.1384 (2026-11-17) - Fixes FaceSwap tracking (galería + umbral embedding) y compatibilidad Gradio 3.x en «Mejorar fotos».**
  - `pip install --upgrade gradio` → 6.32.0 (con gradio-client 2.7.2). Sin internet en el turno anterior; ahora conectado.
  - `ui/tabs/enhance/events.py`: re-añadidos `concurrency_limit=1` y `trigger_mode="always_last"` en los 5 listeners (input_files/input_folder change→then, preview_photo.change, btn_preview.click, slider.change, face_model.change). Gradio 6.x los soporta y evitan renders encolados al mover sliders.
  - `ui/tabs/faceswap/ui.py`: dos sliders con min=max=1 (preview_frame_num, face_selector_slider) rechazados por Gradio 6.x («Slider minimum must be less than maximum»). Cambiados a max=2 como placeholder (el máximo real se actualiza dinámicamente con `gr.update(maximum=...)` al cargar vídeo).
  - `ui/tabs/faceswap/events.py`: `gr.update(maximum=1, value=1)` → `maximum=2` (dos ocurrencias) para el mismo motivo.
  - Verificación: las 5 pestañas (faceswap, enhance, gen, img_editor, animate) construyen OK dentro de `gr.Blocks` con Gradio 6.32.0. Probe enhance UI PASS build.

- **v5.1384 (2026-11-17) - Fixes FaceSwap tracking (galería + umbral embedding) y compatibilidad Gradio 3.x en «Mejorar fotos».**
  - `ui/tabs/faceswap/events.py`: `on_face_selection_click` calcula `_gallery_columns = min(4, max(1, total_faces))` y lo pasa a `_resolve_gallery_index`. Antes se llamaba con `columns=4` fijo aunque la galería filtrada dibujaba menos columnas → el índice fila/columna se traducía a la cara equivocada (swap a la persona incorrecta). Docstring anota el porqué.
  - `roop/ProcessMgr.py`: umbral de actualización del embedding de recuperación `_tracking_embedding_*` de `best_score >= 0.55` a `>= 0.50`. El 0.55 era demasiado estricto para caras débiles/perfiles y acumulaba Lost Count; datos reales muestran caras verdaderas en 0.65–0.99 (mediana 0.98) y falsos positivos < 0.36, así que 0.50 queda bajo la mediana y muy por encima de los FP.
  - `roop/ProcessMgr.py` `_reset_tracking_on_scene_cut`: verificado que limpia `_target_face_assigned_*`, `_tracking_lost_count_*`, `_position_history_*`, `_consecutive_tracking_success_*`, `_strict_recovery_tick_*`, máscaras previas, bboxes/kps previos, `_hard_scene_cut_frame_*`, ventana de reacuisición, boca/oclusión temporal e histéresis de perfil. La identidad origen (`roop.globals.selected_face_references`) se conserva a propósito: el usuario eligió esa cara.
  - `ui/tabs/enhance/events.py`: Gradio 3.41.2 no acepta `concurrency_limit` ni `trigger_mode` (llegaron en 4.x); pasarlos rompía el cableado de toda la pestaña (`TypeError` en `wire_events`). Se eliminan ambos kwargs; la protección contra renders encolados ya vive en los handlers (`state._generation_lock` / `is_generating`). Probe de construcción de UI PASS.
  - Verificación: `py_compile` OK en ProcessMgr, events (faceswap/enhance), gen, animate, img_editor; las 5 pestañas construyen dentro de `gr.Blocks`; logs de proceso muestran TRACK Success con score 0.73–0.97 (todos > 0.50) y sin frames Lost; probe LTX-2 animate wiring PASS 14/14.

- **v5.1383 (2026-11-17) - Revisión profunda: EXIF, costura de luces/sombras, upscale 6x real y filtros de nombre con auto.**
  `process_single_image`, `_load_before` y `autopilot_propose` aplican `ImageOps.exif_transpose` y cierran el `Image.open` con `with`; prueba: JPEG 120×60 con orientation=6 → 60×120. `apply_lighting_color_cv2` sustituye el `np.where` duro en 0,5 por máscaras suaves (`clip((0.5-img)/0.5)`), evitando la costura de Nocturna/Paisaje. `apply_upscale_onnx` aplica 6x como 4x + Lanczos 1,5x (antes el factor 6 devolvía 4x con nombre `up6x`). Los filtros del nombre en `process_batch` usan `img_settings` (ajustes por foto) en vez de `settings` (base), corregido para «Modo automático». Probe PASS 38/38; Paisaje conserva 1764×1176.

- **v5.1382 (2026-11-17) - Mejoras P1-P8 de «Mejorar fotos»: progreso de lote, face analysis CUDA, notas visibles, caras más fuertes, vibrance con protección de piel, Paisaje 2x y grano determinista.**
  `do_batch` es generador con hilo worker + cola: muestra «Procesando N/M» y el paso actual, y la cancelación se comprueba por foto. `_get_face_analyser()` usa CUDA con fallback CPU en `detect_faces_count` y `apply_recover_faces_onnx`. `_render_after(return_notes=True)`: el estado de la vista previa muestra las notas del gate (pasos revertidos) y el resumen del lote cuenta «N foto(s) con pasos revertidos». Blend de caras `det * 0.35` (clip 0,08–0,50). Vibrance protege el tono de piel (hue 0–25/330–360) y «Color vivo» baja a 115. Paisaje usa shadow 0,7 / luces 1,1 y conserva el 2x (1764×1176); el gate de recorte tolera 2,5 pp en `lighting_color`. Deband/grano con semilla por imagen (`crc32` de los bytes) para resultados reproducibles. Estado: «upscale 2x al exportar (preview 1x)». Probe PASS 38/38; hoja regenerada (Ultra realista 4,6 s, Sin ruido 1,0 s, Paisaje 2,2 s); pendiente juicio visual del usuario.

- **v5.1381 (2026-11-17) - Bug crítico del gate de calidad: un paso revertido ya no descarta toda la cadena; Paisaje conserva el 2x.**
  `quality_gate_check` recibía `original` como `before` en todos los pasos; si un paso posterior (p. ej. `lighting_color` con recorte +19,8%) se revertía, devolvía el original y se perdían denoise/deband/sharpen/upscale. Ahora cada paso pasa `step_before = current` y el gate revierte solo ese paso. Verificado: Paisaje 882×588 → 1764×1176 (antes 882×588), notas «lighting_color revertido» sin perder el upscale; probe PASS 38/38. Hoja comparativa regenerada. Pendiente: progreso de lote, face analysis CUDA, notas visibles, blend/máscara de caras, protección de piel en vibrance.

- **v5.1380 (2026-11-17) - Animate Image ya no cae por defecto en CPU: motor ltx2_audio en GPUs ≤8 GB, Wan con block swap y FramePack con menos memoria reservada.**
  El usuario reportó bloqueos en Animate Image por procesos pasados a CPU. Causa raíz: FramePack (24 GB bf16) y Wan 5B (9,31 GB) no caben en 8 GB; el workflow usaba `load_device: "offload_device"` y FramePack reservaba 4 GB (`gpu_memory_preservation`), dejando ~4 GB para el modelo y saturando la CPU. Arreglo: (1) motor por defecto `ltx2_audio` en GPUs ≤8 GB (LTX-2 Q4 sí cabe y no usa offload); (2) Wan añade `WanVideoBlockSwap` con 20/30 bloques (no `main_device`: el propio nodo advierte que requiere 48 GB+); (3) FramePack baja `gpu_memory_preservation` a `max(0.5, vram-7)` = 1 GB en 8 GB; (4) aviso en la UI. Verificado: workflows Wan/FramePack construidos con los nodos nuevos, `gpu_mem=1.0GB`, motor por defecto `ltx2_audio`; el probe de enhance sigue PASS. Pendiente: prueba real de LTX en la GPU del usuario y juicio visual.

- **v5.1379 (2026-11-17) - Presets «Mejorar fotos» con IA real y limpieza: Realista, Ultra realista, Sin ruido, Sin bandas y Color vivo; Deband + Vibrance + motor de caras.**
  Diez presets de fábrica (Realista, Ultra realista, Sin ruido, Sin bandas, Color vivo, Retrato, Paisaje, Nocturna, Producto, Archivo) con valores de partida; «Ultra realista» usa CodeFormer + upscale 2 + grano fino. Nuevos controles: Deband (0–100, bilateral en croma + dither de luminancia), Color/Vibrance (0–200, saturación en HSV protegiendo piel) y Motor de caras (GFPGAN / CodeFormer / RestoreFormer++ / GPEN). `get_face_processor` mantiene un solo modelo activo y libera el anterior al cambiar (8 GB). SCUNet y NAFNet se descargaron y se midieron: SCUNet FP32 114 s/foto y FP16 >180 s, NAFNet 30 s a 256 px → inutilizables; no se usan por defecto (denoise clásico + Deband). Verificación: probe PASS 38/38 claves, 25 eventos, API antes/después reactivo y lote 1/1; hoja comparativa en `diagnostics/out/presets_v51379/contact_sheet.png` (10 presets, 1680×1032; Realista 16,3 s con primera carga, Ultra realista 10,2 s, Sin ruido 2,4 s, Sin bandas 1,1 s). Pendiente: juicio visual del usuario y, si lo pide, habilitar SCUNet como opción lenta.

- **v5.1378 (2026-11-17) - «Mejorar fotos» por fin como se pidió: ANTES | DESPUÉS lado a lado de verdad, primera foto seleccionada, Después reactivo, sin tablas confusas y botón «Abrir carpeta» visible (petición de usuario).**
  Causa raíz del apilado: el `gr.Row` de Gradio usaba `flex-wrap: wrap` y cada `gr.Image` imponía su ancho intrínseco (882 px) como `min-width` de la columna, así que ANTES y DESPUÉS quedaban uno debajo del otro (hero de 1252 px). Arreglo: `.enhance-compare-row { flex-wrap: nowrap }` + `.enhance-view { min-width: 0; width: 0; flex: 1 1 0 }` + `img { max-width: 100%; height: auto }`; verificado en vivo (Chrome headless + CDP): dos vistas de 308×279 en la misma fila (x=160/478, y=174), imágenes 288×192 sin bandas negras, hero y panel alineados a y=163. La foto 1 se selecciona sola al importar (refresh_queue → `preview_photo` = primera) y el panel Después cambia al mover un slider (src distinto; probe API PASS). Se eliminan las dos tablas (`queue_table` y `result_table`/`result_accordion`) que el usuario no entendía: queda una línea de cola «N foto(s) · vista previa: X» y un resumen de lote con errores si los hay. El bloque «Destino» (ruta + «Abrir carpeta» + formato) se sube junto a Importar para que el botón se vea sin bajar (antes estaba a 1541 px, fuera de pantalla); botón 160×36 con texto de 12 px y acuse inmediato (`folder_status`) debajo. Botones secundarios a 32 px y estado a 11 px. Título de la pestaña sin emoji. Repaso de código (mismo turno): el selector usa rutas completas como valor y el nombre solo como etiqueta (dos `window.png` ya no colapsan); `_BEFORE_CACHE` es LRU de 12 entradas; la importación usa solo `change` + `.then(make_pair)` (sin doble render por `upload`+`change`); el probe llama a `/do_batch` contra un `TemporaryDirectory` y ya no sobrescribe `enhance/`. Contrato 35/35 claves, 22 eventos; probe PASS (API + vista previa real 882×588, cambio medio 11,34) y verificación en vivo: dos vistas 308×279 en la misma fila, hero/panel a y=163, botón «Abrir carpeta» a y=599, el src del Después cambia al mover un slider. Pendiente: juicio visual del usuario en la UI viva (reinicio de la app).

- **v5.1377 (2026-11-17) - «Mejorar fotos» usable de verdad: ANTES | DESPUÉS lado a lado, primera foto automática, Después reactivo, sliders alineados, botón de carpeta y tablas plegadas (petición de usuario).**
  Causa raíz de la desalineación: el helper `_slider` inyectaba un `gr.HTML` de etiqueta ANTES del slider dentro del mismo `gr.Row`, así que Gradio repartía 4 hijos por fila (etiqueta-slider-etiqueta-slider) en 4 columnas. Ahora cada `_slider`/`_dropdown` crea su propio `gr.Column` (etiqueta arriba, control abajo) con `min_width` y estilo uniforme. Se sustituye el `gr.ImageSlider` único por dos `gr.Image` lado a lado (ANTES | DESPUÉS, altura 430, `object-fit: contain`, solo botón de pantalla completa) y se añade selector «Foto en vista previa» que se rellena con la cola y selecciona la PRIMERA foto automáticamente al importar (upload/change → refresh_queue → then(make_pair)). Cualquier slider/dropdown re-renderiza SOLO el panel Después (queue=True, trigger_mode=always_last, concurrency_limit=1, sin encolar renders pesados). El «antes» se cachea escalado a ≤1024 px por ruta+mtime y la vista previa omite el upscale (no aporta visual en un panel reducido y era el paso más lento); el lote sigue aplicándolo completo. Nuevo botón «Abrir carpeta» junto a la ruta de destino (`os.startfile` con fallback xdg-open, patrón de Image Editor) y botón «Cargar ejemplo» conectado de verdad (antes no tenía handler). Tablas: resumen de una línea «N fotos en cola · vista previa: foto1» + «Cola» en Accordion cerrado; «Resultado del lote» nace oculto y se abre tras procesar con resumen N/N OK. Evidencia real: `diagnostics/probe_enhance_ui_20261117.py` PASS — contrato 37/37 claves, 26 eventos, GET / -> 200 (333.267 bytes) con ANTES/DESPUÉS/Abrir carpeta, vista previa real 882×588 con cambio medio 11,34 entre ajustes (1,08 s base / 2,43 s ajustada en caliente). La prueba por la API real (`/refresh_queue`, `/make_pair`, `/make_after`, `/do_batch`) confirma: el selector se rellena con la PRIMERA foto (`window.png`), el par antes/después se genera sin selección manual, el panel Después cambia al mover ajustes y el lote deduplica archivo+carpeta (1/1). Dos defectos encontrados y corregidos durante la verificación: `events.py` usaba `gr.update` sin importar gradio (NameError) y la tabla de resultado usaba ✅/❌ (emojis prohibidos en esta pestaña desde v5.1376) -> sustituidos por OK/ERROR. Juicio visual final pendiente del usuario (la UI viva en 7861 no se reinició: lo hace el usuario).

- **v5.1376 (2026-11-16) - «Mejorar fotos» sin emojis sueltos: texto limpio y alineado; los colores ya son el código visual (petición de usuario).**
  Eliminados TODOS los emojis de la pestaña (título, cabeceras de sección, etiquetas de slider, botones, checkboxes, labels de archivos). La identidad visual pasa a ser cromática: barra de color por slider + viñeta azul por sección. Reemplazos desde consola fallaron en 2 cabeceras por codepage (📥/🧩) -> corregidos con ediciones directas. Verificado 0 emojis en ui/tabs/enhance/ui.py; contrato events.py 30/30 («keys faltantes: NINGUNA», «BUILD ENHANCE TAB OK»).

- **v5.1375 (2026-11-16) - «Mejorar fotos» más visual: acentos por mejora, hero con glow y título en degradado (petición de usuario).**
  Capa visual sobre el layout de v5.1374 (estructura intacta): título con degradado azul-cian, viñeta azul en cada cabecera de sección, hero de foto ampliado a 460px con fondo #0d0d0d y glow azul (border #1f4d8f + box-shadow), barra de color a la izquierda de cada etiqueta de slider (border-left currentColor) con un color de acento por función (denoise cian, sharpen violeta, caras rosa, luz ámbar, sombras gris, luces amarillo, temperatura azul claro, grano indigo), botón PROCESAR LOTE con degradado y glow. Contrato events.py 30/30 verificado («keys faltantes: NINGUNA», «BUILD ENHANCE TAB OK»).

- **v5.1374 (2026-11-16) - «Mejorar fotos» continua y visual: menos compartimentos (petición de usuario).**
  Cambio de cajas por categoría a vista de trabajo tipo Topaz: foto protagonista (comparador antes/después 380px) a la izquierda, columna única de controles a la derecha (Importar -> Mejoras como lista continua en filas de 2 sliders, sin tarjetas), banda inferior única de Presets + Exportación (checkboxes, destino, formato en una fila) + resultado del lote + botones de lote. CSS: .enhance-hero y .enhance-panel sustituyen a .enhance-card (ya sin cajas por categoría). Corregido durante la edición: preset_status quedó sin definir en el primer layout -> NameError en test aislado -> reañadido. Contrato events.py 30/30 claves verificado («keys faltantes: NINGUNA», «BUILD ENHANCE TAB OK»).

- **v5.1373 (2026-11-16) - «Mejorar fotos» compacta: misma potencia, menos huella visual (petición de usuario).**
  Compactación del diseño Topaz de v5.1372: paddings reducidos (20->12px), título 30->22px, secciones 12->11px, cards 10->6px, botones y barras más bajos (44->38px), preview 420->300px, subtítulos de tarjeta en 10px, checkbox de exportación abreviados, «Ejemplo» fusionado en la fila de importación, encabezado de «Resultado del lote» eliminado (el Dataframe ya trae label), densidad interna de bloques Gradio acotada al contenedor (.info-text/.block/.gap). Cero controles eliminados: las 30 claves del contrato con events.py verificadas por test aislado («keys faltantes: NINGUNA», «BUILD ENHANCE TAB OK»).

- **v5.1372 (2026-11-16) - «Mejorar fotos» rediseñada estilo Topaz según docs/PLAN_PESTAÑA_TOPAZ_2026-11-15.md §4.2; commit v5.1371 pusheado a origin.**
  Commit 1 (98c5dba, pusheado a origin/experiment/klein-9b): arreglos del día (exe, surrogates, tab duplicado, _slider) + basura excluida del repo (.gitignore: generate/, .kilo/, temporales). Rediseño UI: de 3 tabs a pantalla única Topaz (Importación -> Antes/Después -> Pila de mejoras -> Presets de estilo -> Modo y Exportación) con paleta Topaz (grises #131313/#1c1c1c planos, acento azul #0f82ff, sin gradientes arcoíris); CSS envuelto en <style> (ya no se pinta como texto). Contrato con events.py intacto: las 30 claves del dict de retorno verificadas por test aislado («keys faltantes: NINGUNA», «BUILD ENHANCE TAB OK»). Sin pestañas nuevas ni cambios en otras pestañas (regla del plan §4.3).

- **v5.1371 (2026-11-16) - Arreglado tab «Mejorar fotos»: eliminado el registro duplicado en ui/main.py y reconstruido el helper _slider (NameError que rompía la pestaña).**
  (1) El tab «✨ Mejorar fotos» estaba registrado DOS veces seguidas en cada contexto de Blocks de ui/main.py (líneas ~321/329 y ~496/504, bloques idénticos); eliminado el duplicado en ambos. La página de la UI pasó de 713 KB a 689 KB. (2) El tab mostraba «Error: name 'slider' is not defined»: el helper `_slider` de ui/tabs/enhance/ui.py (archivo untracked, sin historia en git) fue reconstruido a partir de la firma de sus 7 llamadas y del CSS existente (.enhance-slider-wrap/.enhance-slider-label); devuelve gr.Slider con etiqueta HTML coloreada y readout nativo (sin valores falsos). (3) Además faltaban `highlight_slider` y `temperature_slider`, usados en el dict de retorno y en events.py pero nunca creados; añadidos en la tarjeta «Luz y Color» con rangos coherentes con los presets (highlight 0.0-2.0 paso 0.05, temperatura 0-100 paso 1, defaults de config/enhance/enhance.json). Evidencia: build del tab en proceso aislado -> «BUILD ENHANCE TAB OK» tras reproducir antes el NameError exacto. Página de la UI: HTTP 200.

- **v5.1370 (2026-11-16) - Arreglado AutoAuto.exe (ImportError en launcher) y el 500 de Gradio por surrogates; exe vuelve a ser pequeño (6,1 MB); git desbloqueado para el IDE.**
  Diagnóstico con evidencia: (1) Exe no arrancaba: `launcher.py` hace `from run import main` pero `run.py` no definía `main()` (todo el arranque vivía bajo `if __name__ == "__main__"`) -> ImportError reproducida con stderr capturado: `cannot import name 'main' from 'run'`. Dump de crash 13:53 (AutoAuto.exe.13424.dmp) analizado con cdb: ucrtbase!abort desde python310.dll. Fix: el bloque `__main__` de run.py convertido en `def main()` (se conserva `python run.py`); eliminada sonda de debug con ruta absoluta D:\PROJECTS\AUTOAUTO\debug_swap.log. (2) UI daba HTTP 500 en `/` y `/config`: orjson rechaza surrogates; parche diagnóstico en roop/core.py (Patch Gradio UTF-8) localizó la causa raíz: labels con escapes de pares UTF-16 `"\uD83D\uDCC1"`/`"\uD83D\uDCC2"` en ui/tabs/enhance/ui.py (líneas 176 y 180), que Python no combina -> surrogates sueltos. Corregidos a `"\U0001F4C1"`/`"\U0001F4C2"`. El parche sanitizador se conserva como defensa. Verificación real: AutoAuto.exe (6,1 MB) arranca, Gradio `http://0.0.0.0:7861` vivo, `GET /` -> 200 (713 KB), `GET /config` -> 200, sin avisos de saneado. (3) IDE colgado al abrir el proyecto: AutoAuto.exe de 3,7 GB (onefile PyInstaller con modelos) y 32 archivos de venv_ext registrados en git -> cada git status diffaba el binario (>120 s). `git rm --cached` de ambos + .gitignore; git status ahora 0,11 s. Decision de usuario: exe pequeño que llama a `venv\Scripts\python.exe run.py` (launcher original restaurado, e4f6529); el exe autocontenido de 3,7 GB fue eliminado de raiz y dist/. NOTA: la app abre túnel público gradio.live (share=True) — revisar LOCAL-FIRST.

- **v5.1368 (2026-11-16) - Rediseño UI «✨ Mejorar fotos»: 3 tabs (Importar/Ajustar/Settings), sliders diferenciados por categoría, previsualización en tiempo real en todos los controles, estilo Topaz profesional. Sin duplicado de pestaña.

- **v5.1367 (2026-11-16) - UI profesional de «✨ Mejorar fotos» + recompilación de AutoAuto.exe.**
  Rediseño de `ui/tabs/enhance/ui.py`: tarjetas por categoría (Denoise+Upscale, Nitidez+Caras,
  Luz+Color, Grano Fotográfico), cada una con sliders diferenciados (icono, color, valor numérico,
  info contextual). CSS profesional oscuro con gradientes, botones btn-batch/btn-apply/btn-save.
  Claves de retorno mantenidas para compatibilidad con `events.py`. Sintaxis verificada con ast.parse.
  Importación real OK (cuda providers detectados). Recompilación PyInstaller pendiente de ejecución.

- **v5.1366 (2026-11-16) - Recompilación de AutoAuto.exe (PyInstaller) + subida de versión.**
  Se recreó `launcher.py` (entry point del `.spec`, faltaba en working tree) y se compiló con
  `venv\Scripts\python.exe -m PyInstaller --noconfirm AutoAuto.spec`. Resultado: `dist/AutoAuto.exe`
  build OK y arranque verificado (PID activo, sin errores). Solo advertencias de DLLs opcionales
  (cublas, tensorrt) no críticas. Versión subida a v5.1366 en `run.py` banner, `roop/core.py`
  `APP_VERSION`, `AGENTS.md` y `spec/00-PRINCIPIOS.md`.

- **v5.1365 (2026-11-16) - Pestaña «✨ Mejorar fotos» estilo Topaz: F0 PASS.**
  Nueva pestaña `ui/tabs/enhance/` (registrada en `ui/main.py`): importar fotos
  sueltas o carpeta, cola, **slider antes/después** (`gr.ImageSlider`), un slider
  por mejora (denoise, sharpen, recover faces, upscale, exposición, sombras,
  luces, temperatura, grano), **5 presets de fábrica** (Retrato/Paisaje/Nocturna/
  Producto/Archivo) + guardar/cargar en `config/enhance/presets/`, modo automático
  (autopilot mide ruido, nitidez, caras, brillo y tamaño y propone la pila con
  razones explicadas) y lote secuencial a `enhance/` con cancelar. 100 % local y
  GPU-first para RTX 3060 Ti 8 GB: Real-ESRGAN x2/x4 + GFPGAN v1.4 vía ONNX
  Runtime (CUDA EP, singletons, tiles, `torch.cuda.empty_cache()` entre imágenes);
  denoise/sharpen/luz/color/grano vectorizados en OpenCV (no hay ONNX de denoise
  en disco). Gate de calidad por paso con semántica real: revierte recorte de
  tonos (>0,8 pp) en todos, nitidez solo en pasos sin cambio de escala y solo si
  cae >25 %, ruido solo si AUMENTA en denoise; así no deshace denoises/upscales
  legítimos (la primera versión del gate los revertía por medir resoluciones
  distintas — corregido). Salidas con nombre único por lote (dos entradas con el
  mismo stem ya no se pisan). Barra de progreso sustituida por estado HTML +
  tabla de resultados: `gr.Progress` no puede ser output en Gradio 6.
  Evidencia: `diagnostics/probe_enhance_f0_20261116.py` PASS — 10/10 fotos reales
  (lista explícita segura; `testdata/` tiene imágenes sexuales explícitas
  preexistentes que el probe ya NO arrastra), salidas válidas y únicas en
  `enhance/probe_f0/`, upscale 2x conservado, presets/autopilot/gate OK; log en
  `diagnostics/out/probe_enhance_f0_20261116.log`; UI completa construye
  (492 bloques, 30 componentes en la pestaña); ruff limpio en el paquete.
  Pendiente F1–F4 según `docs/PLAN_PESTAÑA_TOPAZ_2026-11-15.md`.

- **v5.1364 (2026-11-15) - Audio en Animate Image: fallos corregidos.**
  LTX-2: `_trim_last_frame` reescrito con ffmpeg (libx264 + audio copy) en
  lugar de OpenCV (mp4v sin audio) → el trim de 49f ya no destruye la pista
  de audio nativa. MMAudio: auto-arranque de ComfyUI si no está vivo (mismo
  criterio que LTX-2), para que el audio ambiente se integre sin levantar
  ComfyUI a mano. Sigue pendiente validación visual con render real.

- **v5.1363 (2026-11-15) - Revisión Animate Image: fallos corregidos.**
  `events.py`: eliminado bloque muerto que importaba `get_vram_gb` y
  `suggest_resolution` desde `roop.animate.animate_photo` (ninguno existe ahí;
  `res_status` se calculaba y no se usaba) → el try siempre fallaba en silencio.
  `img_data` dict handling robustecido (background/image/path → PIL).
  `animate_manager.apply_face_stabilize`: la salida mp4v se transcodifica a
  H.264 (libx264) para que se reproduzca en el navegador/UI; si ffmpeg falla
  se conserva mp4v. UI reconstruye OK (111 deps). Pendiente: render real
  con Wan/FramePack/LTX-2 (GPU ocupada).

- **v5.1362 (2026-11-15) - FaceSwap drag&drop en destino.**
  Destino pasó de evento `.change` a `.upload` (igual que origen, que sí
  funcionaba): en Gradio 6 el drag&drop dispara `.upload`, no `.change`.
  `bt_destfiles` dejó de filtrar por `file_types=["image","video"]` (mismo
  problema que v5.1319 en origen): el filtro bloqueaba archivos válidos en
  cliente/servidor. `_resolve_target_path` resuelve robustamente
  str/dict/NamedString/FileData (`.name`/`.path`). Se añadió wrapper
  `_safe_on_dest_changed` con dedup por clave de archivos (evita recargas
  múltiples al soltar varios archivos). Backend verificado por API
  (`_safe_on_dest_changed` → complete con preview).

- **v5.1361 (2026-11-14) - `clothing` PASS visual + `structural_zone_close` probe.**
  `clothing` («Change the astronaut suit to blue») ahora pinta el traje de azul:
  19,16 % cambiado, bbox (31,198,355,505), cara y fondo conservados. La
  medición de persona ya no falla y el modelo aplica el color (el fallo
  anterior de 2,86 % era de una corrida previa a la corrección de medición).
  `body` sigue con artefactos en la cara (estiramiento); el astronauta sí es
  más alto (bbox 0,43,389,512) pero la fidelidad del resto no.
  `probe_editor_structural_zone_close` PASS (persona que roza la zona se resta;
  persona dominante se conserva).

- **v5.1360 (2026-11-14) - `remove`: inpaint dedicado LaMa (ONNX, CPU).**
  El invento (bandera, insignia, conectores) desaparece: se sustituye el
  muestreo difuso a fuerza 1,00 (que arrancaba de ruido dentro de la máscara y
  pintaba una escena nueva) por LaMa big-lama (`models/lama_fp32.onnx`, 199 MB,
  entrada fija 512², CPU). Nuevo módulo `roop/img_editor/lama_inpaint.py` y rama
  en el pase local cuando `structural_remove` + máscara medida. Tiempo del pase
  ~30 s (antes 85–180 s). La máscara se usa tal cual (sin dilatar): dilatarla
  4–16 px deja un parche moteado en la esquina inferior derecha; sin dilatar
  queda un resto del visor (objeto real, no invención) en la esquina inferior
  derecha. Se eligió el menos dañino: sin dilatar.
  `remove.png` (512²): changed_bbox (283, 301, 508, 512), 14,9 % cambiado,
  sin insignias ni conectores nuevos. Resto del visor pendiente (mejora de
  máscara o un segundo pase sobre ese fragmento).
  Probes: py_compile/ruff OK; `real_editor_category_matrix ... remove` PASS del
  despacho (30 s). Visual: mejora clara frente a v5.1359 (sin invención); resto
  del visor documentado como límite.

- **v5.1359 (2026-11-14) - `remove`: medido el invento y descartados cinco arreglos.**
  Petición «Remove the black helmet from the photo» sobre la foto de astronauta.
  Lo que sí está bien: máscara `black helmet` 11,9 %, bbox (283, 301, 508, 512),
  área cambiada 8,7 % y **0 píxeles cambiados fuera de la zona**; el casco se va
  entero. Lo que falla: el relleno inventa detalles **dentro** de la zona (manga
  nueva con parche de bandera y bandera de EE. UU., insignia redonda duplicada,
  conector, franja de rayas) y cambia en cada semilla. El pase local corre a
  `denoise=1.00` con `cfg=1.0`, así que el negativo se ignora y el init también:
  dentro de la máscara el modelo pinta una escena nueva. No es máscara ni
  despacho.
  Probado y **rechazado con evidencia** (logs `logs/p1_remove_*_20261114.log`):
  1) frase positiva «no copiar emblemas/insignias/logos/texto/herrajes» en
  `removal_reconstruction_prompt` → el invento sigue (dos renders); queda en
  config como requisito, **no se declara eficaz**; 2) rellenar el hueco desde el
  borde (NS) antes de muestrear → el casco queda como masa oscura (mismo límite
  del comentario de `SURFACE_CONTINUATION_DENOISE`); 3) segunda pasada
  vaciar+armonizar a 0,60 → mancha difusa sobre el traje y restos del casco, se
  revirtió entero; 4) reparación por paleta del anillo → borra bandera y aro pero
  deja borrón azul; 5) detector de copias por NCC → no discrimina (`remove` 0,749
  vs la edición correcta de `clothing` 0,997).
  Queda abierto solo con dos salidas posibles, ninguna hecha: modelo de inpaint
  dedicado para `structural_remove` (LaMa ONNX en CPU, medido antes de integrar)
  o aceptar el relleno generativo y firmar el PASS con esa observación.
  Probes: `probe_editor_added_face_routes_20261018` PASS;
  `probe_editor_region_proposal_dispatch` falla en expectativa de conceptos
  («Change the person's blouse to red» → `structural_add`), ajeno a este cambio;
  `probe_editor_local_uncover_20261029` 14/15, el fallo es la foto del caso.

- **v5.1358 (2026-11-14) - Fix del halo de `background` y cierre de P13.**
  **Halo (medido, no supuesto).** `_background_mask_from_people`
  (img_editor_manager.py:4118) dilataba la silueta 7 px, así que el escenario se
  repintaba desde 2,27 px fuera de la persona y esa banda conservaba el fondo
  viejo: era el halo claro pegado al pelo. Ahora acepta `gap` (kernel impar) y las
  demás rutas siguen con 7 px; solo la de escenario pasa 1 px. Además la silueta
  medida sobrecubre ~1 px de fondo en el 45 % de su contorno, y
  `_restore_measured_people` la devolvía tal cual: ahora se entrega 1 px por dentro
  con el mismo desvanecido de 2 px. Medición CPU
  (`diagnostics/diag_background_halo_band_20261114.py`): franja 2,27 px → **0,00
  px**, escenario pintado 57,3 % → 59,4 % (el guard pide ≥12 %).
  **PASS visual pendiente**: al intentar re-renderizar `background`, el puerto 8188
  estaba ocupado por un ComfyUI del Python del sistema (PID 1736, no el venv) con
  un render LongCat en cola; AutoAuto se negó a usarlo y no se tocó nada. La
  comprobación a ojo queda para el siguiente hueco de GPU.
  **P13 CERRADO**: viewpoint 4 OK, selección corporal 13/13, compuesto 61 OK/0
  FAIL, selector 31/31, cámara 4/4, only_finish 5/5, descalzo 9/9, todos exit 0.
  Siguen en FAIL `probe_scene_head_preserve` y `probe_editor_structural_zone_close`:
  verificado hoy revirtiendo el cambio, fallan igual antes → previos, no regresión.
  **Matriz P1 revisada a ojo (v5.1356/v5.1357 en adelante):** `clothing` ya pinta el
  traje azul (`changed_fraction` 0,19; el «2,86 %» de la entrada v5.1356 es de la
  corrida ANTERIOR al fix y no describe el código actual), con restos naranjas en
  hombro/brazo derecho; `body` se retira el PASS (más alto sí, pero con brazos
  duplicados y bandera deformada); `remove` desaparece el casco pero inventa
  insignia/conectores en el pecho.
  **Reinicios del PC**: dos `Event 41 Kernel-Power` (20:34:46 y 21:38:27), sin BSOD,
  sin WHEA/TDR y con 17 `disk` Id 51 del disco USB 2 TB (unidad `E:`) justo antes
  del segundo. Mitigado con `DISKIDLE=0` (AC y DC, confirmado `0x0`). Detalle en
  `docs/DIAGNOSTICO_REINICIOS_2026-11-14.md`.

- **v5.1356 (2026-11-14) - Fix: medición de persona en imperativos inglés (P1 clothing/body).**
  Causa raíz confirmada por reproducción (`diagnostics/diag_body_not_measured.py`):
  `_visual_subject_description` (img_editor_manager.py:6348) devolvía `[]` de
  queries para imperativos en inglés («Change the astronaut suit to blue», «Make
  the astronaut taller») porque `_PREDICATE_RE` no casa «change»/«make» y
  `_determiner_phrases` solo funciona si el texto empieza con determinante.
  Dos cambios mínimos, sin listas de palabras/prendas:
  1. `_visual_subject_description`: fallback a `_leading_visual_subject` (ya
     existente, gramática) cuando no hay predicado ni sintagma determinante
     inicial. «Change the astronaut suit to blue» → queries `['astronaut suit']`
     (antes `[]`).
  2. `_measured_subject_passes` (≈5889): si ningún query mide cobertura y hay
     exactamente UNA cara detectada, mide con `_build_clipseg_person_mask`
     (segmentador genérico de persona ya usado en `_background_mask_from_people`
     y protección de add). El sujeto sigue siendo el del prompt; solo cambia el
     medio de segmentación; la compuerta de solape facial sigue vigente.
  Resultado real sobre `editor_camera_neutral_20261110/before.png`:
  - `body` «Make the astronaut taller» → **PASS visual**: astronauta claramente
    más alto, cara y fondo conservados; log «cuerpo medido, el resto de la foto
    se conserva»; changed_bbox (0, 77, 364, 512); salida
    `diagnostics/out/editor_categories_20261114/body.png`.
  - `clothing` «Change the astronaut suit to blue» → medición arreglada (máscara
    «astronaut suit» 31,4 %; ya no abstención), pero LongCat no pinta el traje de
    azul: cambia el 2,86 % de píxeles (borde en el cuello). Es calidad del modelo
    con cambio de color, no de medición. Salida `clothing.png`.
  Regresiones verdes: selector 31/31, cámara 4/4 (FAIL=0), body regime 13/13,
  subject descriptors OK, ruff sin errores nuevos (E501/W293 preexistentes).
  `probe_editor_compound_fast_dispatch` sigue en FAIL pero es preexistente (falla
  igual con el fix revertido; línea 382 espera mask None y «Pon la pared azul»
  ya medía 2,4 % antes de este fix).
  P1 sigue abierto: clothing (calidad de color), background y remove. Nada se
  declara HECHO sin PASS del tester y APPROVE del reviewer.

- **2026-11-14 — P1: primera revisión visual de la matriz real (9/9 categorías).**
  Fuente `editor_camera_neutral_20261110/before.png`; salidas en
  `diagnostics/out/editor_categories_20261114/`. PASS: `camera`, `people`,
  `features` (pelo negro; Moondream local dijo FAIL y el ojo ve el cambio), `pose`
  y `add` (con observación: pose cambia identidad/escenario; add pega la bandera
  al casco). FAIL: `clothing` y `body` (foto intacta, «cuerpo no medido»),
  `background` (duplica a la persona y cambia ambas caras), `remove` (el casco sale
  por recorte, no por eliminación, y aparece un parche nuevo). Registrado en
  docs/PENDIENTES_2026-11-13.md §P1. Sin cambios de código este turno.

- **2026-11-14 — Revisión de pendientes y última sesión de Codex (sin cambios de código).**
  PENDIENTES P1–P15 revisados. La última sesión de Codex (00:08–00:23) murió por
  límite de uso tras: fix de `view` suelto como cámara, política
  `multi_clause_uncover_enabled=false`, body regime 13/13; el reviewer «Kuhn» se
  quedó sin veredicto. La matriz P1 siguió sola hasta 0:41 y llegó a 8/9 categorías
  (falta `people`); `clothing` y `body` devolvieron «cuerpo no medido; se conserva
  la foto» en la foto de astronauta; ningún PASS visual firmado. ComfyUI 8188
  quedó escuchando (PID 2844); la UI 7861 no; `active_server.json` apunta a un PID
  muerto de v5.1353. HEAD=v5.1348 con 158 archivos sin registrar; versión en
  código=v5.1355. Sin veredicto de reviewer ni PASS visual, nada se declara HECHO.

- **v5.1355 (2026-11-14) — Pendientes en revisión; estado verificable.**
  Alcance aprobado por el usuario: P1–P15 en orden de prioridad y, para la
  retirada corporal coordinada, «solo lo pedido». P13: `view` aislado ya no
  activa cámara (requiere al menos dos palabras originales); el probe de
  cámara y el de mayúsculas/tildes terminaron con exit 0, así como el probe
  de categoría Cámara (4/4). La política
  `config/editor_selection.json:multi_clause_uncover_enabled=false` impide
  inferir «fully nude» por dos retiradas locales; en selección, el soporte
  se amplía bajo la cabeza si dos cláusulas corporales medidas y la ropa
  segmentada es un fragmento. `probe_editor_selection_body_regime_20261021`
  pasó 13/13 (antes 11/13), incluyendo exterior exacto.
  El probe compuesto usaba una foto blanca 48×32 sin región medible; la
  prueba de fallo de cliente ahora fuerza una región medida y avanza, pero
  el probe completo sigue en FAIL por expectativas antiguas de generación
  sobre sujetos no medidos. Log: `logs/probe_p13_compound_20261114.log`.
  P1: iniciada matriz real de nueve categorías sobre la foto de astronauta
  (`diagnostics/real_editor_category_matrix_20261114.py`); aún sin PASS
  visual. P2: sin repetición con v5.1355 ni PASS visual. P3–P12/P14–P15:
  abiertos. Esta entrada NO declara cierre del conjunto. Queda ejecutar
  lint/build, revisión independiente y evaluar las salidas reales.

- **Relevo histórico superado:** P1 y P2 de versatilidad ya constan aplicadas y
  verificadas con 23/23 contratos (ver spec/03). El relevo vigente está al inicio
  de esta memoria y en `plans/WORKLOG_state_logs_fix_20261106.md`: PASS técnico
  v5.1312; calidad visual y vídeo representativo siguen pendientes.
- **v5.1344 (2026-11-13) — Fix: split_compound rompía "sin X y sin Y".**
  `_split_compound_instruction` dividía "LA MUJER DEBE IR SIN MEDIAS Y SIN ZAPATOS"
  en `['LA MUJER DEBE IR SIN MEDIAS', 'SIN ZAPATOS']`, tratando "SIN ZAPATOS" como
  cláusula independiente sin sujeto. La ruta compuesta medía zona vacía para la
  segunda cláusula, abortaba con "Zona solicitada no medida" y devolvía la foto
  intacta. Fix: la rama `else` del split ahora solo separa por "y" cuando el
  fragmento posterior tiene sujeto+predicado; un fragmento sin sujeto se adjunta
  al anterior como objeto coordinado. Verificado: "sin medias y sin zapatos" →
  `[]` (una sola cláusula), "sin medias y el hombre debe ir sin zapatos" →
  `['...', '...']` (dos cláusulas con sujeto). Además se limpió código muerto en
  `_body_state_clause` (bloque `cover` inalcanzable tras `return ""`).
- **v5.1345 (2026-11-13) — Fix genérico: instrucción no clasificada ya no
  devuelve la foto intacta.**
  Log real: «LA MUJER DEBE ir en topless» / «ir sin sujetador» caían en ámbito
  no clasificado; `_proposed_region_passes` pedía al rewriter una zona, el
  rewriter proponía «chest», que no estaba en el texto → `grounded` vacío →
  «La propuesta no identifica una sola zona» → «Región no medida; no se
  sustituye la foto». Resultado: no se editaba nada.
  Fix (sin listas de palabras ni prendas): cuando la propuesta no coincide con
  la instrucción, el sujeto se extrae con la gramática ya existente
  (`_visual_subject_description`: sintagma determinante / patrón de predicado)
  y **el segmentador mide esa zona en la foto**. Las compuertas existentes
  (cobertura ≤ max_coverage 0,9, crecimiento de sujeto y guardas de cara)
  siguen rechazando una medición implausible: nunca se repinta a ciegas.
  Además, el sufijo de desnudez local era «No shoe, sock, garment or covering…»
  para cualquier prenda (confundía al modelo en peticiones de sujetador):
  ahora es «No garment or covering of any kind may remain there.»
  Verificado en medidas: «go topless», «go without bra» y «must topless» dan
  sujeto «The woman»; «go barefoot» y «be bald» siguen clasificados como antes
  (sin regresión). Pendiente: PASS visual real de una muestra.
- **v5.1345b (2026-11-13) — Color del parche y borde del pegado.**
  Medido con la edición real «LA MUJER DEBE ir sin sujetador» (log
  2026-11-13, la que el usuario vio cumplida pero con el pegote):
  `Armonización de tono Lab shift=[-50.0, 17.8, 6.8]` — el shift L caía
  EXACTAMENTE en el tope ±50 y truncaba el desplazamiento real medido de la
  piel (~-55): la zona seguía con el tono equivocado («no era el color»).
  `Pegado exacto: 0px de feather` dejaba una línea dura en el contorno, que
  se leía como corte y como pegote en la espalda. Fix: tope del shift Lab a
  ±80 (piel y fondo; sigue acotando un anillo de referencia sin contenido) y
  feather mínimo de 1–2 px proporcional al cambio (no al recorte) en el
  contorno del pegado, sin atenuar el interior. No cambia el alcance de la
  zona pegada ni las decisiones medidas de pegado.
- **v5.1345c (2026-11-13) — «sin ropa»/«desnuda» ya no declaran OK sin quitar
  la ropa.**
  Medido en el log: `Validación visual OK: cambio real, sin fantasma` en
  «LA MUJER DEBE IR SIN ROPA» y «…IR DESNUDA» aunque la ropa siguiera puesta.
  Causa: `selected_region_edit.py:2572` activaba la verificación de ropa
  residual SOLO con `zone_label == "zona de ropa medida"`; en exposición
  completa la zona es «persona medida menos la cabeza» y la comprobación nunca
  corría. `_validate_visual_edit` solo mide «cambió algo», no que se cumpliera
  lo pedido. Fix: la verificación de ropa residual aplica a cualquier zona de
  un cambio medido `uncover` (ropa medida u persona menos cabeza), y el
  reintento con zona ampliada dentro del soporte también cubre ese caso. Si la
  ropa sigue medida, el marcador TARGET_RESIDUAL_FAIL impide «OK» y se cae a
  la ruta directa. Sin listas de prendas: referencia = ropa medida en la
  imagen original dentro de la zona pegada.
- **v5.1348 (2026-11-13) — Un estado local sin medición no repinta la ropa
  entera.**
  Medido en el log («IR DESCALZA», 2026-11-13 13:14): CLIPSeg devolvió
  `Máscara VACÍA para 'go barefoot'` y el flujo volvió a `zone = clothing`
  (93105 px = 55 % de 168688 px de persona), repintando el cuerpo con shift
  −58,9 L; la foto empeoró. Dos causas y dos arreglos, sin listas:
  1) un ESTADO es piel por definición y `_request_region_on_subject` lo
     descarta («cae en piel del sujeto»); ahora se mide la consulta sobre el
     sujeto **aceptando piel**, acotada a 1–55 % de la persona;
  2) si aun así no hay zona y la ropa medida cubre ≥50 % de la persona con
     una instrucción que NO es exposición completa, se conserva la foto con
     aviso explícito en vez de repintar el cuerpo entero.
- **v5.1347 (2026-11-13) — Zona local: «descalza» ya no repinta la ropa
  entera.**
  Medido en el log («LA MUJER DEBE IR DESCALZA», 2026-11-13 12:42): la ruta
  descrita eligió `zona de ropa medida = 126662 px` sobre una persona de
  145346 px (87 %) y repintó el cuerpo entero para quitar los zapatos; el
  resultado empeoró. Causa: `_requested_removal_target` no devuelve objeto
  nombrado para un estado local («descalza», «topless», «sin sujetador»), así
  que se caía al `zone = clothing` de toda la ropa medida.
  Fix (sin listas): si `full_body` es falso, la dirección es `uncover` y no hay
  objetivo nombrado, la zona la mide el segmentador **sobre la propia
  instrucción** con `_request_region_on_subject` (la misma medición que usa la
  ruta directa). Si no mide una parte plausible, se conserva el
  comportamiento anterior. Verificado: «go barefoot» y «go topless» no son
  exposición completa; «without clothes»/«naked» sí lo siguen siendo.
- **v5.1346 (2026-11-13) — Cierre de la tanda de versatilidad/cumplimiento.**
  Consolida bajo un solo número los arreglos registrados arriba como 5.1345,
  5.1345b, 5.1345c y 5.1345d: instrucción no clasificada medida por sujeto,
  sufijo uncover genérico, tope Lab ±80, feather mínimo del contorno,
  verificación de ropa residual en exposición completa y `step_cap` forwardeado
  con 24 pasos en cuerpo/uncover. Sin listas de prendas ni censura; ninguna
  categoría de la versatilidad queda estrechada.
  Bug medido: `_vision_zone_pass` solo reenviaba `step_cap` cuando la receta
  traía `kontext_img2img_denoise`; en la ruta corporal con visión (LongCat
  Turbo) el `steps` de la receta se ignoraba y el cliente caía siempre a 12
  (log «sin ropa»/«desnuda»/«sin sujetador»: `1.9 s/paso ×12`). El `step_cap`
  se reenvía siempre (cliente acota 4..28) y la receta corporal sube a 24
  pasos con denoise 1.0. Es un cambio numérico de esfuerzo, no de ruta: no
  añade vocabulario ni estrecha ninguna categoría de la versatilidad.
- **v5.1349 (2026-11-13) — Selector «Tipo de edición» en el Image Editor.**
  El desplegable (`ui/tabs/img_editor/ui.py`) es **prior de ruta + ayuda en la
  caja**: con `auto` el comportamiento es idéntico al de v5.1348. Con categoría,
  `_apply_scope_hint` eleva su eje medido salvo que el prompt mida una señal muy
  alta de otro eje (`pose`, `body_intensity >= 0.60`, `scenario_strength >= 0.60`
  nuevo campo de `_prompt_authority_facts`, o POV explícito sin pose en la misma
  cláusula); si hay conflicto se ignora el selector y se escribe en log. No hay
  listas de palabras ni se prohíbe ningún prompt.
  Motor: `_PINNED_ENGINE_BY_HINT` fija `longcat` (9 categorías) y `qwen_camera`
  (cámara). El pin se aplica **solo al modelo** (`engine_for_model`) porque pasar
  `engine="longcat"` desactiva `auto_requested` y perdería las rutas medidas.
  Cámara fuerza el POV explícito y, si el prompt mide una señal fuerte de otro
  eje, el selector se ignora (se registra). Con la categoría Cámara, un resto
  sin región medida ya no tira abajo el reencuadre: se conserva el resultado de
  la pasada de cámara con aviso (antes se devolvía la foto original); sin
  categoría el comportamiento anterior no cambia.
  Bug de log corregido: «CAMBIAR DE POSTURA … MOSTRANDO EL CULO A LA CÁMARA»
  activaba Qwen Cámara por el token «cámara» (240 s, `score=0.00`);
  `_explicit_viewpoint_request` ahora cede cuando la misma cláusula mide postura
  (`camera_counts_as_pose=False`), y «cambia el POV» sigue entrando.
  Modo Calidad: checkbox «🧼 Solo acabado» (`only_finish`) aplica los pasos
  marcados y **el prompt no edita**; con Modo Calidad y un prompt que no mide
  ningún eje se ejecuta el acabado en vez de «Zona local no medida» (logs
  «MEJORA LA IMAGEN, REALISMO, DESPOSTERIZA»: no hacía nada). La zona local se
  sigue midiendo: solo aplica sin máscara pintada y sin selección visual.
  Endpoints API sin cambios (8/9/13/14); `edit_manual_region` pasa a 17 entradas
  (solo escritorio). Verificado en CPU con `diagnostics/
  probe_editor_category_selector_20261113.py` (20/20),
  `probe_editor_category_camera_20261113.py` (4/4) y
  `probe_editor_only_finish_20261113.py` (5/5), UI con Gradio 6.13
  (`edit_manual_region` = 17, H-EDIT-SELECT 17/17) y
  `probe_editor_auto_appearance_recovery_20261104.py` OK. Fallos previos, ya
  presentes en HEAD (comprobado con el manager de HEAD): `view` suelto cuenta
  como cámara en `probe_editor_viewpoint_dispatch`, y 2/13 en
  `probe_editor_selection_body_regime_20261021` + 1 aserción de fallo de cliente
  en `probe_editor_compound_fast_dispatch`. Pendiente: PASS visual real por
  categoría (AGENTS.md).

  Revisión previa al commit (5 hallazgos, todos corregidos): (1) la categoría
  `clothing` conserva la dirección medida (`cover` no se invierte; si el prompt
  ya mide cuerpo sin dirección se deja en `change`); (2) la red de acabado no
  se ejecuta si hay POV explícito o forzado; (3) medidas del prompt memoizadas
  por petición (`_SCOPE_HINT_MEASURED`) y `camera` ya no reeleva hechos; (4) la
  lista del selector vive solo en `ui.CATEGORY_CHOICES`; (5) eje y hechos de
  cada categoría salen de una única tabla `_SCOPE_HINT_EFFECTS`.

- **v5.1350 (2026-11-13) — diagnóstico del log de UI (Image Editor).** Tres
  peticiones reales salieron con `acabado=True` en todas (`[ImgEditor] Solo
  acabado: el prompt no edita`), `categoria=auto` y el prompt ignorado por
  diseño; el checkbox 🧼 estaba marcado. Verificado con Gradio 6.13 real que
  `gr.State` sí se pasa a la función y que `edit_manual_region` recibe sus 17
  argumentos en orden: no hay desplazamiento que active `only_finish`. La línea
  «Solicitud recibida» ahora imprime `acabado=...` para no depender de la línea
  siguiente. Sin cambios de ruta ni de procesamiento.

- **v5.1351 (2026-11-13) — «descalza / sin medias» no cumplía (log 22:04).**
  Causa raíz medida: `_request_region_on_subject` descarta la zona de un ESTADO
  porque es piel (`skin_share>=0.50` y `part<0.08`), y el respaldo v5.1348 del
  estado consultaba solo las frases completas («The woman must go barefoot»,
  «la mujer debe ir descalza»), que CLIPSeg devuelve VACÍAS; el predicado
  «go barefoot» sí mide (1.7 % del lienzo) pero nunca se usaba. Resultado: la
  zona caía en «zona de ropa medida» (751073 px) y se repintaba el torso en
  vez de los pies. Ahora el respaldo mide los predicados de
  `_visual_request_queries` (sin listas) y la zona sale del calzado medido.
  Verificación: el reintento residual ya no pierde el chequeo
  (`check_clothing_residual` se pasa también a los reintentos) y un estado
  medido se valida por PIEL en la zona escrita (`check_state_skin`: sube
  ≥0.10 y llega a 0.35; si no, TARGET_RESIDUAL_FAIL → no se declara
  «Validación visual OK»). Probe nuevo
  `diagnostics/probe_editor_barefoot_state_20261113.py` 5/5 (zona = calzado
  100 %, torso 0 %; fallo y pase de la verificación por piel). Sin cambios en
  `auto` ni en las rutas de selección. Pendiente: PASS visual real del caso
  «descalza y sin medias»; el umbral de residuo de objetivos sigue en 40 %
  (política v5.1311 medida) y el primer pase aceptó 16 %.

- **v5.1352 (2026-11-13) — el selector allana la edición (plan aprobado).**
  Además de elevar el eje y fijar el motor, la categoría desemboca en la zona
  MEDIDA de su eje: `clothing` → estado/predicado/prenda medida (v5.1351);
  `features` → `_request_region_on_subject` + `_complete_request_surface` de
  la cara/pelo nombrados (ya existía); `add`/`remove` → objeto medido
  (`_measured_region_passes`/`_measured_target_zone`); `pose` → persona
  completa; `camera` → POV forzado; `people` → add/remove medido o escenario.
  `background` NO se convierte en zona local: el código consolidado
  («Un pedido de escenario NO es una región local: manda la regeneración
  completa, sin ancla») manda sobre el plan, así que conserva la regeneración
  global y el selector solo fuerza el eje.
  Verificación nueva: una prenda que se lleva puesta (≥80 % de su zona dentro
  de la persona) se valida por PIEL descubierta, no por otra prenda: el
  residuo del 16 % de «stockings» ya no se acepta como OK; se reintenta y, si
  no hay piel, se declara fallo. Probe
  `diagnostics/probe_editor_barefoot_state_20261113.py` 7/7 (estado, prenda
  puesta, fallo/pase por piel). Batería en verde: selector 20/20, cámara 4/4,
  solo acabado 5/5, H-EDIT-SELECT 17/17, UI skip OK, `ruff` OK.
  Pendiente: PASS visual real por categoría y del caso «descalza y sin medias».

- **v5.1353 (2026-11-13) — revisión a fondo del selector (3 defectos).**
  (1) La cadena de precedencia de `_apply_scope_hint` borraba el eje elegido
  cuando el prompt medía OTRO eje débil (<0.60): `features`/`background` con
  un `body` de 0.41 se quedaban sin efecto aunque el log dijera «eje elevado».
  Ahora se limpian los ejes medidos débiles antes de aplicar el efecto del
  selector (las señales fuertes siguen ganando y se registran). (2) El respaldo
  de estado medía los predicados sin dividir la conjunción: «go barefoot and
  without stockings» no lo mide CLIPSeg; ahora se divide por cláusulas
  (`,`/`y`/`and`) y se mide el predicado de cada una. (3) El reintento residual
  sin objetivo escribía la zona AMPLIADA (v5.1323 decía lo contrario): ahora
  el recorte grande solo da contexto y se escribe lo medido.
  Probes: barefoot/estado 8/8 (incluye estado compuesto y prenda puesta),
  selector 22/22 (incluye features/background sobre cuerpo débil), UI skip OK,
  H-EDIT-SELECT 17/17, `ruff` OK. Pendiente: PASS visual real de
  «descalza y sin medias» y de cada categoría.

- **v5.1354 (2026-11-13) — selector sin hardcoding: vale cualquiera de las 9.**
  Auditoría del diff: ninguna frase de usuario en la lógica (solo en comentarios
  de evidencia). La tabla `_SCOPE_HINT_EFFECTS` es la DEFINICIÓN de la opción
  (categoría → eje de hechos), no una lista de palabras del prompt: cada opción
  eleva solo su eje y todo lo demás sale de la medición. Se quitó la dirección
  inventada (`clothing` forzaba «uncover»): la dirección e intensidad salen
  siempre de `_prompt_authority_facts`, así «hazlo bonito» + Ropa queda en
  `change` y no desnuda; «con ropa puesta» conserva `cover`. Se quitó también
  el pin redundante de motor para 8 opciones (`_PINNED_ENGINE_BY_HINT` solo
  `camera → qwen_camera`, lo único que la ruta no elige sola).
  Probado con todas las opciones: `probe_editor_category_selector_20261113.py`
  31/31 (bucle sobre la tabla: cada opción eleva exactamente su eje; auto no
  eleva nada; fuertes medidas siguen ganando). Estado/prenda 9/9 (zona unida
  objetivo+estado por cláusulas, verificación por piel sobre el parche del
  motor sin armonizar), cámara 4/4, solo acabado 5/5, H-EDIT-SELECT 17/17, UI
  skip OK, `ruff` OK. `on_category_change` comprobado con las 10 entradas:
  solo Cámara fija motor; el resto deja el motor en manos de la ruta.

## Pendientes vigentes (2026-11-13, al cerrar v5.1354)
Detalle completo y criterio de cierre: **`docs/PENDIENTES_2026-11-13.md`**.
Diseño/contratos del selector: **`docs/IMG_EDITOR_SELECTOR_CATEGORIAS_2026-11-13.md`**.

1. **PASS visual por categoría** (bloqueante): ninguna de las 9 tiene validación
   visual real; los probes son de despacho/píxel. Mínimo una foto por categoría.
2. **«descalza y sin medias»**: lógica corregida (zona unida objetivo+estado,
   verificación por piel), sin PASS visual. Log esperado: «objetivo medido +
   estado … → zona unida» y «Verificación del estado … piel X% → Y%».
3. **Lentitud**: 72,8 s ComfyUI + 25 s modelo + 52 s texto (frío) y 60–215 s por
   pasada LongCat; cada reintento ~60 s y el fallback otra pasada (4–7 min si
   falla). Falta medir un tope de reintentos sin aceptar resultados falsos.
4. **Residuo de objetos no puestos**: sigue en 40 % (v5.1311); calibrar con fotos.
5. **`background`**: sin zona local por decisión consolidada (regeneración global).
6. **`people`, `features`, `add`, `camera`**: sin verificación medida específica
   (solo «cambió algo» o aviso de no verificado).
7. **Calibrar nuevos umbrales** (`≥80 %` prenda puesta, piel `≥55 %`/`+10`,
   unión `≤60 %`) con fotos reales variadas.
8. **Armonización**: la verificación usa el parche sin armonizar; la imagen
   entregada lleva shift (medido −29 L) — comprobar que no se aleja del esperado.
9. **Prompt genérico + categoría con Modo Calidad**: el hint eleva eje y evita la
   red de acabado; decidir precedencia con medición.
10. **Probes con fallo previo (ya en HEAD)**: `view` suelto como cámara,
    2/13 en body regime, una aserción de fallo de cliente en compound dispatch.
- **2026-11-14 — Revisión de pendientes en curso (petición: todos por prioridad).**
  Documento rector: `docs/PENDIENTES_2026-11-13.md` (P1–P15).
  Estado inicial: `git status` ya tenía numerosos cambios ajenos a este turno;
  no se han revertido. La UI 7861 y ComfyUI 8188 no escuchan; el registro
  `logs/active_server.json` apunta a v5.1353 pero su PID no aparece activo.
  P1/P2 aún no tienen PASS visual nuevo. P13 reproducido con `python -m`:
  `probe_editor_viewpoint_dispatch` falla por `view` suelto;
  `probe_editor_selection_body_regime_20261021` da 11/13, con fallos en
  `garment_fragment_widens_to_the_person` y
  `multi_clause_uncover_writes_full_body_suffix`;
  `probe_editor_compound_fast_dispatch` falla en la aserción de que un fallo
  de cliente devuelve `None`, pero la ruta devuelve «región no medida» antes
  de llamar al cliente. Diagnóstico abierto; ninguna categoría se declara
  resuelta por estos probes. Continuar con reproducción real, correcciones
  acotadas, revisión y verificación.
