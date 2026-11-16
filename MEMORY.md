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

