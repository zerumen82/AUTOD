# WORKLOG — rescate de frames sin swap en vídeo (kps degenerados y máscara), 2026-11-07

Petición del usuario: revisar el lote de 9 vídeos del 07/11 (sin errores de
render) por qué quedaron frames sin cara swapeada y arreglarlo — «Ambos
arreglos» / «SI PROCEDE». Al cerrar: bitácora + MEMORY + commit + push.

## Diagnóstico (logs reales del lote)

- `logs/process_20261107_031047.log` (vídeo 3) y `process_20261107_070133.log`
  (vídeo 9), más los contadores de los 9 archivos.
- 334 frames en el gate de coherencia kps↔bbox (`[ALIGN_REJECT] kps
  incoherentes`, eye_dist/bbox_w 0,003–0,050 frente a 0,40–0,45 normal):
  vídeo 3 = 206/722 (28,5 %), vídeo 9 = 61, vídeos 1/2/6 = 49/14/12.
- 227 frames en el gate 2 de máscara (`[MASK_SKIP] Contorno medido
  insuficiente`): 219 en el vídeo 9, concentrados en 494-2944 (perfil y
  primer plano). 0 rescates de hold y 0 `MASK_SAFETY` en todo el lote.
- El vídeo 3 ya no está en `.autodeep_temp`; solo queda el 9
  (`VID_20261027_174936_599.mp4`). `output\` vacío: los mp4 entregados se
  borraron, la evidencia es de log.

## Fix A — geometría reconstruida en vídeo (ProcessMgr.py)

- En ambos puntos de `[ALIGN_REJECT]` de `_process_face_swap_v21`, en vídeo
  se intenta `_recovered_coherent_kps(source, target, person_idx)` antes de
  caer en `_swap_from_saved_reference` (que en vídeo no tiene refs → frame
  sin swap):
  1. reusa el último kps sano de ESTA persona en ESTA vídeo
     (`_last_good_face_geom`) trasladado al centro y escalado al bbox actual
     del tracker — misma pose → proporciones sanas;
  2. si no hay histórico (arranque), sintetiza los 5 puntos del source
     dentro del bbox trackeado;
  3. si nada pasa el gate (`landmarks_fit_bbox` + 0,05 < eye_rel < 1,20),
     devuelve None → fallback original: nunca se empeora (contrato
     v5.1202 «mejor sin swap que una cara deformada»).
- `_remember_good_face_geom` se guarda en el camino de ÉXITO (junto a
  `_last_swap_bbox`), filtrado por el mismo gate; en la rama «landmarks
  fuera del bbox» `_kps_chk` se re-valida con los kps fijos para que el
  segundo gate (eye_rel) no rechace lo que acaba de sanearse.
- Etiqueta nueva: `[ALIGN_RECOVER]` (con eye_rel antes→después).

## Fix B — hold temporal en el gate 2 de máscara (ProcessMgr.py)

- Antes de `_register_mask_skip()` en «Contorno medido insuficiente» se
  replica el hold del gate 1 (línea ~6090): `video_path && prev_full &&
  last_bbox && _should_hold_previous_swap()` → `_faded_hold_blend(0.88)`
  ≥ 0,10 → `_hold_previous_face_region(...)`. Sin historia/corte duro/fade
  agotado → skip como hasta ahora (solo entonces se cuenta el skip).
- Etiqueta nueva: `[MASK_HOLD] Contorno medido insuficiente...` (sin
  throttling: máx ~3 por episodio y es la evidencia del rescate).

## Verificación

- Probe nuevo `diagnostics/probe_faceswap_video_reject_recovery_20261107.py`
  con datos REALES del lote (bbox [290,553,353,604] eye_rel 0,013 del vídeo
  3, bbox del vídeo 9, región de hold con las tres funciones reales):
  **14 OK, 0 FAIL**. El probe destapó un agujero real: los kps degenerados
  CABEN dentro del bbox, así que `landmarks_fit_bbox` solo los aprueba — el
  filtro `0,05 < eye_rel < 1,20` del histórico era obligatorio (corregido).
- Render real acotado `probe_faceswap_video_recovery_render_20261107.py`
  (vídeo 9, frames 0-900; baseline en ese tramo: 55 REJ + 35 MSK; inyecta la
  referencia de destino detectada en el frame 0 — misma cara ancla del
  lote, bbox (413,380,482,482) a 2,5 px — y una cara origen de testdata):
  **CERRADO, PASS** (1444 s de render, ~0,2 fps, VRAM liberada al final).
  - Conteos: `{'rej': 55, 'recover': 55, 'msk': 4, 'hold': 13}`. Los 55
    frames que en el lote quedaron sin swap ahora tienen `[ALIGN_RECOVER]`
    y el píxel cambió dentro del bbox del tracker para ese frame (ej.
    frame 59: bbox (525,396,591,497), 1398 px > 12). MASK_SKIP
    35 → 4 (solo 4 episodios que agotaron el fade) y 13 `[MASK_HOLD]`.
  - El probe falló primero con un falso negativo: el bbox del tracker SE
    MUEVE (frame 59 ya está lejos del ancla del frame 0), así que la
    comprobación de píxeles usaba el bbox estático. Se corrigió
    `pixel_check` a usar el bbox real del log por frame (TRACK N ↔
    vídeo N-1) y se re-ejecutó con `--skip-render`: PASS.
  - Evidencia: `diagnostics/out/video_recovery_render_20261107/`
    (`report.json`, `VID_20261027_174936_599_0_900.mp4`).
- Regresión verde tras los cambios: align/mouth 9 OK, ellipse 5 OK,
  temporal 5 OK, reference_rescue OK, batch_cancel 30 OK,
  pipeline_controls 10 OK, py_compile de los 116 .py cambiados, ruff
  (E9,F63,F7,F82).
- `probe_faceswap_pipeline_controls_20261017` tenía 3 FAIL **preexistentes
  y ajenos** a estos cambios: el `manager_class()` del probe extrae métodos
  por whitelist AST y no incluía `_mouth_oral_domains` /
  `_clip_occlusion_to_mouth` (añadidos en v5.1288); el `except` interno de
  `_mouth_occlusion_evidence` convertía el AttributeError en evidencia
  0,0 → None. Whitelist actualizada + la aserción `connected → None`
  alineada al contrato deliberado de v5.1288 (rescate recortado a la boca;
  la ambigüedad real de dos componentes sigue absteniéndose): 10 OK.

## Versión y cierre

- Incremento v5.1318 → v5.1319 en `run.py`, `roop/core.py`, `AGENTS.md` y
  `spec/00-PRINCIPIOS.md`; entrada en `MEMORY.md` y en
  `spec/02-REQUISITOS.md` (H-FS-007).
- La UI debe reiniciarse para que cargue el ProcessMgr nuevo; no se
  reinició ningún servicio (ComfyUI en 8188 activo, renders propios hechos
  en proceso aparte sin tocar la cola de la UI).

## Límites

- Vídeo 3 (peor caso) no reproducible: su entrada se borró. Verificación
  sobre vídeo 9 + probes con datos de ambos.
- En fix B el fade del hold dura ~4 frames (v5.1074c): episodios largos de
  máscara insuficiente siguen saltando frames cuando se agota el fade; el
  objetivo era no dejar de probar el hold, no prometer swap continuo.
- Sin PASS visual del usuario todavía: toca mirar el siguiente lote en la
  UI (y el render acotado si se quiere revisar frame a frame).
