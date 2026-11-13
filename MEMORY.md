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