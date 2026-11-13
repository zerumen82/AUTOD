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