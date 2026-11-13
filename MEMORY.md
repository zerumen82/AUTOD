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