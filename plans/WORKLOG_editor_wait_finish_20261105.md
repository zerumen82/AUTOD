# H-EDIT-WAIT-FINISH — plan autorizado y bitácora de tareas

Petición: «se ha colgado»; el usuario acepta arreglar seguimiento, revisar
acabado previo a reintentos y reproducir lentitud con modelos actuales.

Contexto: comfy_progress.py, selected_region_edit.py y generate_intelligent;
MEMORY/documentos separan servidor1189,720s de UI39:12. No atribuir la
discrepancia a VRAM o bloqueo sin medir; no repetir descarga ya ejecutada.

Causas contrastadas: tiempos cliente de pared sensibles a saltos, falta de
latidos con fase constante, publicación dependiente del sondeo HTTP; acabado
TOP de candidato que luego rechaza la validación. Causa de inferencia lenta
aún abierta, presión/offload solo hipótesis.

- [x] Alcance aprobado, contexto leído y requisitos escritos antes de implementar.
- [x] Probes CPU reproducen reloj/progreso y acabado antes de validación (RED).
- [x] Corrección mínima: monotónico/telemetría/config y puerta previa al acabado.
- [x] Probes nuevos GREEN y regresiones pertinentes, lint/compile/build.
- [x] Revisión independiente APPROVE; corregir hallazgos y re-revisar.
- [x] Reproducción neutral con servicio libre y modelos instalados; tester PASS
  acotado sobre tiempos/artefactos sin confundirlos con aceptación visual.
- [x] Versiones/spec/MEMORY y pendientes; entrega con límites explícitos.

Impacto esperado: progreso comprobable y ahorro del acabado de un rechazo.
Se mantiene validación final y rutas, sin estrechar versatilidad. Se registrará
cualquier deuda/regresión; no se cambia motor, calidad, muestreo, arranque,
dependencias, FaceSwap, Android ni decisiones semánticas.


## 2026-11-05 — H-EDIT-WAIT-FINISH: verificación técnica final v5.1310
- Usuario autorizó el alcance. Monotónico, WS antes de HTTP, latidos configurables
  y telemetría; validación antes de TOP, preservando validación final y recetas.
- Tester CPU: 57 unittest,13régimen,28WS+cleanup y11config inválidas PASS.
  Generate:110checks de adherencia y6controles PASS (logs separados).
- Ruff completo de2módulos/3probes, py_compile y lint crítico run/core verdes.
  Build launcher aislado exit0 fix/build_1310.log. Reviewer APPROVE técnico;
  tester PASS técnico acotado en fix/tester/cpu_verdict.json y real_verdict.json.
- Real neutral fix/real_neutral_20261105_220448_580169: LongCat instalado,
  12pasos/cfg1,1088x960, un prompt; servidor74,892s,espera75,781s,
  generación78,781s;75callbacks/gapmáx1,25s/HTTPmáx0,250s/callbackmáx0,109s.
  Seis muestras GPU, cero errores y cola0/0; seis hashes coinciden tras render.
- FAIL visual de conservación: taza azul sí, pero cambia plato/encuadre/café/
  cuchara y añade vapor. No acredita calidad general ni caso zapatos.
- Latidos cooperativos: HTTP/callback bloqueados pueden retrasarlos. Prueba con
  callback JSONL no acredita entrega Gradio. No se explica el39:12 histórico
  ni se atribuye aceleración del modelo al parche. Fuentes primarias sobre
  offload en fix/research.md sin cambios de flags. No se instaló ni reinició,
  no se hizo commit/push; reabrir app carga cambios.
- Toda evidencia: diagnostics/out/editor_stall_20261105/. Reviewer/tester
  independientes; detalles visuales en fix/tester/real_visual_and_provenance.json.

