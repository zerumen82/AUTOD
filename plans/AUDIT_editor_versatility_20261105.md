# Auditoría de versatilidad del Image Editor — 2026-11-05

## Petición y alcance autorizado

El usuario pide revisar a fondo si v5.1308 satisface añadir/quitar personas y
cosas, posiciones/posturas, ropa/apariencia, ángulo de cámara y POV, incluyendo
instrucciones compuestas. Se auditan rutas existentes, contratos y evidencia
visual; el requisito permanente está al principio de MEMORY.md.

## Plan de revisión

1. Contrastar MEMORY/spec y Git con el código actual; conservar cambios locales.
2. Revisar por separado Auto/apariencia, cámara/POV, adición/retirada,
   posiciones/posturas y cumplimiento de todas las cláusulas.
3. Ejecutar suites CPU pertinentes y reproducir defectos sin inferencia;
   distinguir dobles de diagnóstico de ejecuciones reales del modelo.
4. Contrastar muestras reales existentes con sus límites de validación.
5. Obtener revisión independiente y resultado del tester; registrar hallazgos
   y mejoras concretas antes de decidir cambios de alcance/diseño.

## Límites operativos

- No instalar modelos/dependencias, hacer commits, restaurar commits completos,
  reiniciar UI/backend ni interrumpir renders del usuario.
- Diagnóstico no equivale a mejora implementada ni a certificación visual.
- Sin cambio de versión por esta auditoría documental. Si se cambia ejecución,
  actualizar versión y bitácora y verificar el cambio según AGENTS/spec.

## Estado

Auditoría en curso. Reviewer revisa rutas estructurales/compuestas; tester
ejecuta contratos CPU; root revisa Auto, cámara/POV y evidencia visual existente.

## Hallazgos confirmados en código y reproducciones CPU

No son PASS/FAIL visuales nuevos del modelo. Las máscaras sintéticas y los dobles
de inferencia, cuando se usan, están declarados en las reproducciones.

1. **P1 — POV pierde el observador.** Las funciones reales reciben «First person
   POV through the man's eyes» y «First person POV through the woman's eyes»;
   ambos producen `<sks> front view eye-level shot medium shot` y ningún cambio
   restante. La ruta de cámara envía solo esa pose. Lo mismo sucede con una
   petición sobre el hombro. Evidencia: `root_semantic_observations.json`,
   `qwen_camera_comfy_client.py::camera_pose/remaining_edit` y
   `img_editor_manager.py` alrededor de 7485–7500. Falta conservar el observador
   y distinguir primera persona de una órbita alrededor del sujeto.
2. **P1 — Eliminación ordinaria expande al exterior completo.**
   `_build_object_removal_mask` calcula `dark = arr < 50` sobre la MÁSCARA,
   luego suma ese exterior a la zona editable. Reviewer reproduce una máscara
   binaria de 1.600 px convertida en 38.400 px (100 %). El método además no
   recibe el objetivo pedido: consulta categorías genéricas. Hay que corregir
   objetivo y soporte antes de acreditar retirada de objetos. Tiene una llamada
   en la ruta estructural tardía; no se atribuye a todo Auto sin comprobar ruta.
3. **P1 — Retirada de persona puede excluir al propio objetivo.**
   `_closed_structural_zone` resta todas las personas si ocupan menos del 55 %
   de la caja. Reviewer reproduce 4.600 px del objetivo reducidos a cero mientras
   habilita 15.000 px de fondo. Debe distinguir objetivo de vecinos protegidos.
4. **P1 — Adición compuesta sin zona puede quedar fuera del pegado.** El bucle
   compuesto anota cláusulas sin región, pero `structural_add` no cuenta como
   cláusula local que exige cobertura. Si otra cláusula sí se mide, se muestrea
   y pega únicamente esa unión. El modelo puede recibir «hazla calva, añade una
   persona a su lado» con un soporte que solo permite escribir la cabeza.
5. **P1 — Puntos entre órdenes eluden la protección de compuestos de Auto.**
   `_split_compound_instruction` no reconoce el punto como separador de órdenes.
   Reviewer reproduce `The woman must be bald. The wall must be blue.` con
   split vacío y elección focal Kontext; con coma, conserva dos cláusulas y
   LongCat. La pared puede quedar fuera del recorte. Este hueco afecta a la
   recuperación de Auto de v5.1306; no está resuelto por la plantilla v5.1308.
6. **P2 — Retirada compuesta pierde el objetivo directo de la individual.**
   `_compound_clause_zone` no transmite `structural_remove` a
   `_measured_region_passes`. Se pierde la extracción directa del objetivo y la
   aceptación de objetos pequeños por debajo del suelo genérico del 4 %.
7. **P2 — Cobertura add/remove más débil de lo registrado.** Los probes
   `probe_editor_object_removal_20261102.py` y
   `probe_editor_addition_mask_20261102.py` contienen comprobaciones constantes
   y búsquedas de texto. Un 7/7 o 4/4 de esos probes no acredita localización,
   fuga, protección de vecinos ni accesibilidad de esa rama desde Auto.

## Riesgos que aún requieren verificación específica

- **Ropa:** «the woman must wear a blue coat» / «la mujer debe llevar un abrigo
  azul» mide `surface_recolor=True`, `body=False`. Es riesgo de tratar una prenda
  nueva como cambio de color. El prompt se conserva: no afirmar un fallo visual
  solo por esta clasificación. Hace falta probar forma/corte y zona de prenda.
- **Apariencia física:** «larger eyes» mide cuerpo, no apariencia focal; la
  recuperación de Auto no cubre todas las características físicas por igual.
- **Posición/postura:** tanto mover a la mujer como mover una silla hacia la
  izquierda miden pose; falta acreditar el destino y la conservación del resto
  con muestras reales. Desanclar el marco por sí solo no demuestra cumplimiento.
- **Compuesto sin cambio:** persiste el caso ya documentado del bloque que
  detecta no-op, ejecuta `pass` y retorna enseguida. No contarlo como fallo nuevo.

## Evidencia visual inspeccionada en esta revisión

- Auto productivo v5.1308: `appearance_reference_20261023/1793866121580301900/`.
  Calvicie visible, sin bloque amplio, con brillo/textura lisa y vello corto.
- Grace candidata controlada: `appearance_reference_20261023/1793865362633754700/`.
  Conserva gafas/rasgos mejor que el baseline, pero retira gorra y cambia detalles.
  No presentarla como segunda ejecución nativa nueva de producción.
- Histórico `coffee_remove_spoon_v3/compare_real.png`: la cuchara sigue visible.
  No es una ejecución nueva de v5.1308; confirma por qué el pendiente visual sigue.
- Histórico `refine_recipe_20261104/neutral_without_expansion/compare_real.png`:
  restos/orla visibles. El posterior Auto focal mejoró el caso; no confundir ambos.

## Fuente primaria contrastada para cámara

[Ficha oficial del LoRA de fal](https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA)
(consultada 2026-11-05): enumera 8 azimuts, 4 elevaciones y 3 distancias con
prompt de órbita. Inferencia de auditoría: ese contrato no expresa por sí solo
qué persona observa desde sus ojos; conservar semántica de POV exige más que
seleccionar la combinación frontal. No se propone instalar otro modelo.
