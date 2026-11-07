# Revisión general de color y posición — 2026-10-26

Alcance autorizado: auditoría general con ejemplos no explícitos. No se han
modificado código ni parámetros de producción, ni reiniciado procesos,
instalado dependencias o ejecutado inferencia. El workspace ya contenía
modificaciones de otras sesiones; esta revisión no las aprueba.

## Hallazgos comprobados

1. **El control opcional de calidad no mide fidelidad cromática.**
   `top_tier_finish.py:367` mide nitidez y clipping; los pasos color/luz
   consultan esas dos métricas (`774`, `1018`). Una carta sintética con
   canales R/B intercambiados pasa la puerta habilitada: clipping 0 en ambas,
   nitidez 10,57 → 17,13, sin aviso. Esto demuestra una limitación del control,
   no un intercambio de canales en producción. La conversión real RGB/BGR
   del finisher pasa un round trip exacto. El registro del usuario indica
   `calidad=False`: este acabado opcional no explica por sí solo aquel caso.

2. **El pegado conserva las coordenadas, pero acepta proporciones distintas.**
   `_integrate_local_generation` pega en `(x1,y1)` (`img_editor_manager.py:3125`),
   pero redimensiona ambos ejes directamente al recorte (`2981`). En una
   entrada sintética 160×40 destinada a un recorte 320×240, el círculo queda
   como una elipse 48×144, proporción 3:1. El pegado de identidad es exacto
   y cambia 0 píxeles fuera de la máscara. Las coordenadas de pegado correctas
   no certifican que el contenido generado conserve su posición o forma.
   No se ha demostrado que este riesgo explique la foto del usuario.

3. **La preparación reversible de referencia tiene contratos verdes.**
   `probe_editor_reference_canvas_20261019.py`: 7/7 PASS, incluyendo escala
   uniforme, padding reversible, marcadores y rechazo de tamaños inesperados.
   Esto valida este helper; no acredita que todas las rutas lo utilicen ni
   que el modelo respete la postura.

## Evidencia ejecutada

- Nuevo probe: `diagnostics/probe_editor_color_position_review_20261026.py`.
- Ejecución CPU: exit 0; log `logs/editor_color_position_review_20261026.log`.
- Métricas y PNG con nombre fresco:
  `diagnostics/out/color_position_review_20261026/1793041192759990700/`.
- Inspección visual del artefacto `circle_composed.png`: elipse vertical.
- Contratos existentes: reference_canvas 7/7 PASS; paste_guard PASS con fuga
  0; recolor_surface_zone PASS, 0 fallos (cobertura de pared 88 %, otra pared
  100 %, sin atribuir segmentación simulada a inferencia real).
- Ruff completo del nuevo probe: `All checks passed!`.
- `python -m py_compile` del nuevo probe: exit 0. No hay build empaquetado:
  no se modificó producción ni un artefacto distribuible.
- sd-reviewer independiente `review_color_position`: **APPROVE** al probe y
  diagnóstico CPU, sin aprobar resolución visual ni el resto del workspace.

## Límites y continuación

No se declara corregido el color ni la posición. Falta un par original/resultado
no explícito del mismo caso y un render real para atribuir la causa al motor,
al reescalado o al pegado. Los riesgos generales quedan abiertos en spec/03.
La revisión recomienda medir ambas cosas por separado antes de decidir una
corrección o modificar parámetros del modelo.
