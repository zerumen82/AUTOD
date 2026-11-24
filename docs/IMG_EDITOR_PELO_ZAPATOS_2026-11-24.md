# Image Editor — pelo en el calvo y zapatos aceptados sin quitarse (2026-11-24)

## Reporte del usuario
- «EL HOMBRE DEBE SER CALVO» → se dejó pelo en la cabeza.
- «LA MUJER DEBE IR SIN ZAPATOS» → las botas siguen puestas.
- Pregunta: ¿por qué el domingo o el sábado sí iba?

## Qué cambió desde el domingo
`git log` del editor entre el 22 y el 24 de noviembre: **un solo commit tocó el
Image Editor**, `9357685` (v5.1444, 24/11 11:07). Cambios:

| Qué | Antes (domingo) | v5.1444 |
|---|---|---|
| `local_appearance_min_steps` | 20 | **28** |
| `local_appearance_max_work_side` | sin tope | **1536** |
| `_head_zone_for_subject` | unión de todas las cabezas ∩ persona | una cabeza elegida por frase/`_leading_face` |

Entrega del domingo 23/11 23:52: `img_editor/edit_1795474328.png` — cabeza casi
pelada (revisada). Entrega del 24/11 12:57: `img_editor/edit_1795521434.png` —
resto de pelo glossy en la coronilla (revisada).

## Causa medida — calvo
Log del 24/11 (tres pasadas de apariencia, todas con la imagen de trabajo
acotada a 1536 px de lado):

```
[ImgEditor] Cambio medido sobre el sujeto: maximo 208 niveles, umbral Otsu=86,
            respuesta en 40.5% del sujeto
[ImgEditor] Cambio medido sobre el sujeto: maximo 89 niveles, umbral Otsu=62,
            respuesta en 2.9% del sujeto
[ImgEditor] Cambio medido sobre el sujeto: maximo 55 niveles, umbral Otsu=55,
            respuesta en 2.0% del sujeto
```

La respuesta sobre el sujeto se hunde en las pasadas 2 y 3 y el pegado escribe
solo el cambio medido: lo que el modelo no cambió (el pelo de la coronilla) se
conserva de la foto original. LongCat Turbo es un modelo destilado; el valor con
el que se produjo el render bueno del domingo era **20 pasos**, no 28.

Además, la medición independiente de pelo residual
(`_bald_hair_coverage`) devolvía `None` **sin traza** en cuatro caminos (máscara
de cabeza vacía, cuero cabelludo <0,5 % del fotograma, segmentador sin pelo en
la entrada, segmentador sin pelo en el resultado). En el log del 24/11 no
aparece `Pelo en cabeza solicitada`, así que la verificación no llegó a medir y
nadie supo que el chequeo se saltó.

## Causa medida — zapatos
Log del 24/11 13:09–13:12:

```
[ImgEditor] Objetivo local «shoes» medido en la personne: 238686 px (9.0 % de elle)
[ImgEditor] Sin selección: motor con visión (LongCat) en el recorte de la zona medida (955, 2335) → (320, 768)
[ImgEditor] Verificación del objetivo: «shoes» reducido a 28206 px (12 % de lo medido antes)
[ImgEditor] Validación visual OK: cambio real, sin fantasma
```

`residual_failure_fraction` era **0,15**. La medición que fijó ese valor
(2026-11-15) dice que una eliminación correcta deja **1,5 %** y un objeto que
sigue presente mide **91,8 %**. Un 12 % es 8× el residuo de una eliminación
correcta: no es un PASS. Con el umbral antiguo el resultado se aceptó, la ruta
no reintentó y las botas se guardaron.

## Cambios aplicados (v5.1445)
1. `config/editor_selection.json`
   - `local_appearance_min_steps`: **28 → 20** (valor del render bueno del domingo).
   - `local_appearance_max_work_side`: se mantiene en **1536** (guarda contra el
     stall de 5 min del codificador Qwen VL en recortes 3480×2901; el cliente ya
     muestrea a ~1 MP).
   - `residual_failure_fraction`: **0,15 → 0,05** (3,3× el residuo de una
     eliminación correcta, 18× por debajo del objeto presente).
2. `roop/img_editor/img_editor_manager.py`
   - `_bald_hair_coverage` imprime el motivo cuando no puede medir pelo residual
     (máscara vacía, cuero cabelludo pequeño, sin pelo en entrada o en resultado).

## Verificación
- `py_compile` de `roop/img_editor/img_editor_manager.py` y JSON válido de
  `config/editor_selection.json`.
- Revisión visual de los PNG entregados (`edit_1795474328.png`,
  `edit_1795521434.png`, `edit_1795522333.png`) para confirmar el síntoma.

## Límite y siguiente paso
**Sin PASS visual nuevo**: no se ha vuelto a renderizar con ComfyUI en este
turno. Hay que reiniciar la app y repetir los dos prompts reales:

- «EL HOMBRE DEBE SER CALVO» con las dos personas → cabeza pelada sin resto en
  la coronilla y sin tocar a la otra persona.
- «LA MUJER DEBE IR SIN ZAPATOS» → sin botas; si queda residuo ≥5 % debe
  aparecer `el motor no lo quitó` y la ruta debe reintentar/entregar aviso, no
  «Validación visual OK».

Si con 20 pasos el pelo sigue, el siguiente lever medido es ampliar la zona de
la cabeza al pelo real del sujeto (la elipse `_head_mask_from_faces` no siempre
cubre el pelo que sobresale) antes de tocar el prompt.
