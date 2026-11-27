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

## Revisión de los renders posteriores (log `ui_start_20261124_140346.log`, 14:03–19:23)

App relanzada a las 14:03, es decir con v5.1445 ya cargada (20 pasos, residual
0,05). Ocho peticiones, todas «el hombre debe ser calvo»:

| Hora | Salida | Calidad | Resultado visual |
|---|---|---|---|
| 14:26 / 14:31 / 14:34 | `edit_1795526760`, `edit_1795527096`, `edit_1795527239` | no | cabeza con pelo |
| 16:16 | `edit_1795533370` | no | cabeza con pelo |
| 17:32 | `edit_1795537947` | sí | coronilla pelada, banda de pelo en sienes/nuca |
| 17:38 | `edit_1795538329` | sí | igual |
| 18:44 | `edit_1795542250` | sí | igual |
| 19:20 | `edit_1795544427` | no | igual |
| 19:23 | `edit_1795544625` | sí | igual, con mechones sueltos sobre el cuero cabelludo |

Medidas de los tres últimos pases (idénticas entre sí):

```
[ImgEditor] Zona pedida 'bald' medida (1.7%, 0 px fuera de la silueta)
[ImgEditor] El pelo de la misma cabeza entra en la zona (+5424 px)
[ImgEditor] Apariencia: una pasada, 20 pasos, fuerza 1.00
[ImgEditor] Cambio medido sobre el sujeto: maximo 162-170 niveles, umbral Otsu=71-74,
            respuesta en 7.8-7.9% del sujeto
[ImgEditor] Zona pegada medida sobre el sujeto: 1.6-1.7% del recorte
[ImgEditor] Refino focal omitido: el cambio medido no abre una ventana focal
[ImgEditor] solo la zona medida, el resto de la foto se conserva | cumplimiento visual no verificado
```

Lectura:

1. **Los 20 pasos están activos** («una pasada, 20 pasos») y el cambio ya no se
   hunde como en v5.1444: la coronilla sale pelada desde las 17:32. La subida a
   28 pasos era, por tanto, parte del problema, pero no todo.
2. **La zona de generación sigue siendo pequeña para un calvo completo**: el
   segmentador mide `bald` en 1,7 % de la foto y la continuación de pelo de esa
   cabeza solo aporta 5 424 px. El modelo repinta lo que cae dentro de la
   máscara; la banda de pelo de sienes y nuca queda fuera y sobrevive al pegado
   («cambio medido ∩ permitido»). Es el límite que la nota anterior dejó como
   siguiente palanca, ahora medido.
3. **La verificación independiente de pelo no se ejecutó en ninguno de los ocho
   renders**: no aparece `Pelo en cabeza solicitada` ni ninguna de las trazas
   nuevas de `_bald_hair_coverage`. La causa está en el código, no en la
   medición: el bloque de pelo vive en `img_editor_manager.py` hacia la línea
   13 378 y exige `_focal_head_plan` o `_head_hole_plan`, pero la ruta de
   apariencia local sale antes, en el retorno de la línea ~10 426 con
   `cumplimiento visual no verificado`. Es una red de seguridad muerta en esta
   ruta: el aviso que debía impedir entregar un calvo con pelo nunca se calcula.
4. El mensaje final que ve el usuario ya admite `cumplimiento visual no
   verificado`, así que el problema no es que la app «diga OK»: es que entrega
   igual y sin reintento.

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

## Post-v5.1446 — zona medida más amplia y verificación medida en ruta rápida

### v5.1446 — cabeza más amplia y verificación medida en la ruta rápida, sin hardcode
(commit `e12d4fd`, 20:04)
- `_head_mask_from_faces`: la elipse de la cabeza pasa de 0,78×1,05 a **0,88×1,35**
  con lóbulo posterior 0,50×0,55; las sienes y la nuca ya entran en la zona de
  cualquier edición anclada a la cabeza.
- `_complete_hair_surface`: alcance de `span*0.35` → `span*0.85`, así el pelo de
  la cabeza se mide o se excluye completo (antes solo el mechón a 70 px del pico).
- La ruta rápida que antes terminaba en «cumplimiento visual no verificado» ahora
  ejecuta `_measure_requested_change` antes de entregar y avisa lo que no se
  cumplió.
- Validación: `py_compile`, probes de apariencia en verde; sin PASS visual nuevo.

### v5.1447 — el agujero de cabeza incluye el pelo medido (sin rastros)
(commit `0f2d497`, 20:58)
- Nuevo helper `_head_with_its_hair`: el pelo medido por CLIPSeg se une al
  agujero de la cabeza solo si pertenece a ESA cabeza (≥15 % del mechón sobre el
  elimpse, sin solapar otra cabeza ≥15 %); las demás cabezas se restan. Aplica
  en las dos rutas de cabeza (entrega por cabeza y cabeza-como-agujero).
- Efecto: el modelo repinta todo el pelo de esa persona y el pegado ya no
  deja banda original. El helper es general (no depende de la palabra
  «calvo»).

### v5.1448 — el modelo ve la persona entera para repintar la cabeza (sin rastros)
(commit `e635587`, 22:41)
- Para una edición de apariencia anclada a una cabeza, la generación usa la
  máscara de la persona entera (contexto completo) para que el modelo repinte
  con decisión; el pegado sigue restringido a la cabeza+pelo medidos
  (cuerpo intacto). Causa medida del residuo: una máscara parcial hace que el
  modelo sea conservador (6,8 % de cambio, 1,6 % pegado).

### v5.1449 — cuelgue en fotos grandes al pegar el pelo a la cabeza (árbol, sin commit confirmado)
- `_head_with_its_hair` sobre foto de 3480×2901: dilata/componentes conectados
  bajan a la cota de 768 px que usa el resto del cálculo de máscaras y la máscara
  resultante se reescala al tamaño original. Evita el bloqueo; sin cambio de
  comportamiento.

### Desplegable de zonas (v5.1450, árbol sin commit confirmado)
- `ui/tabs/img_editor/events.py`: `region_choices` de `.input` → `.change` (se
  dispara al elegir en un Dropdown multiselect en Gradio 6; el default
  «seleccionar» no marcaba). Inofensivo: `on_select` re-renderiza la zona
  elegida o vacía.

## Limitaciones medidas de esta review
- No se ha vuelto a renderizar con ComfyUI en esta turna (sin PASS visual). Los
  valores verificables son los de despacho (`py_compile`, probes verdes).
