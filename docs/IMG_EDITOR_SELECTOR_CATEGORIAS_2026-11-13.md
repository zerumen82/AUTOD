# Selector «Tipo de edición» del Image Editor (v5.1349–v5.1354)

Registro de la funcionalidad, contratos y mediciones. **Pendientes abiertos:
`docs/PENDIENTES_2026-11-13.md`.**

## Qué hace

Desplegable sobre la caja de prompt con 10 entradas: `auto` + 9 categorías.
Es un **prior de ruta y ayuda en la caja**, nunca una prohibición:

- con `auto` el comportamiento es el de v5.1348 (idéntico);
- con una categoría se eleva **solo su eje medido**;
- una **señal muy alta** del prompt gana al selector y queda registrada en el log;
- no sustituye ni recorta el prompt.

Entradas: `auto`, `pose`, `clothing`, `body`, `features`, `background`,
`camera`, `add`, `remove`, `people`.

## Tablas (definición de la opción, no listas de prompt)

`roop/img_editor/img_editor_manager.py`

```python
_PINNED_ENGINE_BY_HINT = {"camera": "qwen_camera"}

_SCOPE_HINT_EFFECTS = {
    "pose":       {"axis": "pose",             "facts": {"pose": True}},
    "clothing":   {"axis": "body",             "facts": {"body": True}},
    "body":       {"axis": "body",             "facts": {"body": True}},
    "features":   {"axis": "local_appearance", "facts": {"local_appearance": True}},
    "background": {"axis": "scenario",         "facts": {"scenario": True}},
    "camera":     {"axis": "camera",           "facts": {}},
    "add":        {"axis": "structural_add",   "facts": {"structural_add": True}},
    "remove":     {"axis": "structural_remove","facts": {"structural_remove": True}},
    "people":     {"axis": "people",           "facts": {}},
}
```

- `people` sin `structural_add/remove` medido eleva `scenario` (edición de marco).
- `camera` no eleva hechos: fuerza el POV explícito en la ruta.
- `clothing` **no inventa dirección**: la dirección e intensidad salen de
  `_prompt_authority_facts` (una frase neutra queda en `change`).
- Solo `camera` fija motor (`qwen_camera`). El resto deja el motor a la ruta.

## Flujo

1. `ui/tabs/img_editor/events.py::on_generate(..., scope_hint)` fuerza
   `engine="auto"` cuando hay categoría (para no perder `auto_requested`).
2. `ImgEditorManager.generate_intelligent(..., scope_hint=...)` publica el hint
   en `_SCOPE_HINT` (ContextVar) y lo resetea en `finally`.
3. `_prompt_authority_facts` aplica `_apply_scope_hint` al final:
   - limpia los ejes medidos **débiles** (no pueden borrar el eje elegido);
   - aplica la tabla; conserva las señales fuertes (se registran y no se aplica);
   - `authority = pose or body`.
4. Campo nuevo `scenario_strength` (solo lectura) = máxima puntuación de
   `structural_background` por cláusula; umbral fuerte `0.60`.
5. `_explicit_viewpoint_request` cede ante una **postura medida en la misma
   cláusula** (`camera_counts_as_pose=False`). «cambia el POV» sigue entrando.
6. «Modo Calidad» + checkbox «🧼 Solo acabado» (`quality_steps["only_finish"]`):
   aplica solo los pasos marcados, el prompt no edita. Red de seguridad: con
   Modo Calidad y un prompt sin ningún eje medido se ejecuta el acabado en vez
   de «Zona local no medida»; no aplica con máscara pintada, selección visual ni
   POV explícito/forzado.

## Zonas medidas y verificación (ruta sin selección)

`roop/img_editor/selected_region_edit.py`

- `_measured_state_zone`: mide el ESTADO (`descalza`, `topless`) por cláusulas
  (`re.split` por `,`/`y`/`and`) con el segmentador, aceptando piel, y **une**
  las partes medidas (pies + medias). Devuelve `(máscara, etiqueta, fracción)`.
- `edit_described_subject`: si hay objeto nombrado **y** estado, la zona es la
  **unión medida** objetivo ∪ estado (≤60 % de la persona).
- Reintentos residuales: verifican también `check_clothing_residual` y
  `check_state_skin`; el recorte ampliado solo da contexto, se escribe lo
  medido (contrato v5.1323).
- `check_state_skin`: mide piel en la zona escrita sobre el **parche del motor
  antes de armonizar**; pasa si `piel ≥ 55 %` o `+10` puntos. Si no,
  `TARGET_RESIDUAL_FAIL` y no se declara «Validación visual OK».
- Prenda puesta (`≥80 %` de su zona dentro de la persona) ⇒ misma verificación
  por piel.

## UI

- `ui/tabs/img_editor/ui.py`: `CATEGORY_CHOICES` (única fuente), `category`,
  `category_help`, `quality_only_finish`.
- `ui/tabs/img_editor/events.py`: `on_category_change` (fija/desbloquea motor y
  muestra ejemplos sin escribir la caja), `_on_generate_desktop` (17 entradas).
- Endpoints API sin cambios: `edit` 8, `edit_options` 9, `edit_select` 13,
  `edit_region` 14; `edit_manual_region` 17 (solo escritorio).

## Pruebas

| Probe | Resultado |
|---|---|
| `diagnostics/probe_editor_category_selector_20261113.py` | 31/31 |
| `diagnostics/probe_editor_barefoot_state_20261113.py` | 9/9 |
| `diagnostics/probe_editor_category_camera_20261113.py` | 4/4 |
| `diagnostics/probe_editor_only_finish_20261113.py` | 5/5 |
| `diagnostics/probe_editor_selection_ui_contract_20261019.py` | 17/17 |
| `diagnostics/probe_editor_ui_skip.py`, `probe_editor_auto_appearance_recovery_20261104.py` | OK |

Fallos previos (también en HEAD, no causados por el selector):
`probe_editor_viewpoint_dispatch.py` (el token suelto `view` cuenta como cámara),
`probe_editor_selection_body_regime_20261021.py` (2/13),
`probe_editor_compound_fast_dispatch.py` (una aserción de fallo de cliente).
