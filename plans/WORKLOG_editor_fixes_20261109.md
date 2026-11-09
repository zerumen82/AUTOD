# WORKLOG — Correcciones Image Editor, Desvestir/Sin Ropa y Extracción de Sujeto (v5.1320), 2026-11-09

Petición del usuario: revisar los logs de las últimas ediciones, proponer mejoras, revisar MEMORY y demás .md, proceder con los arreglos y subir versión anotando los cambios.

## Diagnóstico (logs reales de la sesión 2026-11-09)

1. `logs/ui_start_20261109_161142.log` y `logs/ui_start_20261109_191039.log`:
   - Petición: «la mujer debe ir sin ropa» en Auto.
   - En `selected_region_edit.py`, cuando la ropa no alcanza el umbral de silueta, la zona pasa a `persona medida menos la cabeza`.
   - La condición `vision_zone = (zone_label == "zona de ropa medida" or target_zone is not None)...` evaluaba `False`.
   - Se intentó cargar Kontext (`flux1-kontext-dev-Q4_K_M.gguf`), pero en Auto (`automatic_longcat=True`) solo se había autorizado LongCat al inicio de `edit_described_subject`.
   - La compuerta de `edit_execution.py` abortó con:
     `[ImgEditor] Auto detuvo la solicitud porque una ruta intentó sustituir un motor permitido sin una decisión medida. Modelo rechazado: flux1-kontext-dev-Q4_K_M.gguf`
     devolviendo la foto sin cambios y forzando al usuario a reescribir «sin vestido».
2. `logs/ui_start_20261109_191039.log`:
   - Petición: «el hombre tumbado debe ser calvo».
   - Traducido a «the lying man must be bald».
   - `_leading_visual_subject` descartaba «the» y tomaba solo la primera palabra restante (`tokens[0]`), extrayendo el participio/adjetivo `lying` en vez del sustantivo `man` o sintagma `lying man`.
   - CLIPSeg consultó `a photo of lying` obteniendo 0,00% de cobertura. Al no ubicar la región, el editor abortó la edición («Omitida: no se pudo localizar de forma inequívoca la región solicitada; se conserva la imagen»).
3. `logs/ui_start_20261109_191039.log`:
   - Petición: «la mujer debe ir en ropa interior».
   - Crash C++ en ComfyUI v0.19.0: `Windows fatal exception: access violation` en `comfy_kitchen/tensor/base.py:400` en `_handle_clone` durante el ciclo de descarga `unpatch_model` al alternar entre Kontext y LongCat con `--disable-dynamic-vram`. Provocó WinError 10054 / 10061 en la UI.
4. Modo Calidad:
   - Caras con posterización previa omitían CodeFormer incluso después de desposterizar.
   - Restauraciones faciales que degradaban parecido facial por debajo de 0,60 pasaban el corte relativo.

## Cambios implementados

1. **`roop/img_editor/selected_region_edit.py`**:
   - `edit_described_subject` autoriza explícitamente tanto `get_longcat_model_config()["model_file"]` como `get_kontext_model_config()["model_file"]` con `permit_automatic_model` al inicio y antes de invocar la carga en ComfyUI cuando `automatic=True`.
   - `vision_zone` ampliado para incluir `persona medida menos la cabeza` y `banda inferior` en peticiones `direction == "uncover"`, enrutando al motor con visión (LongCat) para retirar ropa y preservar la pose sin censura.
2. **`roop/img_editor/img_editor_manager.py`**:
   - `_leading_visual_subject` ahora extrae el sintagma nominal completo antes del verbo modal/auxiliar (`must`, `should`, `is`, `be`, etc.), obteniendo `lying man` o `man on the right` para que CLIPSeg pueda localizar la máscara focal.
   - Re-medición facial de posterización tras desposterizar en la etapa de pulido.
3. **`roop/img_editor/top_tier_finish.py` & `config/quality_mode.json`**:
   - Parámetro `min_absolute: 0.60` para revertir restauraciones que dejen el parecido por debajo de ese suelo.
4. **`roop/img_editor/image_quality_pipeline.py`**:
   - Caché de análisis por digest `blake2b` (`_analysis_cache`) para evitar repetir escaneos de artefactos costosos en imágenes 4K.
5. **`ui/tabs/comfy_launcher.py`**:
   - Bandera `--disable-dynamic-vram` para GPUs con ≤ 8 GB.

## Verificación

- `diagnostics/probe_editor_versatility_integrity_20261105.py`: **23/23 PASS** (8,96 s).
- `diagnostics/probe_editor_selection_contract_20261019.py`: **15/15 PASS**.
- `diagnostics/probe_editor_selection_body_regime_20261021.py`: **13/13 PASS**.
- `py_compile` y `ruff` (E9, F63, F7, F82): **0 errores** en todos los módulos modificados.

## Versión y Documentación

- Versión incrementada a **v5.1320 (2026-11-09)** en `run.py`, `roop/core.py`, `AGENTS.md` y `spec/00-PRINCIPIOS.md`.
- `MEMORY.md` y `spec/03-PENDIENTES.md` actualizados con el registro detallado.
