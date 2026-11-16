# Validación del fix v5.1334 (log 2026-11-11)

## 1. Cambios en el código

**Fallo A — residual aceptado**  
- `roop/img_editor/img_editor_manager.py:7702‑7708` ahora **rechaza** el resultado con residual y **cae a la ruta fuerte** (LongCat) en vez de devolverlo como si estuviera bien.  
- El comentario v5.1311 ya advertía que el pase cambió píxeles pero no obedeció; ahora se respeta esa advertencia.

**Fallo B — sin cambios medidos**  
- `roop/img_editor/img_editor_manager.py:8425‑8464` ahora **recurre a `_measured_subject_passes`** cuando `body_edit` es verdadero y **no hay zona medida** (`subject_mask is None and not subject_passes`).  
- El fallback de zona de persona ya existía (v5.1257), pero ahora se activa antes de que el editor termine en silencio.

**Saneado de posesivos**  
- `roop/img_editor/img_editor_manager.py:4881` usa `_strip_possessive()` para que `man's neck` → `man`, `woman's shoulder` → `woman`.  
- Las queries de soporte ahora se proyectan a tokens reales, evitando que CLIPSeg reciba `'man s'` / `'woman s'` y mida mal.

### 2. Qué probar

Ejecuta el probe de validación real en la sesión que tiene ComfyUI activo (puerto 8188 abierto):

```powershell
venv\Scripts\python.exe diagnostics\probe_v51334_20261111.py
```

### 3. Qué debe pasar

- **Fallo A**: la imagen guardada debe mostrar **sin fantasma rosa/blanco** y sin ropa; la métrica de cambio debe ser > 0 px y el residual < 40 % → PASS.  
- **Fallo B**: la imagen guardada debe mostrar al hombre **con top de mujer** y sin cambios medidos; el log debe contener `"Zona de soporte sin medida: la edición recae en el sujeto medido de la instrucción"` → PASS.

Si ambos casos devuelven **PASS** y los archivos `summary.json` no reportan errores, el fix está completo.  
Si alguno falla, revisa los logs de `generate_intelligent` para ver si la ruta fuerte se ejecutó o si el fallback de zona de persona no se activó.

## 4. Versión

- **Nueva versión**: v5.1334 (2026-11-11)  
- **Cambios de versión**: actualizados en `run.py`, `roop/core.py`, `AGENTS.md`, `spec/00-PRINCIPIOS.md` y `MEMORY.md`.  
- **Commit**: se realizará con mensaje "v5.1334: Fix residual acceptance and fallback for no-measured-zone edits (sanitized possessive queries)".