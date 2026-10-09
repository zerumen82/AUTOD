# spec/04 — Ejército de agentes y flujo

## Roles (global `~/.agents/AGENTS.md`)

| Rol | Uso en este proyecto |
|---|---|
| sd-coordinator | Punto de entrada; **pregunta al usuario** antes de decidir alcance. |
| sd-editor | Implementa historias aprobadas (`spec/02`). No decide alcance. |
| sd-tester | Verifica con evidencia (imágenes/logs reales). Su PASS/FAIL es la última palabra. |
| sd-reviewer | Calidad de código: seguridad, sin hardcode, mantenibilidad. |
| sd-scout | Busca skills antes de reinventar; solo instala con aprobación. |
| sd-researcher | Investiga antes de implementar, con fuentes. |

## Flujo obligatorio

```
petición → leer spec/ + MEMORY.md
  → ¿especificada? no → redactar historia → PREGUNTAR al usuario
  → implementar en pasos pequeños
  → probes + py_compile + prueba real tras cada paso
  → revisión de calidad
  → validación visual del usuario si el objetivo es calidad de imagen
  → actualizar spec/02 + MEMORY.md → HECHO
```

## Comandos de verificación habituales

```powershell
$env:PYTHONPATH='D:\PROJECTS\AUTOAUTO'
& venv\Scripts\python.exe -m py_compile <archivos tocados>
& venv\Scripts\python.exe diagnostics\probe_editor_compound_fast_dispatch.py
& venv\Scripts\python.exe diagnostics\probe_editor_viewpoint_dispatch.py
& venv\Scripts\python.exe probe_v51140_gen.py     # contrato de Generate
# Generación real (arranca ComfyUI solo si no está corriendo):
& venv\Scripts\python.exe diagnostics\probe_editor_bald_compliance.py real <label> <foto> "<prompt es>" "<prompt en>"
```

Nota: `EDITOR_REAL_SEED=<n>` fija la semilla en el modo `real` para A/B controlados.

## Reglas operativas

- No reiniciar ni interrumpir la UI ni un render activo sin comprobar su estado.
- Imágenes de diagnóstico con **nombre de archivo fresco y único** antes de leerlas.
- No declarar cerrada una regresión sin validación visual del usuario.
