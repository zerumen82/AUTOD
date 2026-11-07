# Relevo de pendientes — 2026-10-29

Petición: revisar pendientes y seguir, dejando el estado antes de agotar contexto.
Leer este relevo junto con la cabecera de MEMORY.md y spec/03. El workspace
contiene muchos cambios de sesiones anteriores: este turno no los aprueba.

## Arreglo terminado y alcance

`ImgEditorManager._appearance_work_region` tenía un bloque que leía
`body_direction` sin definir. Fallaba también en ediciones locales neutrales,
sin persona ni inferencia. Se retiraron únicamente esas cinco líneas; la
inclusión geométrica de la persona ya estaba implementada más abajo.

Probe nuevo: `diagnostics/probe_editor_focal_window_20261029.py`.
Antes: 6 errores de subcasos por NameError. Después: 5 métodos PASS; comprueba
tres tamaños, bordes, soporte de persona, máscara vacía/ausente, crop exacto y
fuente intacta. No usa dobles de modelo ni ejecuta inferencia.

Reviewer independiente `review_pending`: APPROVE acotado al fix/probe.
Tester independiente `test_focal_fix`: PASS acotado; lint y compilación del
manager/probe verdes. Foto astronaut neutral instalada, dos llamadas reales
al helper: fuente intacta, crop exacto, 0 pérdida de máscara/persona.
Las máscaras de esta prueba son geometría explícita, no detecciones de modelos.
Root y tester inspeccionaron el overlay; no equivale a validar un render.

## Evidencia

- Root: `diagnostics/out/pending_audit_20261029/145422/`:
  `report.json`, `focal_before.log`, `focal_contracts_before.log`,
  `focal_contracts_after.log`, `ruff_after.json`, `launcher_build.log`.
- Tester: `diagnostics/out/pending_audit_20261029/tester_145653/`:
  `report.json`, `tester_review.md`, logs, fuente y overlays neutrales.
- Hash del manager verificado por tester:
  `c4f476c972f0d297e328f591925dea36070c826ff846e3d917562148054e9da5`.
- Regresiones PASS: referencia 7/7, identidad 14/14, alcance 10/10, color 11/11,
  controles Generate 6/6, estado de motor 18/18, paste_guard y recolor dispatch.
  Los contratos con inferencia sustituida no prueban calidad visual.
- Build local PyInstaller PASS:
  `diagnostics/out/pending_audit_20261029/145422/dist/AutoAuto.exe`.
  Se empaqueta launcher.py con fuentes/venv externos. No se ejecutó el launcher;
  no es una distribución autónoma del motor ni prueba de la UI.
- Ruff del área Editor/Generate sigue FAIL: 91 incidencias antes, 90 después;
  ninguna en el manager después. No declarar lint global verde.
- `git diff --check` por defecto trata CRLF como whitespace; chequeo equivalente
  con `-c core.whitespace=cr-at-eol` PASS. Ambos logs conservados; sin conversión
  masiva de finales de línea ni ocultar incidencias de lint.

## Documentación reconciliada

spec/00,01,05 describen la excepción explícita existente en edit_execution.py,
autorizada después de la política absoluta de octubre. No se modificaron motor,
parámetros, prompts ni rutas corporales. spec/03 marca históricas las deudas de
identidad/F841/contador del 24/10 corregidas el 25/10 y el render Grace ya repetido.
spec/02 advierte que la aprobación del 25/10 no aprueba cambios posteriores.
Reviewer independiente: APPROVE acotado también a documentación y relevo.
Corregidas sus dos observaciones menores de gramática y marcado histórico.

## Abierto y próximo paso

La prioridad se consultó al usuario por herramienta (Editor no explícito,
FaceSwap o Android/Generate); no hubo respuesta durante este trabajo.
Se continuó únicamente con el diagnóstico y el bloqueo genérico reproducido,
dentro del alcance previo de H-EDIT-003/H-AUDIT-FIX.

- Lote previo CHANGES_REQUESTED: lista manual que evita conceptos medidos en
  measured_full_body_exposure y umbral literal 0,35. Incumple reglas del usuario;
  anotado, no modificado aquí. No optimizar generación de desnudez ni usar
  pruebas explícitas para resolver una deuda de código general.
- Integración local acepta proporciones incompatibles y estira: riesgo ya
  medido el 26/10; falta atribución causal sobre una edición real neutral antes
  de elegir corrección. No cambiar diseño de encuadre sin decisión del usuario.
- Validación visual del Editor en varias fotos/semillas y aceptación del usuario.
  Color guard está verificado en CPU; no cerrar calidad universal.
- FaceSwap: perfiles, halo/oclusión y vídeos completos siguen abiertos; hace
  falta ruta vigente del vídeo representativo. Contratos/fotos no lo sustituyen.
- Android: build/lint previos pasan, falta teléfono conectado y prueba instalada.
- Ruff global: deuda declarada; no mezclar limpieza masiva con el fix acotado.

- Arreglos de esta sesión (documentados sin validación visual final):
  • Editor: `img_editor_manager.py` umbral literal `0.35` → `policy["body_state_min_intensity"]`; `is_true_full_body_exposure` sin lista manual.
  • Editor: `selected_region_edit.py` zona local de ropa (`clothing`) no expande a `full_body` para `uncover` no medido (zapatos/shorts/blusa/tanga).
  • FaceSwap: `quality_enhancements.py` `detect_foreground_occlusion` — `mouth_area` ampliada y estimación basada en puntos disponibles (`_feat`) para perfiles.
  • Motor: `roop/core.py` restaurado (`git checkout`) tras corrupción por `IndentationError`.

No se instalaron paquetes/modelos, no se hicieron commits/push ni se reiniciaron
UI, ComfyUI o aplicaciones del usuario. No hay renders o procesos propios
pendientes de esta sesión.
