# Relevo Image Editor — 24/10/2026, petición «anótalo todo»

## Actualización 25/10 — reparación H-AUDIT-FIX

El usuario autoriza corregir los defectos de la auditoría. Retirado el fallback
que ampliaba alcance por contar cláusulas; el fallo `clauses` sin inicializar
desaparece. Medición y límites compartidos en configuración. Calidad comprueba
identidad individual con correspondencia uno-a-uno y revierte evidencia inválida.
F841 eliminado; Ruff actual PASS. Reviewer APPROVE acotado al código nuevo;
regresiones CPU independientes 17/17 suites PASS. La actualización de pruebas
Generate comprueba el modelo físico Kontext y FluxGuidance del alias legacy,
sin añadir un alias público ni restaurar estados LongCat artificiales.

Build launcher PASS y APK debug offline PASS; lintRelease/lintVitalRelease PASS
(0errores/65avisos previos). Usuario autoriza resolver dependencias Gradle:
lintDebug también PASS (0errores/68avisos), sin suprimir advertencias.
La calidad visual general sigue abierta. Se conservan las
firmas/URLs elegidas por el usuario; no hay dirección fija ni creación de claves
con credenciales predeterminadas. No se instalan dependencias sin autorización ni se reinician
servicios ajenos. GPU autorizada para verificar tras preflight el 25/10;
dos renders Kontext completados y revisados por root/tester en MEMORY.md.
Calvicie PASS acotado; azul en Grace cumple color pero añade corte bob/flequillo,
sin certificar conservación exacta del peinado. Ambas entregas con exterior0.
Chroma nuevo también PASS acotado (escena neutra, T5único/CFG3.8), evidencia
propia en MEMORY.md. El vídeo FaceSwap histórico ya no existe en su ruta;
solicitada otra vigente, sin convertir la prueba en retrato en certificado de vídeo.

Los logs del 25/10 ya registran renders Kontext terminados169.81s/157.29s;
no equivalen a inspección visual de entrega. Las notas siguientes conservan
la cronología del 24/10 y sus resultados de entonces; «actual» en ellas no
sustituye esta actualización. Detalle y límites vigentes: MEMORY.md y spec/03.

## Actualización documental: commit/push y prueba por el usuario

Petición posterior: actualizar documentación, commit y push; el usuario probará.
No se ejecutan nuevos renders ni se reinician UI/ComfyUI/Ollama en este turno.
Los siguientes resultados GPU son históricos y mantienen sus límites.

Reviewer actual: **CHANGES_REQUESTED del lote completo**, por cambios corporales
posteriores al APPROVE de apariencia/cierre descrito abajo. Dos cláusulas por
coma bastan para `measured_full_body_exposure=True` aunque solo pidan recolor;
`_body_state_clause` puede añadir desnudez completa no solicitada y asignarla
al primer sujeto («The sofa must be gray, the woman must go barefoot» produce
«The sofa is fully nude» con dirección uncover/intensidad0.8). Reproducción CPU
real; deuda abierta en spec/03. No atribuir APPROVE global al lote actual.
Los umbrales de ese cambio están como literales en código, no en configuración.

Reviewer repitió en CPU apariencia10/10, corporal10/10 y cleanup2/2 PASS;
esos contratos no detectan los falsos positivos anteriores. Además, la puerta
de identidad de Calidad usa el máximo entre caras; no garantiza preservación
individual de todas las personas. La validación visual general sigue abierta.

Chequeos actuales: compilación del editor/UI PASS; Ruff del lote FALLA F841
(`scores` sin usar en `image_quality_pipeline.py:1520`). El commit documental
conserva el código local sin publicarlo: una prueba desde otro checkout no
recibirá los cambios de procesamiento pendientes. No declarar lint global verde.

Para probar: reabrir AutoAuto y analizar de nuevo la foto para obtener los
contornos nuevos; comparar entrada/salida final y comprobar el exterior y las
otras personas. No confundir calvicie con pelo blanco ni el raw con la entrega.
Grace sigue sin resultado validado. Conservar Ollama activo.

## Instrucciones del relevo anterior (histórico)

- Petición original: Image Editor devuelve la misma imagen al pedir calvicie,
  con y sin selección. Continuación: «arréglalo».
- **Mantener Ollama activo.** Respuesta explícita del 24/10: «Sí, consérvalo
  activo». No descargar sus modelos, detener su servidor ni matar sus procesos.
- Petición de entonces: registrar todo. No declarar la tarea visual general cerrada.
- Reglas: AGENTS.md global/proyecto, SDD, sin listas de palabras para enrutar,
  sin instalaciones ni modelos nuevos, sin commits en aquel turno, sin tocar renders ajenos.

## Estado del relevo anterior: apariencia y cierre (histórico)

- Código de apariencia/cierre corregido y reviewer independiente **APPROVE**;
  no cubre el fallback corporal posterior ni equivale a APPROVE del lote actual.
- Protección global **20/20 PASS**, apariencia **10/10 PASS**, selección
  **15/15 PASS**, régimen corporal **10/10 PASS**, contrato API PASS.
- Ruff completo de archivos tocados y compilación del editor/UI verdes.
- Primera foto neutra: **PASS visual con selección y en ruta global**.
- Segunda foto neutra (Grace Hopper): **SIN VEREDICTO VISUAL**. El último
  intento se interrumpió intencionalmente; no hay resultado final validado.
- Cierre del verificador: **2/2 contratos PASS y dos cargas/cierres GPU reales
  PASS**, liberando 1.224 MB cada vez, sin acumulación entre cargas.
- No hay una nueva prueba de rapidez de edición después de esa corrección.
- No se generó ninguna edición de la foto explícita original del caso.

## Cambios ya implementados

### v5.1249 — selección y pegado

- `roop/img_editor/selected_region_edit.py`: una persona inventariada como
  `object` que contiene una sola cabeza medida usa referencia focal de cabeza,
  por solape configurado, sin reglas por nombres/géneros/prendas. Cero/dos
  cabezas y selección múltiple conservan sus contratos. Escritura limitada a
  la selección aceptada y protección de las otras cabezas.
- `ImgEditorManager._scale_kontext_focal_reference`: amplía recortes pequeños
  al presupuesto existente `maximum_reference_side=768`.
- `element_selection._complete_head_proposals`: completa contornos ANTES de
  mostrarlos/seleccionarlos, reutilizando `_complete_localized_region_mask`.
  Asocia cabeza/persona con máscaras ORIGINALES y unicidad; excluye cabezas
  ajenas del soporte FINAL de la propuesta. Actualiza bbox/cobertura.
- `_blend_head_boundary`: composición NORMAL_CLONE con centro de la bbox real
  de máscara; exterior restaurado. `edit.head_blend_method=normal_clone` en
  `config/editor_selection.json`. Si el ajuste anula el cambio o falla, se
  mantiene el feather anterior. La guarda mide píxeles, no semántica universal.
- Eliminada llamada `generate` duplicada en rama `vision_body`: se descartaba
  el primer resultado y se generaba de nuevo. Suite corporal8/10→10/10.

### v5.1250 — referente débil y global

- `_head_mask_from_region_plan(... eligible_faces=None)` conserva otros callers.
  Kontext compara TODAS las detecciones y exige elegibilidad DESPUÉS de elegir
  ganador. Evita editar a un vecino creíble cuando el referente tiene baja
  confianza, incluido caso de cabezas próximas.
- Evidencia de región medida sobre detección descartada veta el fallback a
  otra persona. Sin región medida, una sola cara creíble sigue siendo editable
  aunque haya detecciones espurias; el fallback protege las otras detecciones.
  **No volver a exigir `len(detected_faces)==1`: ya produjo regresiones.**
- Global focal usa plantilla regional configurada con la instrucción traducida
  íntegra. No se inyectan edad/framing inferidos en ese prompt de muestreo ni
  se añaden recetas por palabras. La instrucción explícita del usuario permanece.
- Retrato con una cara creíble usa `edit.crop_max_area` existente; varias caras
  creíbles conservan la guarda local anterior. Evita omitir retratos grandes.
- Global focal comparte `_blend_head_boundary`, con máscara de escritura que
  excluye otras cabezas y la misma guarda/fallback de composición.
- `InstructionVerifier.unload`: cierra Llama y proyector explícitamente; el
  handler instalado carece de `close()` y libera `clip_ctx` a través de
  `_exit_stack.close()`. Estado a None antes del cierre, intentos separados e
  idempotentes. Se cierra proyector aunque falle el cierre del texto.

## Evidencia visual y fallos: no mezclar intentos

Raíz: `diagnostics/out/appearance_reference_20261023/`.

| Directorio | Resultado real |
|---|---|
| `1792789243500027500` | FAIL permisos de aislamiento al subir PNG; cero inferencias |
| `1792854592938590500` | Modelo rapa; máscara incompleta vuelve a pegar pelo lateral; FAIL visual |
| `1792855222215129000` | Rapada sin pelo restaurado, pero óvalo luminoso; FAIL integración |
| `1792855675823830800` | Selección: PASS visual rapada sin arcos/óvalo; 30.656px cambiados, exterior0; una pasada20steps,96s modelo/131,313s callback |
| `1792858216525529500` | Global baseline: pelo BLANCO en vez de calvicie; FAIL visual |
| `1792858417311810800` | Grace baseline: omisión por cabeza>35%; cero llamadas; FAIL |
| `1792858682799844700` | Global corregido astronaut: PASS visual; rapada sin arcos/óvalo; una pasada20steps,127s modelo/225,968s callback; 39.789px cambiados, exterior0 |
| `1792858908810863000` | Grace global corregido: INTERRUMPIDO, sin imagen final; no es PASS |

En el global corregido la máscara es suave: exterior0 significa soporte
`mask>0`, **no núcleo>=128**. En astronaut hay 7.647px cambiados en borde1..127.
Los replays `replay_*` son composición de un raw existente, NO renders nuevos.
Los resultados válidos tienen `source.png`, `raw_0.png`, `result.png`,
`mask.png` y `metrics.json`; inspeccionar la entrega final, no solo el raw.

## Interrupción de Grace, propiedad de procesos y Ollama

- Grace se ralentizó: pasos de4–83s y parada prolongada en18/20. No se atribuye
  a OOM: **la interrupción fue intencional del coordinador**.
- Antes de POST `/interrupt` se comprobó cola con SOLO nuestro prompt
  `d5a81117-91ba-4a6c-aff7-09181308fced`, fichero `fast_1792858988.png`, pending0
  y sin UI en7861. Se guardó `INTERRUPTED.json` y la respuesta fue200.
- El callback de UI reintentó automáticamente porque un interrupt externo de
  ComfyUI no cambia `state.is_generating`. Se comenzó una pasada adicional.
- Para impedir ese reintento se envió Ctrl+C a **nuestro propio proceso de
  diagnóstico** (sesión91114). Su grupo/proceso ComfyUI propio terminó; luego
 8188 rechazaba conexión. No se descargó/mató Ollama ni otra aplicación.
- Ese intento no debe usarse para prometer rapidez ni una única pasada acabada.
- `ollama ps` mostró `huihui_ai/qwen3.5-abliterated:2B`,1,7GB,100%GPU. Es carga
  AJENA que el usuario confirmó estar usando. PID observado2008 puede cambiar:
  **no usar un PID histórico como objetivo**. Con esa carga quedaban≈5,3GB libres.
- El proceso del render interrumpido tenía cargado el código anterior del
  verificador: editar el archivo no cambió sus recursos ya residentes.

## VRAM: comprobación nativa completada

Probe: `diagnostics/probe_instruction_verifier_vram_20261024.py`.
Log: `logs/verifier_cleanup_native_20261024.log`.
Datos: `diagnostics/out/verifier_cleanup_20261024/1792859771980905000/metrics.json`.

| Carga | Libre antes | Libre cargado | Libre tras cerrar | Liberado |
|---|---:|---:|---:|---:|
| 1 | 5341MB | 3991MB | 5215MB | 1224MB |
| 2 | 5215MB | 4011MB | 5235MB | 1224MB |

PASS: carga real GPU y liberación, sin acumulación entre dos ciclos. No prueba
que un render posterior vaya rápido si otra aplicación mantiene VRAM ocupada.

## Pruebas, revisión y logs para retomar

- `diagnostics/probe_editor_kontext_people_protection_20261018.py`:20/20.
  Antes15/16; añadidos referente débil cercano, fallback espurio protegido,
  retrato de una cara y prompt sin edad inferida.
- `diagnostics/probe_editor_appearance_reference_20261023.py`:10/10.
- `diagnostics/probe_instruction_verifier_cleanup_20261024.py`:2/2, con error
  de cierre de texto e idempotencia.
- Regresiones: selección15/15, corporal10/10, API antigua conservada.
- Reviewer APPROVE independiente de elegibilidad, global y cierre de recursos.
- Tester PASS acotado global astronaut y CPU20/20. Grace NO tiene veredicto.
- Logs: `weak_grounding_20261024_before.log`,
  `weak_grounding_20261024_near_target.log`,
  `global_focal_20261024_before.log`, `global_focal_20261024_final_cpu.log`,
  `global_focal_reviewer_20261024.log`,
  `global_appearance_reference_reviewer_20261024.log`,
  `appearance_global_real_20261024_baseline.log`,
  `appearance_global_real_20261024.log`, `verifier_cleanup_native_20261024.log`.

## Próximo paso exacto, todavía pendiente

1. Conservar Ollama activo. Comprobar puertos/colas/VRAM actuales sin tocar
   procesos ajenos. No repetir un render normal bajo presión de VRAM sin
   estudiar la reserva de memoria de ComfyUI propio.
2. Solo repetir Grace desde proceso LIMPIO con el cierre corregido y conservar
   el intento interrumpido. La orden preparada es:

   ```powershell
   & .\venv\Scripts\python.exe -u diagnostics/probe_editor_appearance_real_20261023.py --mode global --sample grace --prompt 'La persona debe tener el pelo azul'
   ```

   Este probe escribe imágenes en ComfyUI/input/output: el aislamiento anterior
   denegaba esas escrituras; solicitar escalación cuando corresponda. Guardar
   log nuevo, sin sobrescribir el intento interrumpido.
3. ComfyUI instalado admite `--reserve-vram <GB>` (cli_args.py149). Solo se
   consultó como posible forma de coexistir con Ollama: NO se cambió launcher,
   defaults ni configuración de reserva. Evaluar con las reglas de alcance y
   configuración del proyecto antes de introducir un cambio de producto.
4. Inspección visual de source/raw/result de Grace y tester independiente:
   pelo azul realmente visible, contexto conservado, exterior0, intentos y
   tiempo reales. No basta una máscara no vacía o diferencia de píxeles.
5. Registrar el veredicto. No afirmar validación en DOS fotos hasta ese PASS.
   Falta también validación visual del usuario; no prometer calidad universal.

Para usar el código nuevo en AutoAuto hay que reabrir la UI y volver a analizar
la foto: una selección anterior conserva el contorno antiguo. No se reinició
una UI de usuario, no hubo instalaciones, commits ni cambios en
FaceSwap/Generate/Android. El workspace ya contenía cambios previos: preservarlos.
