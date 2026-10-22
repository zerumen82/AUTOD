# Propuesta — seleccionar elementos detectados en Image Editor

Estado: aprobado por el usuario con «hazlo» el 2026-10-18. Código integrado y
revisado el 2026-10-19; calidad visual completa pendiente (ver MEMORY.md).

Punto de relevo solicitado por el usuario: `plans/HANDOFF_editor_selection_20261018.md`.
Ampliación autorizada el 2026-10-19: el usuario responde «SI» a probar SAM2
ya instalado, sin descargar nada, e integrarlo únicamente si demuestra máscaras
completas y seguras. Esta autorización revoca solo la exclusión de SAM2 para
esta evaluación; no autoriza GrabCut ni instalaciones ni coordenadas inventadas.
Continuación 2026-10-19: inventario abierto verificado por descripción visual y
extracción local, backend y UI integrados, contratos e HTTP reales verdes.
Ediciones reales inspeccionadas aún incumplen partes de la instrucción y
objetos incompletos. El primer preflight JSON vacío se conserva como intento fallido,
superado por el inventario en dos etapas; no equivale a éxito de edición.

## 1. Contexto

El usuario propone que el Editor analice personas, objetos y otras regiones de
la imagen y permita seleccionar cuál editar. Esto evita que el modelo tenga que
adivinar el referente a partir del texto. Sustituye la propuesta pendiente de
solo mejorar la selección automática por texto.

Flujo propuesto: subir foto → Analizar imagen → ver regiones detectadas
resaltadas y su lista → pulsar una o varias regiones/elementos → ver la
selección confirmada → escribir el cambio → generar con Kontext sobre la unión de
las regiones elegidas.

**Ampliación aprobada el 2026-10-18**: la selección puede ser **múltiple**
(una o varias cosas/personas). Las regiones elegidas se unen en una sola
máscara y se envían en **una sola pasada** de Kontext (una sola carga de GPU).
Puede seguir utilizándose el editor sin selección explícita, con las guardas
actuales.

## 2. Reglas del proyecto

- «El usuario decide alcance, diseño y prioridades»: este documento se consulta
  antes de añadir la interfaz o un contrato nuevo.
- «NO HARDCODE»: categorías y descripciones proceden del análisis visual de la
  foto; las regiones proceden de detección/segmentación real. Sin catálogo de
  géneros, prendas, objetos ni ediciones escrito a mano para enrutar.
- «LOCAL-FIRST» y sin instalaciones no autorizadas: comprobar primero los
  modelos y bibliotecas ya instalados. La capacidad real debe probarse antes de
  ofrecer un tipo de selección. No prometer localizar cualquier objeto.
- Se conserva la decisión anterior sobre el Editor: selección de elementos
  detectados con vista previa, sin volver a introducir pintado libre de máscaras.
- El historial rechazó coordenadas inventadas por Moondream y experimentos con
  SAM2/GrabCut. No recuperar esas soluciones silenciosamente. Un VLM puede
  describir propuestas, pero sus coordenadas no son una máscara verificada.

## 3. Pasos de implementación propuestos

1. Especificar aceptación en `spec/02-REQUISITOS.md`, arquitectura en `spec/01`
   y límites en `spec/03`; conservar decisiones en `MEMORY.md`.
2. Probar, con fotos no explícitas y modelos instalados, un inventario visual
   abierto y regiones medidas por los detectores/segmentador. Personas detectadas
   y objetos descritos deben corresponder a píxeles verificables. Crear un módulo
   de propuestas de selección en `roop/img_editor/` solo después de esa prueba.
3. Añadir Analizar/selección/vista resaltada en `ui/tabs/img_editor/ui.py` y
   eventos/estado en `events.py` y `state.py`. Mantener la foto original intacta;
   nunca enviar a Kontext la imagen con las marcas de interfaz.
4. Pasar identidad de selección y región medida al manager. La selección
   explícita gana sobre la deducción textual de persona/objeto; el prompt sigue
   describiendo el cambio completo. Reutilizar recorte, composición y protección
   final del exterior; no recuperar LongCat.
5. Invalidar la selección si cambia la foto, al reutilizar un resultado o si la
   propuesta ya no corresponde al contenido. Mostrar ausencia/ambigüedad real.
6. Conservar endpoints Android 13/9/8. Añadir contrato separado para selección
   si resulta necesario, sin cambiar la firma de clientes antiguos. La elección
   visual nueva se implementa primero en la UI de escritorio existente; no se
   recompila ni cambia la APK sin ampliar expresamente el alcance.

## 4. Riesgos y límites

- La propuesta visual puede omitir elementos o unir objetos en contacto. La
  vista resaltada permite comprobar la región antes de generar; si no es válida
  no se presenta como una selección fiable.
- Una cara detectada no equivale por sí sola al cuerpo entero de una persona.
  El resaltado debe mostrar exactamente la región que se puede editar.
- No reducir márgenes ni añadir excepciones para que pasen las fotos de prueba.
- La propuesta no incluye instalaciones, modelo nuevo, cambios de motor,
  ampliación de Android ni recuperación de sistemas rechazados. La selección
  múltiple sí está incluida por la ampliación aprobada del apartado 1.
  Si los recursos instalados no permiten objetos fiables, se presenta la evidencia
  y se consulta el siguiente diseño antes de ampliarlo.

## 5. Impacto

La elección explícita del referente resuelve una ambigüedad que hoy acaba en
persona equivocada u omisión. Se integra con el pegado protegido existente y
conserva contratos antiguos y flujos Generate/FaceSwap. El análisis solo se
ejecuta al pedirlo; no añade una carga visual a todas las ediciones.

## 6. Verificación

- Propuestas reales sobre varias fotos: dos personas, varios objetos, objetos
  en contacto y elementos no detectables; guardar regiones y salidas únicas.
- Clic y lista eligen la misma región medida. Nueva foto elimina selección.
- Cliente real recibe la foto limpia y la selección correcta. Otra persona y
  exterior se mantienen píxel a píxel después de generación y acabado.
- El referente elegido se conserva aunque el rewriter devuelva otro genérico.
- Contratos API 13/9/8, cámara, compuestos y Generate verdes; Ruff completo y
  compileall de afectados verdes; reviewer APPROVE y tester PASS.
- Comprobar visualmente una edición real completada sobre una persona y otra
  sobre un objeto; la presencia de regiones o un render terminado no bastan.
