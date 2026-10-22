# spec/02 — Requisitos (historias)

## H-EDIT-SELECT — selección de elementos detectados, aprobada 2026-10-18

El usuario propone analizar personas/objetos y permitir seleccionar el referente
en la imagen. Diseño concreto en `plans/editor_object_selection_20261018.md`,
aprobado por el usuario con «hazlo». Inventario abierto medido sobre la foto,
vista resaltada y selección por clic/lista; sin catálogo escrito a mano ni
reintroducir pintado libre. Conservar imagen limpia y API/Android antiguas.

- [ ] Análisis real local de varias fotos no explícitas, con cabezas detectadas
  y objetos del inventario visual abierto validados por segmentación. No usar
  coordenadas inventadas por un VLM ni categorías de objetos escritas a mano.
- [x] Analizar, seleccionar por clic/lista y mostrar contorno de la región real;
  distinguir cabeza/rostro de persona completa. Selección **múltiple**: una o
  varias regiones/elementos, unidos en una sola máscara para una sola pasada.
- [x] Estado aislado por sesión y vinculado a la huella RGB de la foto. Nueva
  imagen o reutilizar resultado invalida propuestas y selección previas.
- [x] La selección explícita prevalece sobre inferencias textuales del referente.
  Kontext recibe la foto limpia/recorte, no el resaltado de interfaz. Restaurar
  exterior y otras cabezas después del acabado; rechazar selecciones caducadas.
- [x] API Android 13/9/8 conservada. Callback/endpoint separado para el selector
  de escritorio; sin cambiar APK, motor, modelos ni instalar dependencias.
- [ ] Ruff y compilación, contratos UI/manager, tester PASS, reviewer APPROVE;
  UI local servida/probada y ediciones reales de persona/objeto inspeccionadas.

Evidencia 2026-10-19: backend 14/14, UI 14/14 (incluidas solicitudes de
análisis que empiezan tarde tras cambiar/quitar/reutilizar foto), API legacy
13/9/8 y regresiones Generate 108/31/5 verdes; reviewer APPROVE del código.
UI servida y callbacks HTTP reales con dos sesiones y reutilización PASS en
`diagnostics/out/editor_selection_served_20261019/1792421405518127600/`.
La prueba encontró y corrigió reuse de ndarray; salida PIL/PNG explícita.
Ediciones reales guardadas en `diagnostics/out/editor_selection_20261019/`:
café con 0 píxeles fuera de soporte, pero taza parcialmente azul por máscara
incompleta y etiqueta «saucer» incorrecta; cabeza seleccionada con pelo rojo y
exterior/otra persona intactos, pero corte corto/expresión no exactos.
Otro render de cuchara conserva exterior, pero no vuelve oro su superficie.
**Historia abierta por cobertura/fidelidad visual y validación del usuario.**
La UI presenta regiones como propuestas revisables; una máscara real no prueba
que su etiqueta sea correcta ni que cubra el objeto entero.

Corrección autorizada 2026-10-19, usuario «PUES ARREGLALO»: resolver cobertura
y obediencia con recursos instalados, sin reducir umbrales por foto. Reproducir
deformación 150×174→128×128 y comparar con referencia isotrópica a resolución
útil, padding reversible y pegado sobre soporte original. Contrastar evidencia
CLIPSeg global/local y etiquetas derivadas semánticamente del inventario real,
sin catálogos manuales. Comparar receta/texto por separado, misma semilla;
verificar visualmente taza/cuchara/cabeza y otras fotos antes de afirmar mejora.
Mantener contexto exterior y contratos anteriores, reviewer/tester independientes.
La selección de objeto necesita contexto visual para que el editor reconozca
el elemento completo: el bbox de una segmentación parcial no basta como foto
de referencia. Contexto de objeto configurado por procedencia medida, sin
palabras de enrutado; selección de cabeza conserva su recorte localizado.
Mostrar más contexto al modelo no autoriza escribir fuera de la máscara.

Ampliación SAM2 autorizada 2026-10-19 («SI»): evaluar el modelo local instalado
sin descargas con semillas medidas y propuestas automáticas de la foto. Guardar
alternativas, procedencia y tiempos. La confianza del modelo no demuestra
completitud: exigir inspección visual independiente y evaluar cobertura y
contaminación sobre núcleos diagnósticos congelados, nunca usados como entrada.
Integrar solo si supera la máscara actual en objetos completos sin invadir
vecinos; preservar selección múltiple, huella, protección y endpoints antiguos.
GrabCut sigue excluido y la calidad general requiere más de una foto.

## Kontext — proteger a las otras personas y retirar LongCat, 2026-10-18

Petición: revisar la edición «el hombre es calvo» y que no quede LongCat.
Comparación entrada/salida: Kontext convirtió a la mujer en un hombre calvo y
conservó el pelo del hombre original. El usuario confirma proteger automáticamente
a las otras personas mediante región medida, sin añadir pintado manual.

- [x] Los nombres y versiones legacy LongCat se resuelven a Kontext antes de
  cargar pesos; no se borran modelos, artefactos ni historia.
- [x] La receta de Kontext usa pasos y guía de su configuración, sin quedar
  absorbida por una rama legacy inaccesible.
- [x] Apariencia focal de una persona: seleccionar una cabeza por solape visual
  inequívoco, generar un recorte contextual con Kontext y componer solo su región.
  Sin selección o recorte fiables, conservar la entrada e informar la omisión.
- [x] Proteger las cabezas ajenas y el exterior después del suavizado y acabado;
  contratos con dos sujetos y edición adversaria de todo el recorte.
- [x] Conservar instrucciones compuestas, cámara y contratos Android 13/9/8.
- [x] Refuerzo «NO QUIERO HARDCODING»: ninguna regla por palabras/géneros/prendas
  nueva. Kontext no invoca las cuatro recetas legacy específicas de calvicie;
  conserva el texto traducido y usa solo metadatos del rewriter. El traductor
  detecta inglés antes de aplicar mappings españoles que corrompían palabras.
- [ ] Lint, compilación, reviewer APPROVE, tester PASS y prueba real no explícita
  con evidencia visual. No certificar la foto fallida por contratos CPU.

Decisión H-FS-005 (2026-10-17): el usuario confirma conectar Similitud al matching destino conservando exactamente los umbrales vigentes en el valor por defecto. Menor valor exige mayor similitud; selección de fuente y márgenes de ambigüedad intactos. Confirma retirar «Previsualizar Swap (Realtime)», conservando los previews existentes.

Trabajo activo: corrección de los fallos del proyecto autorizada el 2026-10-13 («arreglalo todo», confirmado: fallos del proyecto). Las historias siguientes conservan sus criterios y límites de validación.

## Completar código pendiente — 2026-10-18

Instrucción del usuario: «SIGUE HASTA DEJARLO SOLO PRUEBAS. TODO EL CÓDIGO».
Completar las correcciones pendientes en los flujos existentes y dejar una lista
verificable de pruebas reales y validaciones visuales. Conservar elecciones
vigentes de motor/modelo, compatibilidad Android y cómputo local. No instalar
paquetes/modelos, hacer commits ni reiniciar procesos activos.

- [ ] Cámara: las poses ya admitidas de altura, distancia y combinaciones de
  orientación se conservan mediante medición de conceptos, sin regex de usuario;
  parámetros de decisión en configuración y contratos de regresión.
- [ ] Editor/Generate: corregir fallos reproducidos del código pendiente y
  completar configuración de los defaults ya elegidos. Preservar máscaras
  manuales, instrucción completa, personas/regiones no solicitadas y APK.
  Aclaración posterior del usuario: las máscaras se habían eliminado de la
  edición global Kontext. No reintroducir pintado manual en la UI; escritorio
  y Android conservan su entrada Image y sus trece/ocho/nueve parámetros.
- [ ] FaceSwap: contrastar deudas con implementación vigente; completar los
  defectos de código reproducidos sin cambiar modelo/identidad ni recuperar
  soluciones rechazadas. Separar comprobaciones de código y calidad del modelo.
- [ ] Lint, compilación, contratos pertinentes, revisión independiente y bitácora;
  actualizar spec con pendientes actuales en vez de marcar como resueltos
  defectos visuales todavía observados. Las pruebas pueden descubrir más código
  necesario: un defecto de calidad conocido no se convierte en «solo pruebas».

## Continuación del Editor — 2026-10-18: contrato compuesto de Kontext

Continuación solicitada con «CONTINUA». Se verifica el contrato previo documentado
en `diagnostics/probe_editor_kontext_default_20261018.py`: Kontext sin máscara
manual recibe la instrucción compuesta completa, como edición global por prompt.
El splitter legado no debe convertirla en varias subediciones que omiten cláusulas
sin región confirmada. Se conservan el gate de cámara y las máscaras manuales.

- [x] Contrato CPU rojo→verde: rewriter con varias regiones, una generación
  Kontext y ambas cláusulas presentes; sin recursión por regiones.
- [x] Prueba GPU no explícita de sustitución de chaqueta por blazer roja:
  guardar traza, salida única y medición de rojo; la variación del lienzo por sí
  sola no acredita obediencia. Inspección visual y límites en MEMORY.md.
- [x] Lint y compilación afectados, regresión pertinente, reviewer y tester.
  No cerrar calidad general ni la revisión de cámara con esta comprobación.
- [ ] Blazer de corte exacto, casos adicionales y validación visual del usuario:
  la muestra genera cuero rojo, pero conserva corte y costuras de cazadora.

## Continuación de correcciones — 2026-10-13

Dentro de H-EDIT-003/H-EDIT-004 y las deudas ya declaradas, sin sustituir Turbo ni el modelo FaceSwap y sin instalar dependencias:

- [ ] Toda edición corporal localizada conserva el ancla espacial, tanto con parte medida como con persona de respaldo; se comprueba el workflow real.
- [ ] Las superficies no consultan objetos fijos ajenos a la instrucción; la protección de otras personas usa la evidencia visual existente. Ningún mensaje atribuye SAM2/GrabCut a una operación que no los ejecuta.
- [ ] Los compuestos conservan todas las zonas medidas de cada cláusula; una petición explícita de postura mantiene el contrato de permitir nueva geometría.
- [ ] El reintento corporal comprueba cambio dentro de la zona pedida; cambios colaterales no cuentan como aplicación. Los fixtures respetan el tamaño real de la ventana.
- [ ] Los controles existentes de Generate, FaceSwap y Animate corresponden a capacidades ejecutadas o explican su límite antes de iniciar trabajo.
- [ ] Lint/compilación, probes pertinentes, reviewer APPROVE y tester PASS, con logs de la versión final. Las mejoras visuales requieren muestras reales adicionales y validación del usuario; los contratos no las certifican.

Prioridad confirmada después: revisar FaceSwap por caras y calidad incorrectas en las salidas de `output`. Corregir el contrato general de geometría de H-FS-004: los cinco landmarks deben pertenecer a su bbox, independientemente de confianza y orientación. Reutilizar el margen de contención vigente, declarado en configuración; rechazar o rescatar desde una referencia válida antes de estimar la matriz. No sustituir modelo, identidad, escala ni umbrales de boca. Verificación con contratos CPU y retratos no explícitos; la revisión del lote no autoriza regenerar imágenes sexuales explícitas.


## H-GEN-001.1 — Controles avanzados en la UI (Fase 1)

**Como** usuario **quiero** fijar seed, steps, CFG y negative prompt **para** poder reproducir y ajustar corridas.

- [x] Aparecen 4 controles en `ui/tabs/gen/ui.py` (seed vacío = aleatorio; steps/CFG vacíos = valor de `config/model_configs.json`; negative vacío = el de config).
- [x] `on_generate_image` los pasa a `client.generate(...)` y llegan al `KSampler`/`CLIPTextEncode`.
- [x] Misma seed + mismos parámetros ⇒ misma imagen (verificado con 2 corridas reales y diff de píxeles ≈ 0).
- [x] Sin controles, el comportamiento previo es idéntico (probe existente en verde).

## H-GEN-001.2 — `color_grade` y resolución libre (Fase 1)

- [x] Desplegable de `color_grade` con las opciones de `gen_prompt_modifiers.py:63-69`; `auto` por defecto.
- [x] Sliders `width`/`height` visibles (512-1536, paso 64) sustituyen/augmentan la orientación.
- [x] El preview de modificadores refleja el `color_grade` elegido (no `"auto"` fijo).

## H-GEN-001.3 — Motores (Fase 1)

- [x] El desplegable incluye los alias ya soportados por `load()`: `longcat`, `longcat_full`, `flux_q2` además de los actuales.
- [x] El default fue `flux_dev_abliterated` hasta H-GEN-002; el 2026-10-05 se sustituyó por `flux_dev_q4` tras el fallo visual reproducido del modelo anterior y la instrucción del usuario de continuar los arreglos.
- [x] Un motor no instalado no aparece (comportamiento actual conservado).

## H-GEN-001.4 — Puente Gen→Edit (Fase 1)

- [x] Botón «Editar este resultado» junto a la imagen de Generate que carga esa imagen en la entrada del Editor y salta a su pestaña.

## H-GEN-001.5 — Imagen de entrada en Generate (Fase 2)

- [x] Campo de imagen opcional en Generate.
- [x] Si hay imagen y **no** hay instrucción de edición: img2img con referencia (`LoadImage` + `VAEEncode`, denoise < 1 tomado de config).
- [x] Si hay imagen **e** instrucción de edición (añadir/quitar persona u objeto, ropa, fondo, POV/ángulo, pose): se enruta a `generate_intelligent()` del Editor y el resultado se muestra en Generate. **Actualización 2026-10-16**: la ruta por defecto pasó de `auto` a `img2img` (`generate_ui.default_route` en `config/generation_prompt.json`) porque el usuario quiere Generate y Editor **paralelos e independientes**: Generate ya no enruta sola al Editor; `editor`/`auto` son elección manual en el desplegable. Verificado: UI construida con `route.value='img2img'`, adherencia 108/0 y contrato APK 31/31.
- [x] Sin imagen: txt2img como hasta ahora (regresión cero).

## H-GEN-001.6 — VRAM, progreso, calidad y muestras (Fase 3)

- [x] Tras generar, Generate libera la VRAM del servidor (`POST /autoauto/unload`) igual que el editor.
- [x] La barra de progreso muestra paso/ETA derivados de `/history` (este ComfyUI no expone `/progress`).
- [x] Selector de calidad de salida (hd / 4K) usando `image_quality_pipeline` como el Editor.
- [x] Selector de nº de muestras (1-4) con `batch_size` parametrizado.

## H-EXIT-RELEASE — al cerrar la webview con la X se libera la GPU y se cierra ComfyUI, 2026-10-20

Pedido literal del usuario: «cuando se cierra la webview desde la X debe liberar
la gpu siempre y cerrar comfuy». Confirmado con «siempre, sin excepciones» para
el caso de una ventana secundaria que reutiliza una sesión ya abierta.

- [x] Causa: `roop/core.py::run()` devolvía tras `webview.start()` sin tocar
  nada y el propio mensaje anunciaba «ComfyUI permanece disponible para otras
  sesiones». No había `comfy_launcher.stop()` ni liberación de VRAM local.
- [x] `roop/session_shutdown.py`: observa estado (VRAM por `nvidia-smi`, cola
  ComfyUI, batch local), da una gracia configurable, cancela el batch local con
  la bandera documentada si sigue vivo, detiene ComfyUI **siempre** y descarga
  los tenedores locales de GPU. Idempotente (la ventana y `atexit` no duplican).
- [x] Cableado en `roop/core.py`: `register_shutdown_hook()` antes de
  `webview.start()` y `shutdown_session("ventana-cerrada")` al volver de ella.
- [x] Política en `config/session_shutdown.json`; sin valores mágicos. Cada
  cierre escribe `logs/session_shutdown_<ts>.json` con antes/después.
- [x] Verificación real, tres rutas: propietario (tree-kill del PID tracked),
  secundario (`port 8188 freed`) y `atexit` en un proceso hijo. Puerto cerrado
  en las tres, informe JSON escrito en las tres (este último en `logs/`).
- [x] La primera versión falló en las tres comprobaciones de `atexit` y se
  corrigió con evidencia: (1) `RuntimeError: can't register atexit after
  shutdown` porque los imports perezosos dentro del callback estallan — ahora
  los módulos se precargan al registrar el hook y la descarga local solo opera
  sobre módulos ya cargados; (2) `UnicodeEncodeError` en consola cp1252 por
  `→`/`⚠`/`✔` — mensajes propios en ASCII y salida tolerante con reemplazo
  durante el cierre; (3) violación de acceso al salir con imports a medias,
  desaparecida con el precalentamiento.
- [x] VRAM real liberada: FaceAnalysis cargada en la GPU del proceso principal
  y soltada por `_release_local_gpu()` → dos mediciones, `1588→994 MB`
  y `1609→999 MB` (Δ ≈ −600 MB).
- [x] Ruff y `py_compile` verdes en los tres archivos; 73/73 comprobaciones PASS
  en `diagnostics/probe_session_shutdown_20261020.py`.

Evidencia: `diagnostics/out/session_shutdown_20261020/` (informes + results.json)
y `logs/session_shutdown_*.json` de la ruta de producción.

## H-EDIT-SELECT-NEARBY — la selección regional no debe tocar a la persona vecina, 2026-10-20

El usuario pide que al elegir una cabeza en el Image Editor se edite SOLO esa
persona: el vecino cercano debe quedar intacto. Requisito derivado del defecto
medido del 18/10 (referente equivocado) y del trabajo de protección focal.

- [x] Causa raíz medida y corregida: `head_crop_context_fraction` (0,15) estaba
  en la configuración pero `selected_region_edit.py` lo ignoraba y aplicaba el
  contexto genérico (0,35). Contexto de cabeza ceñido, con validación de
  configuración inválida.
- [x] Cegado de las cabezas NO seleccionadas dentro de la referencia que ve
  el modelo, sin borrar píxeles de la cabeza seleccionada (resta de máscara,
  dilatación con radio configurado y restauración del halo). Si no hay nada que
  cegar, la referencia no se altera.
- [x] La composición final usa el recorte ORIGINAL como respaldo, nunca la
  referencia cegada: el parche gris no puede llegar a la foto.
- [x] Verificación real end-to-end con FLUX Kontext sobre foto real, en las dos
  direcciones: seleccionando `head-2` cambian 17.809 px y `head-0`/`head-1`
  quedan con 0 px; seleccionando `head-0` cambian 9.194 px y `head-1`/`head-2`
  quedan con 0 px. En ambos casos 0 px de fuga gris y 0 px cambiados fuera de
  la máscara efectiva. Una generación, sin máscara de modelo.
- [x] Contratos CPU verdes: protección de vecino 4/4 y contrato de selección
  15/15. Ruff limpio en los archivos tocados.

Evidencia: `diagnostics/out/editor_nearby_head_20261020/1792483031212730200/`
(seleccionada `head-2`) y `.../1792483175438315400/` (seleccionada `head-0`).
Bitácora en `MEMORY.md`, sección "la selección debe editar SOLO la cabeza
elegida y no al vecino cercano".

## H-EDIT-SELECT-BODY — la selección regional debe obedecer el cambio corporal medido, 2026-10-21

El usuario reporta que al seleccionar una persona y pedir «la mujer esta
desnuda, sin zapatos» la ruta de selección devuelve **una copia de la foto con
la ropa intacta**. Decisión del usuario: reutilizar el **régimen medido** de la
ruta global y mantener el **soporte medido estricto** (escribir solo dentro de
la máscara medida; el exterior queda protegido píxel a píxel). El usuario
rechaza explícitamente cualquier lista de palabras escrita a mano: la decisión
debe salir de los conceptos ya medidos.

- [x] Diagnóstico medido sobre el render real del 21/03 (`edit_1792523032.png`):
      la referencia recibida por Kontext llevaba **4.358 px grises sobre la cabeza
      de la mujer seleccionada** (el cegado ignoraba la regla de inclusión que
      la máscara de protección ya aplicaba) y la ruta usaba un régimen **fijo**
      (EDIT con ancla `offset`, sin negative por dirección y sin frase de
      estado), que con tokens limpios copia la foto (Q4+offset=copia).
- [x] Cegado y protección usan **la misma regla medida** de inclusión
      (`head_inclusion_coverage`): la cabeza que pertenece al sujeto
      seleccionado llega visible al modelo y sigue siendo editable; las demás
      cabezas del recorte se ciegan igual que antes.
- [x] La ruta regional **mide** la instrucción con el mismo analizador de
      conceptos que la ruta global y reutiliza su régimen: negative por
      dirección, frase de estado final, método de referencia y cláusula de
      enforcement. Cero vocabulario de usuario en el código.
- [x] La cláusula de enforcement («el cambio pedido debe verse… no conservar
      el estado anterior») es un único texto compartido por ambas rutas.
- [x] **Medición del motor sobre la foto real** (la que falló el 21/03), que
      corrigió la hipótesis inicial «reutilizar el motor de la ruta global»:
      - con ancla, el cambio **no se ejecuta**: la generación sale registrada
        (correlación 0,999; desplazamiento 0 px) pero con la ropa intacta;
      - sin ancla (img2img puro, la receta global) sobre el **recorte
        entero**, la generación **recrea otra escena** (correlación 0,625;
        desplazamiento 24 px) y al componer solo la zona medida el
        resultado **no encaja** con la foto original.
      - La vía correcta es el modo «foto» del cliente: la **zona medida** es
        la que se regenera (ruido solo dentro de la máscara, nodos 14/15/16)
        y el resto de la foto lo conserva el muestreador. Con ella el cambio
        sí se ejecuta (torso 41,9-49,0 niveles de diferencia media; el
        resultado fallido anterior era 4,3) y el exterior queda con **0 px**
        de diferencia.
- [x] La zona que se **regenera** se mide aparte de la que se **escribe**:
      con la silueta entera el modelo re-inventa a la persona (rostro y pelo
      distintos). La zona correcta es la **ropa medida** (segmentador de
      ropa) sin la cabeza medida, y la escritura sigue siendo el soporte
      medido estricto.
- [x] **Límite medido, no resuelto**: con el modelo Q4 el texto no siempre
      lleva a piel desnuda; en varias corridas sustituye la prenda por otra
      (jersey, tank top). Más esfuerzo tampoco ayuda (28 pasos y guía 4,5:
      46,0 niveles frente a 44,7 de la receta base). Lo que sí queda
      probado es que la ruta deja de devolver **una copia de la foto**: la
      zona medida cambia de verdad (torso 44,7 y 46,8 niveles con dos
      semillas fijas, frente a 4,3 del resultado fallido) y el exterior queda
      con **0 px** de diferencia. La ruta global con su propia receta sobre
      la MISMA foto tampoco lograba la piel (generaba un personaje anime,
      100 % de píxeles cambiados): la referencia del usuario era de
      **enrutamiento**, no de calidad.
- [x] **Causa medida del límite**: en FLUX.1/Kontext el prompt **negativo
      es inerte** (guía en el nodo FluxGuidance, muestreo a CFG=1), así que el
      negative por dirección heredado de la ruta global no puede cambiar el
      resultado en este motor; solo cuenta el texto positivo. Medido con la
      participación de piel en la zona de prenda medida: original 0,2675 →
      0,0676 con el texto medido, y 0,0740 con solo la frase de estado
      (sin plantilla ni enforcement): el modelo **viste más**, no menos.
      Y el conflicto medido del motor es de ancla: sin ancla el modelo
      regenera la escena (registro 0,625, desplazamiento 24 px → collage);
      con ancla y ruido acotado a la zona de prenda conserva la ropa
      (registro 0,999). La receta que hace executes el cambio es la que
      rompe el encuadre.
- [x] **Vía ejecutada: el motor CON VISIÓN para el cambio corporal medido**
      (2026-10-21, v5.1229-v5.1230). La causa anterior era del motor, no de la
      ruta: el negative es inerte en Kontext y el texto solo no basta. El motor
      con visión (Turbo + Qwen-VL: el modelo **ve** la foto y el texto manda) es
      el único del proyecto medido que sigue una instrucción de estado final.
      - El retiro de LongCat (2026-10-18) normalizaba en el cliente
        (`load()`) cualquier motor con visión a Kontext, de modo que la ruta
        global llevaba desde el retiro ejecutando Kontext en su «Turbo».
        Ahora hay una **excepción explícita y estrecha**
        (`load(…, allow_vision_engine=True)`), usada **solo** por la ruta de
        selección y **solo** para la dirección medida «quitar»; el resto del
        retiro se mantiene (`probe_editor_retired_engine_20261018` en verde).
      - Receta medida: motor con visión, `prompt_authority=True`,
        `no_kontext=True` (sin ancla de latentes), 12 pasos, CFG 1,0, denoise
        1,0, **sin agujero**, y el texto = la composición medida de la ruta
        global + el enforcement compartido + el negative medido dentro del
        texto (con CFG=1 el negative no tiene efecto propio).
      - Escala medida (participación de piel en la zona de prenda medida;
        original 0,2675): Kontext 0,0676 (vestía más) → con agujero 0,4346
        (el modelo completa el atuendo con ropa de playa) → **sin agujero
        0,5299** (la persona desnuda y sin zapatos).
      - **Decisión del usuario (2026-10-21): cumplimiento total, identidad
        sacrificada.** El cambio corporal medido en dirección «quitar» se
        ejecuta con el motor con visión y **reinterpreta el marco**: se
        entrega el marco reinterpretado (no el recorte pegado dentro del
        soporte, que medido es un collage roto). El resto de instrucciones
        mantiene el **soporte medido estricto** (exterior píxel a píxel).
      - **Con otras personas en la foto el marco NO se reinterpreta**
        (v5.1231): la receta de reinterpreitación redibuja también las caras
        de los demás (medido en la foto de dos personas del corpus: 85,9 y
        66,3 niveles de diferencia, 99 % de los píxeles de las dos caras) y
        ese fue el motivo real del retiro de LongCat del 18/10. Con otras
        personas medidas, la zona de ropa medida es el agujero y la foto de
        alrededor sobrevive: las otras personas y el encuadre quedan
        intactos, con cumplimiento parcial (el modelo se queda en ropa de
        playa: piel 0,4346 frente a 0,5299 reinterpretando).
      - Render real e inspección visual: la persona sale desnuda y sin
        zapatos en una foto coherente; la identidad y el encuadre **no** se
        conservan (cabeza 105,9 niveles de diferencia), como pidió el usuario.
      - **No hay censura en el motor con visión** (medido: obedece el estado
        final pedido). Lo que impedía el cumplimiento en Kontext eran dos
        cosas medidas: el negative inerte (CFG=1) y el modelo completando el
        atuendo al rellenar un agujero. Equivalentes sin censura ya
        instalados: LongCat Turbo (el que usa la ruta), LongCat **Full**
        (más calidad, ~180 s), Qwen-Image-Edit **2511** y el **LoRA NSFW**
        de Qwen-Image-Edit-Plus.
- [ ] **Límites que quedan, y su causa medida** (no son defectos de la ruta):
      - con agujero (la variante que conservaba identidad y encuadre) el
        modelo se queda en ropa de playa: 0,4346 frente a 0,5299 sin
        agujero. La variante con agujero **sigue disponible** como receta
        medible si en el futuro se quiere el otro compromiso;
      - el motor alternativo con visión medido (Qwen-Image-Edit, ya en disco)
        **no** sirve para este cambio: con el mismo agujero vestía más
        (0,1414) y salía casi copia (registro 0,975) a 1220 s por pase;
      - la cláusula «sin zapatos» de la ruta **global** sí se cumple porque
        esa ruta regenera el marco entero; en la ruta de selección, el
        cumplimiento llega por la misma vía (el marco reinterpretado).
- [x] **La ruta global (sin selección) también obedece el cambio corporal
      medido** (v5.1232). Antes respondía con un personaje anime (100 % de
      píxeles cambiados) y la ropa intacta. Causa medida: esa ruta medía la
      intensidad corporal **en plano** (0,20) y la frase de estado final exige
      ≥ 0,30, así que el estado medido **no llegaba al modelo**; con la
      medición **por cláusula** (0,745, la misma que usa la ruta de
      selección) sí llega. Medido: piel 0,2101 → **0,5583** (desnuda y sin
      zapatos; original 0,2675). La medición por cláusula alimenta solo la
      frase de estado y el motor del cambio corporal: el enrutado global
      conserva sus guardas medidas. La salida vuelve con la geometría de la
      foto de entrada (el motor con visión entrega a 1 MP). Contrato CPU
      `probe_editor_global_body_prompt_20261021` 7/7.
- [x] **La cabeza seleccionada con una petición de apariencia también tiene que
      obedecer** (v5.1232). Caso real del usuario (log de la UI 22:14, foto
      3000×2000 con dos personas, cabeza `head-1` seleccionada, «debe ser
      calvo»): la ruta fue a **Kontext con ancla** (`EDIT ref=offset`, 20
      pasos) y **no editó nada** — el ancla copia la foto. Es el mismo
      copy-machine del cambio corporal, y la excepción del motor con visión
      solo cubría el cuerpo.
      - Regla: un cambio de **apariencia medido** (`local_appearance`) sobre
        una **cabeza seleccionada** va al motor con visión (sin ancla, con la
        cabeza medida como agujero y la receta medida de calvicie en el texto,
        la que evita el «gorrito» documentado).
      - Verificado sobre la MISMA foto e instrucción: pelo **108,2 → 204,2**
        (calvo de verdad, sin gorrito ni pelo blanco), **0 px** fuera de la
        cabeza, 101,7 s. El precio medido: la cara de la cabeza repintada
        cambia (**54,4** niveles: el modelo rehizo la pose y la expresión).
- [x] **Una instrucción que el analizador NO mide no es un pedido débil**
      (v5.1233). Caso real del usuario (log de la UI 00:09, la misma
      selección, «el hombre de la derecha debe ser **cxalvo**»): el
      analizador no cubre esa palabra → la instrucción queda sin medición
      (`body`, `local_appearance`, `local_region`, `pose`, `authority` y
      `structural_remove` todos False) → la ruta usaba la receta anclada → el
      ancla copiaba la foto → el pelo se quedaba igual. Con la ortografía
      correcta (`attribute_appearance` 0,189) la ruta mide y edita.
      - Regla: **señal cero = señal, pero avisado**. La ruta de selección
        escribe solo la zona medida, así que no puede tratar una instrucción
        sin medir como una señal fuerte (eso rompe los casos validados de
        calidad, ángulo y postura: medido, el contrato pasó de 15/15 a
        13/15). Lo que sí hace es **decirlo**: el resultado avisa «sin señal
        medida en la instrucción (revisa la ortografía o el vocabulario:
        puede que el motor no la ejecute)». Un no-op silencioso pasa a ser
        visible.
      - Descartado con medición: el reintento por no-op con el detector del
        proyecto (no cazaba la errata —el modelo mueve 649 px en la cabeza—
        y rompía el contrato de las referencias estrechas).
      - Contrato CPU `probe_editor_selection_body_regime_20261021` **8/8**;
        selección 15/15, object-context 4/4, UI 14/14, lienzo 7/7, motor
        retirado y default en verde.
- [x] **La cabeza: mejor rehacer la escena que un parche malo** (v5.1234).
      Caso real del usuario (log de la UI 01:40, su foto 800×511, cabeza
      `head-0` seleccionada, «EL HOMBRE DE LA IZQUIERDA DEBE SER CALVO»): la
      ruta fue al motor con visión (la instrucción se midió y se ejecutó) pero
      el resultado fue **una edición pésima**: con la cabeza entera como
      agujero el modelo **sustituyó la cabeza medida por una inventada**.
      Las otras recetas medidas tampoco sirven para la cabeza: con agujero
      solo en el pelo sale gorrito blanco o pelo blanco; con Kontext anclado
      no cambia nada.
      **Decisión del usuario (22/10)**: el cambio de apariencia sobre la
      cabeza seleccionada **reinterpreta el marco**, igual que el cambio
      corporal (motor con visión, sin ancla, sin agujero, texto medido con la
      receta de calvicie, negative dentro del texto) y conserva las caras
      medidas de las otras personas.
      - Pero el usuario exige además que **la selección funcione** («lo de
        la selección de formas detectadas no funciona»): reinterpretar el
        marco ignora la zona marcada, así que esa vía no vale aquí.
      - **Receta final (v5.1235)**: la **focal** de la ruta global — el
        recorte con su contexto **escalado a buena resolución**, la edición
        **anclada** y el texto medido de apariencia. Es la única medida que
        edita una cabeza en su sitio, con la misma persona y la misma
        escena, y con **0 px** fuera de la zona marcada. Verificado sobre la
        foto del usuario: pelo 98,0 → 132,0, 4.388 px dentro de la cabeza y
        **0 px** fuera.
      - Escala medida de las cinco recetas (misma foto): lienzo + ancla → no
        edita; motor con visión + agujero en la cabeza → otra persona
        («edición pésima»); motor con visión + marco reinterpretado → ignora
        la selección; recorte sin escalar o img2img puro → inventa el
        fragmento; **recorte escalado + ancla → en su sitio, misma persona**.
        Lección: la **resolución del recorte** es la variable (una cabeza de
        136 px en una foto de 2841 hace que el modelo invente) y el **ancla**
        es lo que conserva a la persona.
      Contrato CPU: el cambio de apariencia no lleva lienzo, la referencia es
      el recorte escalado (la zona marcada) y lo que se escribe es solo el
      soporte medido. Pendiente: la verificación real sobre la foto del usuario
      (la GPU estaba ocupada con un render suyo).
- [x] Contrato CPU fail→pass: `probe_editor_selection_body_regime_20261021`
      **6/6** (el cambio corporal medido carga el motor con visión de forma
      explícita, sin ancla, con el texto medido; con otras personas medidas
      se conserva el marco y sus caras; sin ellas se reinterpreta el marco;
      edición ordinary con su receta Kontext, su exterior píxel a píxel y la
      cabeza seleccionada no cegada) y testigo
      `probe_editor_selection_body_regime_witness_20261021` que demuestra que
      los **tres** comportamientos anteriores **fallan** ese contrato. Suite
      de selección existente 15/15, UI 14/14, object-context 4/4, lienzo 7/7,
      motor retirado en verde. Ruff y compilación limpios.

Evidencia: `diagnostics/out/selection_regime_real_20261021/` (render real sobre
la foto original con la instrucción que falló) y bitácora en `MEMORY.md`.

## H-EDIT-SELECT-POSE — la selección y la ruta global obedecen el cambio de ángulo/postura medido (v5.1238/v5.1239), 2026-10-22

**Como** usuario **quiero** que «CAMBIA EL ANGULO, IMAGEN DE LATERAL» (y
«que se vea de perfil», «ponlo de lado») rehaga el encuadre pedido **para**
que la instrucción no vuelva una foto intacta (copy machine).

Hechos medidos (2026-10-22): la instrucción caía en la ruta ordinaria
(Kontext + lienzo + ancla) porque `pose` medía False. Causa raíz: el
sustantivo del MEDIO («imagen/foto/image») estaba dentro de los textos de
concepto de `structural`, `structural_remove` y `attribute_appearance`
(«from photo», «from image», «cambio de imagen»); `_score_texts` normaliza
por palabras del prompt, así que una sola palabra genérica daba +0,25 a
cualquier petición que mencionara la foto (structural_remove 0,2505) y la
competición de pose (≥1,25× del resto) era inalcanzable. Separado: la
sesión del 19/10 quitó sin commitear el ancla POV de
`CAMERA_POSES["azimuth"]["front view"]` sin actualizar el contrato («cambia
el pov» → 0,000 → default «front-right quarter view elevated», probe 3 FAIL).

- [x] Textos de concepto sin el sustantivo del medio en `structural`,
      `structural_remove` y `attribute_appearance` («cambio de imagen» →
      «cambio de look»), y `CONCEPTS["pose"]` extendido EN+ES con ángulo de
      cámara/lateral/perfil (texto de concepto; precedente dance/dancing).
- [x] Medido: ángulo/perfil/lado/de pie → POSE=True; calvo y «que mire a la
      cámara» → POSE=False/LOCAL=True sin regresión; dance/vestido 6/6.
- [x] Rama `pose_change` en la selección (v5.1238): motor con visión, foto
      completa, sin ancla ni agujero, marco reinterpretado, caras de las
      demás personas conservadas. Contrato nuevo
      `pose_change_uses_the_vision_engine` → suite **9/9 PASS**.
- [x] Ruta global sin selección: `explicit_viewpoint_request` no secuestra
      la petición; con kontext va a **Qwen Camera** con la pose entrenada
      medida (`<sks> left side view eye-level shot medium shot`, azimuth
      0,231 ≥ umbral 0,12) y si Qwen falla devuelve mensaje honesto.
- [x] Ancla POV restaurada con comentario de contrato → probe de cámara
      **10/10**, `probe_editor_viewpoint_dispatch` rc=0.
- [x] Batería: selección 9/9, witness OK, protección 16/16, kontext_default
      OK, adult_gate PASS, body_clause_guard PASS, ruff limpio.
- [ ] Render real del ángulo sobre la foto del usuario + inspección visual
      (requiere reinicio de la app para cargar Generate + pose + conceptos +
      ancla POV).
- [ ] Deuda medida, decisión pendiente (commit `9b5b6d0`, preexistente):
      el alias `longcat→kontext` (6104) deja inalcanzable la ruta Turbo
      (6388-7634) → 2 probes rojos (`probe_editor_added_face_routes_20261018`,
      `probe_editor_compound_fast_dispatch`) y la compuerta de caras
      añadidas v5.1220 sin efecto para kontext (la integridad visual solo
      avisa, no restaura): portar la compuerta a la ruta ordinary o aceptar
      el riesgo documentado.

Evidencia: `MEMORY.md` (entrada 2026-10-22 de esta historia) y contratos en
`diagnostics/probe_editor_selection_body_regime_20261021.py`.

## H-EDIT-SELECT-V — Guard de visión Moondream (2026-10-20)

**Como** usuario quiero que el análisis visual nunca aborte la aplicación, para
que un fallo de GPU o de VRAM se reporte como mensaje y no como caída nativa.

El proyector LLaVA instalado no expone selector de dispositivo
(`clip_model_load(path, verbosity)`), así que el backend ggml decide. Con el
backend de CPU la fase de visión muere con `STATUS_STACK_BUFFER_OVERRUN`,
que no puede capturarse como excepción Python. El modelo de texto en CPU es
seguro.

Criterios de aceptación:

- [x] Antes de crear el handler de visión, un preflight comprueba que el
      backend ggml expone descarga a GPU (`llama_supports_gpu_offload`).
- [x] El preflight comprueba que la VRAM libre cubre el proyector; la fuente
      es `nvidia-smi` (usada/total), porque `torch.cuda.mem_get_info` no ve la
      VRAM retenida por otros procesos.
- [x] Si el preflight falla, no se carga ningún modelo y se devuelve un
      mensaje claro; el proceso sobrevive.
- [x] Aplicado a las dos rutas de producción que usan el proyector:
      `element_selection._moondream_description` e
      `instruction_verifier.InstructionVerifier._load`.
- [x] El rechazo no se degrada a «análisis parcial»: `VisionUnavailableError`
      se propaga al llamador.
- [x] Parámetros en `config/editor_selection.json` → `vision_guard`
      (`enabled`, `vram_margin_mb`), sin valores mágicos en el código.
- [x] Verificado con ejecuciones reales: sin CUDA, con VRAM insuficiente, con
      GPU ocupada por otro proceso (rechazo y salida limpia), y con GPU
      libre (descripción real). Sin regresión del análisis de producción.

## Criterios comunes de la historia

- `py_compile` + probes de Generate existentes en verde; Image Editor y FaceSwap sin regresión.
- Prueba real de Generate tras cada fase, con imagen guardada en `generate/` y anotada en `MEMORY.md`.
- Parámetros nuevos con default en `config/`, sin valores mágicos.

## H-GEN-002 — Adherencia al prompt y control NSFW

**Como** usuario quiero que Generate respete el sujeto y estilo pedidos, y que el contenido NSFW solo se active con un control manual, para evitar escenas no solicitadas.

Decisiones ya aprobadas por el usuario el 2026-10-04: checkbox NSFW manual, gate de estilo por conceptos medidos sin listas de palabras y verificación mediante probe y render real.

- [x] El checkbox NSFW llega al ensamblado de prompt y filtra prefijo y LoRAs adultas cuando está apagado (probe de ensamblado 52/0).
- [x] El gate de estilo omite añadidos fotorreales ante una petición no fotorreal (probe de ensamblado 52/0).
- [x] La petición CASTILLO CON UNICORNIOS, ESTILO DIBUJO produce visualmente castillo y unicornios en estilo dibujo con `flux_dev_q4` y NSFW apagado: semillas 123456, 234567 y 345678, salidas en `diagnostics/out/gen_castillo_style/`. Con `flux_dev_abliterated`, OFF/ON fallaron con la misma semilla.
- [x] La configuración normal de la UI selecciona `flux_dev_q4` y abre NSFW apagado; ambos defaults viven en `config/generation_prompt.json` y la UI construida devolvió esos valores.
- [ ] El contrato de diez parámetros de la APK, Image Editor y FaceSwap siguen funcionando; el endpoint `/generate_image` respondió por HTTP con diez argumentos y la APK debug compiló, pero falta instalarla en un teléfono y la validación visual de Editor/FaceSwap continúa abierta.
- [x] Render real por la ruta servida `/generate_image_ui`: `generate/gen_1791179741.png`, 768×512, seed 123456, castillo y unicornio en ilustración, inspeccionado visualmente; 20/20 pasos ComfyUI y estado «Listo».
- [x] Review final y verificación independiente de Generate: reviewer APPROVE y sd-tester PASS de la versión final (72/0 y 31/0 repetidos; PNG abierto y verificado; APK debug y SHA256 comprobados). El tester no repitió el HTTP servido ni probó la APK instalada en un teléfono.
- [x] A/B real sobre el motor por defecto (2026-10-16): `probe_gen_castillo_real.py` parametrizado por motor (`--engine`, default `get_default_generation_engine()`); `flux_dev_q4` semilla 123456 con NSFW OFF/ON produce salidas **byte-idénticas** (SHA256 `7db9c50e…`; el ON lo sirve la caché de ComfyUI porque el workflow final es el mismo) y la imagen es el castillo con unicornios en estilo dibujo. El checkbox ya no secuestra el prompt no adulto en la ruta por defecto. Con `flux_dev_abliterated` el prompt también llegó idéntico ON/OOFF, pero ese motor renderiza una mancha adulta incluso con OFF (cautela documentada; no es el default).

## H-GEN-003 — LoRAs instaladas y compatibles en Generate (2026-10-17)

Continuación autorizada: revisar las LoRAs locales antes de descartarlas/copiar (sesión OpenCode del 17/10) y «continua y solucionemos los problemas de Generate». Corrección de disponibilidad y compatibilidad; no cambia el motor por defecto ni los gates del prompt. El usuario está probando POV: no reiniciar UI ni ocupar la GPU para estas comprobaciones.

- [x] El selector ofrece únicamente archivos instalados compatibles con el motor elegido; cambiar de motor conserva una selección válida o vuelve a Auto.
- [x] Una selección manual ausente, inválida o incompatible devuelve error antes de subir imágenes o enviar workflows. La elección válida conserva la fuerza del catálogo y no depende del clasificador de intención.
- [x] Las dos LoRAs de estilo ya existentes en WebUI se copian sin sobrescribir a la carpeta de ComfyUI, con hashes comprobados; el catálogo declara SDXL/Pony y selección manual, sin añadirlas automáticamente a escenas.
- [x] Auto explica por consola cuándo el motor no admite el catálogo automático SDXL; ninguna selección manual se ignora en rutas delegadas.
- [x] Los cuatro modificadores extra seleccionados llegan al ensamblado (límite en configuración); no desaparecen plano/color por los defaults de luz/piel. Las opciones incompatibles se ajustan en los cuatro selectores y el preview inicial muestra los defaults.
- [x] LongCat conserva CFG real del JSON/override; FLUX.1 mantiene FluxGuidance y su negativo a CFG=1 se explica en UI/logs. HD/4K se identifica como perfil de acabado, sin prometer una resolución final que el flujo no garantiza.
- [ ] Contratos de disponibilidad/compatibilidad rojo→verde, regresión Generate/APK, lint y compilación verdes; reviewer APPROVE y tester independiente. La carga GPU y el efecto visual siguen sin verificarse hasta prueba real en un momento que no interfiera con el Editor.

## H-FS-005 — composición y controles FaceSwap (2026-10-17)

Alcance autorizado por «SOLUCIONALO» tras la auditoría del 17/10. Conservar modelos, identidad origen, selección destino, umbrales vigentes y funcionamiento de Generate/Editor/Android; no instalar dependencias ni reiniciar la UI activa. Las decisiones sobre el slider «Similitud» y preview realtime se consultan al usuario mientras avanzan las correcciones independientes.

- [x] Componer una segunda cara conserva swaps previos fuera de su zona, también ante saltos/errores de esa cara; el original sigue siendo la referencia para detección, oclusión y color (contratos CPU).
- [x] Vídeo principal y fallback reciben el valor real de suavizado temporal desde UI/core; llamadores antiguos conservan el default existente (contratos CPU).
- [x] Auto-Rotar desactivado omite solo pasadas rotadas, conservando detección original y rescates de escala; activado conserva el comportamiento anterior (contratos CPU).
- [x] El rescate frontal localiza componentes de la máscara ya medida, exige evidencia oral/anillo y cobertura vigentes y se abstiene si no hay un único candidato; no repite el mismo detector ni amplía el gate del 40 % (contratos CPU).
- [x] Código de vídeo usa máscara alfa suavizada conservando textura del parche; no desenfoca los píxeles del swap en la transición (contratos CPU).
- [ ] Verificar un vídeo completo permitido, sin equiparar un clip corto con certificación de perfiles/cortes largos.
- [x] Controles de similitud/preview corregidos según la decisión del usuario del encabezado, sin un éxito simulado ni controles que aparenten funcionar (Gradio y callback reales aislados de inferencia).
- [ ] Probes rojo→verde, lint y compilación afectados verdes, reviewer APPROVE y tester PASS. Registrar inferencias reales y evidencia visual permitida, límites y pendientes en MEMORY.md.

## H-FS-003 — Perfil, halo y selección del enhancer

Alcance confirmado el 2026-10-05 por el usuario: «TODO» y «sigue+», tras presentar FaceSwap, Editor y revisión de logs.

Comparación aislada aprobada explícitamente: descargar Hyperswap1c256 oficial y compararlo con la fuente y el perfil de referencia. Descarga y comparación completadas; similitud medida y definición mejoran, pero la marca deformada de frente permanece incluso antes del enhancer. La prueba no autoriza cambiar el modelo de producción ni cierra la historia.

- [x] La vista previa inicializa el enhancer elegido aunque la selección global anterior fuese otra; ejecución real y salida guardada en `diagnostics/out/faceswap_preview_CodeFormer_1791184533405190500.png`; reviewer APPROVE y sd-tester PASS (2026-10-05).
- [x] Los logs de calidad nombran el procesador ejecutado, incluido el fallback, y la dimensión real de su salida; derivación desde `enhancer_key` revisada y CodeFormer 512 px verificado en el log real.
- [ ] El perfil de `1994859933.jpg` mejora visualmente en proporción, nitidez y parecido con la fuente; se conservan oclusiones y personas ajenas.
- [ ] El cambio visual se verifica en fotos adicionales y los vídeos representativos completos; un clip corto o una métrica de cambio no cierran el defecto.
- [ ] Probes pertinentes y compilación Python verdes, revisión APPROVE y verificación PASS; evidencia y límites en MEMORY.md.

## H-EDIT-003 — Edición localizada estable con Turbo

Mismo alcance confirmado el 2026-10-05. Se conserva la elección previa de Turbo y la localización por conceptos y regiones medidos.

Recordatorio del usuario el 2026-10-05: el Editor debe ser versátil porque puede pedir cualquier edición. «Calvo» es un caso de reproducción del defecto; la corrección debe conservar la capacidad general de interpretar instrucciones.

Diseño aprobado el 2026-10-05: cuando una edición de región no encuentra zona existente, el modelo de lenguaje ya instalado propone una zona mencionada en la instrucción y el segmentador la comprueba en la foto. El usuario eligió «Sí, modelo local y comprobación de la zona» tras revisar la alternativa de añadir una taza sobre la mesa. No sustituye la instrucción ni descarga modelos.

Ampliación aprobada explícitamente después: descargar y probar Qwen2.5-3B-Instruct Q4_K_M para interpretar zonas (2,1GB, repositorio oficial Qwen). Si mejora la evidencia real, su configuración se limita a propuestas de región; los clientes de Generate y el rewriter general conservan sus ajustes. Parámetros y ruta del modelo en configuración, sin descarga automática desde el flujo del editor.

Continuación aprobada por el usuario el 2026-10-05 con «continua», tras presentar la propuesta y solicitar aprobación: usar el mismo modelo3B para traducir la instrucción completa únicamente en la ruta de región ambigua. Diagnóstico CPU inicial de nueve instrucciones conserva objetos/ubicaciones y evita planta→floor, pero suaviza modalidad en dos casos; un prompt posterior mantiene must/must not en diez casos reales (`region_llm_1791222472492290000.json`). Mantener medición de la región desde RAW y abstención si falla, sin sustituir el traductor general ni afirmar calidad visual sin render real.

- [x] En la ruta ambigua, la región se mide desde RAW y solo la instrucción completa traducida con el modelo local configurado llega al generador. Modelo cerrado antes de ComfyUI; ausencia, error, respuesta vacía o truncada conservan la foto con mensaje explícito. Resto de rutas conserva su traductor. Reviewer APPROVE acotado y tester PASS de integración/contratos (2026-10-05).
- [x] La traducción integrada conserva acción, objetos, ubicaciones, obligatoriedad y negación en diez pruebas CPU reales (`region_llm_1791223077064804600.json`); manager real `versatility_remove_plant_translated3b_20261005` manda plant y conserva personas, pero eliminación parcial con manchas, FAIL visual (91s/1,25 %). Este criterio verifica la conexión y su evidencia; no cierra la calidad de la historia.

Continuación aprobada por el usuario tras la propuesta: «quiero que sigas para solucionarlo todo». Se explicó que añadir/quitar objetos son las ediciones estructurales: extender la receta8/referencia espacial ya usada para apariencia y recolor a esas ediciones con región medida. A/B aislados con misma ventana/máscara/prompt/seed: planta45s elimina mejor copa y manchas, pero conserva restos; taza43s añade taza con asa más visible, personas conservadas. Tester PASS acotado de mejora en una escena, sin calidad universal. Evidencia `plant_native_recipe_1791223360498195100`, `cup_native_recipe_1791223582924147400` y sus composiciones `local_replay_compare_1791223496827217600`/`1791223740393913300`. Parámetros en configuración de región, sin cambiar rutas globales ni decidir por vocabulario; verificar manager completo y continuar cobertura del objeto.

Ampliación de pruebas aprobada explícitamente el2026-10-05: «Sí, descargar y probar ambos» para SAM2.1HieraTiny156MB (repositoriooficialMeta) y Hyperswap1a256384MB (FaceFusionoficial). SAM2 se compara desde puntos derivados de máscaras reales y guarda todas las alternativas: comprobar objeto completo, exclusión de otraspersonas y tiempo, con pesoslocales y sin paquetesnuevos. Hyperswap1a reutiliza alineación/normalización/probeCPU revisado y se compara con1c/baseline sobre perfil/frontal e identidad. Esta aprobación de descarga/prueba no declara mejora ni sustituye producción; integrar solo tras evidencia visual/review y decisión concreta del usuario.

- [ ] Si la medición directa de una edición estructural falla, se aceptan únicamente propuestas del modelo local presentes en la instrucción y medidas por el segmentador; una región pequeña válida puede servir de soporte. Parámetros en configuración y abstención si no hay zona.
- [ ] La ruta integrada añade un objeto en una foto real sin sustituir las personas ni el resto de la escena; se prueba otra petición estructural y los rechazos de propuestas inventadas, ausentes o saturadas.
- [ ] Un fallo de carga o generación Turbo devuelve el error sin cambiar silenciosamente al motor Full.
- [ ] Una propuesta de zona que cubre una cara existente no puede usar a esa persona como superficie para insertar otro objeto; la continuación conserva el mismo rechazo.
- [ ] El router corporal recibe los conceptos de vista/postura medidos por el router general, sin variables no definidas al usar Kontext.
- [ ] Si los conceptos medidos sugieren una región pero empatan con apariencia corporal sin pedir cuerpo/postura, se comprueba una propuesta basada en la instrucción original antes de generar. Sin región verificada, no se repinta la foto entera; una traducción incorrecta no sustituye esa instrucción.

- [ ] La decisión de ruta se apoya en conceptos y regiones medidos, sin listas de palabras, prendas, géneros ni tipos de petición escritos a mano. Se verifican ediciones reales distintas de quitar pelo, además de los contratos de despacho.
- [ ] Si el segmentador mide una parte solicitada dentro del sujeto, un recolor usa esa parte y conserva el sujeto restante; no extiende su máscara a toda la persona ni elimina la parte por pertenecer a una persona.

- [ ] «El hombre debe ser calvo» cambia toda la superficie pedida sin pelo residual, franja translúcida ni cabeza añadida.
- [ ] Se conserva la pose y la otra persona en la foto de 800×533 y su versión 1248×832, con varias semillas y fotos adicionales.
- [x] Una corrección de workflow se compara con la misma foto, máscara y semilla antes de integrarse; control 8 pasos sin referencia frente a 8 con referencia (424242), otras dos semillas y manager real 800×533; parámetros en `longcat.appearance_edit` y `longcat.steps`.
- [x] Generate, ediciones compuestas y cambios de vista conservan sus contratos y probes verdes: Generate 72/0 y 31/0; dispatch compuesto y viewpoint exit 0; workflow localizado verifica VAE de foto y referencia `index_timestep_zero`.
- [ ] Compilación Python, revisión APPROVE y verificación PASS; evidencia visual real, validación del usuario y bitácora.

## H-FS-004 — Causa raíz warp/modelo (nueva, aprobada 2026-10-05)

Continuación de tracking autorizada «hazlo»2026-10-06: respetar corte duro solicitado aunque closeup esté activo, limpiar boca/oclusión/fotograma/hold del plano anterior y conservar referencias/fuente y estado de otros vídeos. Reinicio suave explícito de zoom conserva continuidad. Comprobar cortes en frames sin detección antes de mantener composición antigua. Exigir contrato antesFAIL/despuésPASS, probes/lint/compilación, reviewerAPPROVE/testerPASS y secuencia normal real para tracking/bocas; vídeo completo con evidencia y revisión visual para cerrar calidad. No cambiar modelos ni umbrales sin causa reproducida; no certificar tracking con fixtures de estado o clips cortos. Registro `diagnostics/out/tracking_reset_fix_20261006/`.

Verificación de esta corrección acotada2026-10-06: reviewer APPROVE35/0; tester PASS453 reset +27 integración, baseline54FAIL; lint/compilación afectados verdes. Inferencia CUDA completa antes/actual sobre vídeo normal134frames (130swaps por versión),0px de diferencia en clip sin cortes; inspección root de todos los renders. Defecto de estado reproducido/corregido, continuidad corta sin regresión. H-FS-004 permanece abierta: falta vídeo largo/cortes nativos y validación visual del usuario; pérdida de detección26–29 observada y declarada en spec/03.

Continuación de revisión de bocas solicitada2026-10-06 («revisa los logs para mejorarlo»): corregir uso de EMA/histéresis/hold oral en fotos y frames sin smoothing, conservar protección de objetos medida y mostrar evidencias/motivo por foto. Verificar código real y retratos normales; no inferir de ratio0,15/restore0,91 que el objeto sea falso sin occ/rel/ring. No cambiar umbrales ni modelos con ese log incompleto.

Continuación «continua»: corregir inconsistencia geométrica del anillo peri-bucal al ampliar el crop con GFPGAN. Misma huella256→512 cambia ring0,065→0,349 y activa gate0,30 sin evidencia nueva. Escalar radios del anillo con dimensiones reales raw→referencia; tamaños base al config local, umbrales/paste/modelos vigentes conservados. Exigir regresión de escala con controles de labios aislados y oclusión que cruza labios, comparaciones con fotos normales/modelos reales y reviewer/tester independientes. Esto no acredita semántica universal ni el destino504612489.

Diagnóstico posterior de bocas realmente abiertas: MediaPipe medía puntos14→17 (grosor del labio inferior), no abertura13→14. Cuatro fotos normales con inferenciaCPU real confirman ratios erróneos0,10–0,15 frente a abertura0,25–0,61, corroborado por topología oficial. Corregir constantes y campos de labios0/13/14/17, mantener denominador, umbrales, contorno de composición y modelos; verificar abiertas/cerradas y protección real, reviewer/tester y comparación visual sin declarar identidad universal.

Verificación acotada de esta continuación: escala APPROVE/PASS56contratos y206verificaciones reales; apertura APPROVE27controles/PASS147verificaciones, ochoA/BCPU con128/XSeg/GFPGAN y MediaPipe reales. Tres bocas abiertas mejoran visualmente; dos cerradas y pelo03206 conservan0px, positivo parcial conserva paste0,895. Primer control P407 mal etiquetado cerrado y primeras corridas con hash previo distinguidos en bitácora. H-FS-004 permanece abierta por vídeos completos, perfiles/identidad/oclusión difíciles y caso0,91. Evidencia `diagnostics/out/mouth_math_review_20261006/summary.json`.

Verificación oral acotada2026-10-06: reviewerAPPROVE/testerPASS52/0; lint/compilación/probes verdes y ocho casos con128/XSeg/GFPGANCPU reales. Tres controles con igual flag producen0px distintos entre historia limpia/previa; vídeo suavizado conserva memoria. Primera aserción fallida y confusor de colorLAB documentados. Pendiente causalidad del0,91, objetos orales y calidad visual/vídeos; H-FS-004 permanece abierta. Evidencia `diagnostics/out/mouth_review_20261006/summary.json`.

Continuación aprobada el2026-10-06 por «DALE», tras auditoría reviewer CHANGES_REQUESTED: retirar fallback Hyperswap incompleto y restaurar rescate128 de referencias sanas, corregir convención/escala de alineación y conectar configuración, eliminar regeneración elíptica de las máscaras fallidas, y revisar falsos rechazos de fuentes con modelos locales. Corregir lint/compilación identificados y verificar extracción real, regresiones, reviewer y tester. No instalar modelos/dependencias ni sustituir producción128. La aprobación de reparar no cierra calidad visual ni vídeos.

Corrección acotada verificada2026-10-06: extracción real6/7 (una cara realmente cortada), lint/compilación y11probes verdes; reviewerAPPROVE y testerPASS acotados. Composición sana en dos retratos normales con modelosCPU reales, antes/actual idénticos; evidencia `diagnostics/out/source_fix_20261006/summary.json`. H-FS-004 permanece abierta por los criterios visuales siguientes; la referencia explícita no se utiliza en nuevos renders, se requieren muestras normales de perfil/oclusión.

**Como** usuario **quiero** que el perfil `1994859933.jpg` y los vídeos completos mejoren proporción/nitidez/parecido **para** cerrar el defecto que los gates/máscaras no corrigieron.

Diagnóstico que la motiva (MEMORY.md + código): la alineación `norm_crop2` usa solo 5 `kps` (`roop/ProcessMgr.py:4747-4776`); con `eye_rel=0,028` genera warp `1,6×` desplazado y proyecta `512 px` sobre ventana `~1246 px`. El gate `0,05<eye_rel<1,20` + rescate `_swap_from_saved_reference:4662-4712` detecta pero no re-estima `M`; `diagnostics/faceswap_batch_first_rescued.jpg` sigue defectuosa. Los fallbacks recrean elipse (`boost 0,92`, `broad_floor 0,65`, `safety recovery`, `create_soft_mask` en `roop/quality_enhancements.py:275-304`) y el doble `_limit_face_paste` solo contiene (`12 %/16 %` frontal, `6 %/10 %` perfil). `128` estándar `MAE 0` vs `facefusion`; `256` no conserva identidad (`-0,016` vs `0,450`); CodeFormer/GFPGAN y erosiones `11/21 px` no corrigen visualmente.

- [ ] Re-estimar `M` con `bbox+kps` (o `FFHQ/68pts`) o cambiar el modelo `128` por uno con identidad en perfil probada; no cerrar con otro gate/máscara.
- [ ] El fallback de máscara colapsada no puede ser elipse; sin halo/blur elíptico fuera del rostro.
- [ ] El perfil de `1994859933.jpg` mejora visualmente en proporción, nitidez y parecido con la fuente; se conservan oclusiones y personas ajenas.
- [ ] El cambio visual se verifica en fotos adicionales y los vídeos representativos completos; un clip corto, una sola foto o una métrica de píxeles/cambio no cierran el defecto.
- [ ] Probes pertinentes y compilación Python verdes, revisión APPROVE y verificación PASS visual del tester; un probe verde sin PASS visual no cierra la historia.
- [ ] Parámetros nuevos en `config/`, evidencia y límites en `MEMORY.md`, validación visual del usuario.

## H-EDIT-004 — Objeto completo antes de receta (nueva, aprobada 2026-10-05)

**Como** usuario **quiero** retiradas/añadidos con objeto completo y una sola receta `remove/add` **para** no dejar restos ni redibujar la escena.

Diagnóstico que la motiva: `CLIPSeg` mide núcleo (`planta copa ~2 %`, `silla 3,26 %` sin ruedas, `mesa 1,2 %` rechazada por piso `4 %`); `ISNet 1,93 %→28,82 %` toma ambas personas; `visual_forward 62,7 %` escena; `SAM2 CPU 2,1-2,4 s` mejora contorno sin objeto completo; `GrabCut` solo diagnóstico. Receta `8 pasos/ref espacial regional` conserva personas pero deja `tronco/maceta/hojas`, `respaldo hueco`, `anillo claro`; `12/no_ref global` retira pero mueve escena; `ruido ventana completa` redibuja. `Calvo 800×533`: contar caras `2 vs 3` no prueba calvicie; `424243` evita cabeza extra pero deja pelo parcial; `fullsurface` crea cabeza ajena.

- [ ] Cerrar objeto completo (`SAM2+GrabCut` integrado) con `IoU/cobertura` medida y exclusión de otras personas antes de tocar `denoise/steps`; no ampliar solo `paste`.
- [ ] Una sola receta `remove/add` con preservación medida: retira/añade sin restos, sin mover personas/objetos vecinos, sin segunda cabeza ni franja gris.
- [ ] Verificación en varias semillas y fotos adicionales además de la escena de referencia; contratos `CPU/dispatch` en verde no equivalen a calidad visual.
- [ ] Probes pertinentes y compilación Python verdes, revisión APPROVE y verificación PASS visual del tester; evidencia visual real, validación del usuario y bitácora.

## H-GEN-004 — Contratos Generate UI/eventos/APK y cancelación (propuesta 2026-10-20)

**Como** usuario **quiero** que Generate no se rompa por cableado **para** poder generar desde escritorio y APK sin errores de aridad.

Hechos verificados en disco antes de redactar (2026-10-20, sin tocar código): `on_generate_image` declara 21 parámetros (`ui/tabs/gen/events.py:155-167`, AST) y el evento `generate_image_ui` pasa 21 inputs en el mismo orden (`:445-457`); el botón legacy de la APK pasa los 10 primeros (`:460-465`), que coinciden con los 10 que envía `GradioClient.callGenerate` (`GradioClient.kt:482-491`). El `except` del worker escribe `result["error"]` y el generador hace `yield` de 4 valores; no existe `return None, gr.Error`. El preview usa `preview_modifiers(style, l_safe, s_safe, f_safe, c_safe)` con la firma real. Corrección a una revisión anterior con índice obsoleto que afirmaba un `TypeError` de aridad, un `return` de 2 valores y un `NameError` inexistentes: quedan retractados.

- [ ] Probe CPU sin modelos: firma de `on_generate_image` (21 nombres por AST) frente a `inputs` de ambos eventos (21 y 10) en el mismo orden y frente a los 10 valores de `callGenerate`. Sin GPU ni ComfyUI.
- [ ] Todas las ramas del generador `yield` 4 valores para `[output_img,status_html,gen_btn,btn_cancel]`. Ningún `return` con aridad distinta.
- [ ] Cancelación a mitad de muestreo deja botones en idle y no deja job zombie. Estado final, no tiempo real.
- [ ] `py_compile` + Ruff verdes en `ui/tabs/gen/`. Sin regresión en Image Editor (las historias H-GEN/H-EDIT declaran convivencia).
- [ ] Definición de Listo `spec/00`: 1 render real + `MEMORY.md` con evidencia y límites.

## H-FS-006 — Unificar cargador swapper legacy (propuesta 2026-10-20)

**Como** usuario **quiero** un único origen de modelo/providers para FaceSwap **para** que Editor y pipeline den la misma identidad.

Hechos verificados en disco: `roop/swapper.py:35` lee `roop.globals.providers`, que no existe (`roop/globals.py:45` declara `execution_providers`); solo se ejecuta con CUDA ausente, por eso no cae en la RTX del proyecto. `roop/swapper.py:21-24` prefiere `inswapper_256.onnx` si existe, mientras `AGENTS.md` fija producción en `inswapper_128_facefusion.onnx`; `swapper.py` lo usa `roop/img_editor/face_preserver.py:131-132`. H-FS-004 (warp/modelo) no se toca en esta historia.

- [ ] `roop/swapper.py` no referencia `roop.globals.providers`. En CPU usa `execution_providers` o fallback explícito `["CPUExecutionProvider"]`.
- [ ] `get_face_swapper()` y `ProcessMgr` resuelven el mismo `.onnx` de producción declarado en `AGENTS.md`, o documentan por qué difieren.
- [ ] Fallback `INSwapper` pasa `providers` explícitos, no defaults silenciosos.
- [ ] `mouth_region_to_frame_points` no lanza `ValueError` sin capturar ante array impar; devuelve `None`. `Run` sin modelo loguea en vez de devolver `None` silencioso.
- [ ] `py_compile` + Ruff verdes, contratos CPU verdes de H-FS-004/H-FS-005 sin regresión, `MEMORY.md` con límites. Sin tocar umbrales, warp ni modelo.

## H-GEN-005 — Deuda Generate menor (propuesta 2026-10-20)

- [ ] `open_generation_folder` usa `xdg-open`/`open` fuera de Windows y no traga el error en silencio.
- [ ] `set_orientation` no destruye un tamaño personalizado sin aviso, o se documenta como comportamiento.
- [ ] Quitar `get_compatible_dropdown_choices` calculado y no usado, o declarar por qué se conserva.
- [ ] Valorar exponer helpers públicos en `flux_*_client` en vez de `_privados` (`_alias`, `_is_sdxl`, `_get_model_conf()`, `_clip_type`, `_is_longcat`, `_queue_is_idle`, `_sync_unload`). Solo propuesta; el refactor se decide aparte por su riesgo sobre el Editor.
