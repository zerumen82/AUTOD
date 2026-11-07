# Corrección del control cromático del acabado — 2026-10-26

Autorización: «hazlo con algo no explícito». Alcance: H-QUALITY-COLOR-GUARD,
acabado opcional color/luz en Solo mejorar y después de editar. Aplicado en
código y configuración; sin reiniciar UI, instalar dependencias o modelos,
cambiar motores, parámetros de generación o geometría.

## Cambio y efecto

El control anterior solo verificaba clipping y nitidez en el paso color/luz;
aceptaba una desviación de tono moderada. Ahora compara el candidato con el
resultado ya editado inmediatamente anterior al acabado. Si la deriva supera
los límites, o la medida/configuración/geometría son inválidas, conserva ese
resultado y registra el motivo. No lo compara con la foto original: así no
deshace un recolor solicitado. acceptance_gate=false omite este control.

La medida es distancia de Lab a,b con RGB float normalizado, excluyendo L.
Esta representación sigue los rangos de la [documentación oficial de
OpenCV](https://docs.opencv.org/4.11.0/de/d25/imgproc_color_conversions.html).
No es DeltaE76 completo ni una garantía universal de color percibido.

Parámetros en config/quality_mode.json y los defaults del pipeline:
chroma_mean_tolerance=2, chroma_p95_tolerance=6, chroma_max_side=1024.
Calibración acotada: los acabados normales de las muestras tienen media/p95
0,42/1,14 (tetera), 0,14/0,70 (paisaje), 0,40/1,12 (HD Solo mejorar),
0,24/0,73 (HD después de editar). El desvío moderado del test mide 6,19/6,34;
el localizado en 10 % mide 1,30/13,03 y se detecta por p95. Los límites son
de seguridad del acabado, no estimadores de calidad estética universal.

Archivos de producción: roop/img_editor/top_tier_finish.py (helper y dos
checks), image_quality_pipeline.py (defaults), config/quality_mode.json.
Snapshots previos de este turno:
diagnostics/out/quality_color_fix_20261026_1793041552794/.
No se aprueban las otras modificaciones preexistentes del workspace.

## Evidencia

1. **Rojo real antes del cambio:** probe nuevo 4/10 PASS, 6 fallos,
   logs/quality_color_guard_before_20261026.log y
   diagnostics/out/quality_color_guard_20261026/before_1793041598098260900/.
   Se medían aceptaciones incorrectas de canales alterados, tono moderado,
   daño localizado, geometría/configuración inválidas y ambas rutas del acabado.
2. **Verde después:** 10/10 iniciales; tras ampliar los casos a error/NaN,
   tester independiente obtiene **11/11 PASS por imagen**, sin inferencia IA.
   Fallos de procesamiento inyectados SOLO en pruebas, nunca producción.
3. **Acabado real no explícito:** tetera roja/taza azul y paisaje pasan ambas
   funciones reales sin parches. Entrada/salida, métricas y un caso de fallo
   controlado etiquetado en cada directorio:
   - diagnostics/out/quality_color_guard_20261026/tester_teapot_1793041766024180900/
   - diagnostics/out/quality_color_guard_20261026/tester_landscape_1793041765790009400/
   PNG: source.png, quality.png, after_edit.png. injected_cast.png y
   guarded_cast.png son demostraciones de fallo controlado, no una salida
   espontánea de un modelo. El control devuelve exactamente la fuente en
   ese caso (distancia cromática final 0). Inspección visual de las dos salidas
   normales por root/tester: colores y encuadre conservados.
4. **Cobertura adicional:** 3/3 (imagen 2304 px reducida para el control,
   luminosidad permitida y pasos apagados conservando recolor). HD real
   1920x1080 COLOR_ONLY: diferencia RGB media 4,4990 en Solo mejorar y 1,5916
   tras editar, sin revertir el paso completo ni cambiar dimensiones.
5. **Regresiones:** reference_canvas 7/7, paste_guard PASS,
   recolor_surface_zone 0 fallos, Generate 31/31 (contrato CPU/red simulada,
   no generación IA). Logs quality_color_guard_tester_*_20261026.log.
6. **Lint/compilación:** Ruff completo y py_compile de los dos módulos de
   producción y dos probes afectados, verdes. Diff-check respetando CRLF verde.
7. **Build:** python -m PyInstaller --noconfirm --workpath
   diagnostics/out/quality_color_fix_20261026_1793041552794/build --distpath
   diagnostics/out/quality_color_fix_20261026_1793041552794/dist AutoAuto.spec
   -> exit 0; logs/quality_color_fix_desktop_build_20261026.log. Es el build
   del launcher del proyecto; no acredita una APK ni empaqueta por sí solo
   todos los modelos/módulos cargados después por run.py.
8. **Revisión:** color_fix_review: APPROVE a implementación/configuración;
   color_fix_tester: PASS al alcance CPU, con PASS adicional HD. Ninguno
   aprueba todo el workspace o sustituye valoración visual del usuario.

## Límites pendientes

Pendiente valoración visual del usuario sobre estas salidas. No hubo un nuevo
render IA: se ejecutó el acabado clásico real sobre imágenes existentes.
Media/p95 con resolución acotada pueden omitir daños diminutos. El registro
original tenía calidad=False: este arreglo no demuestra ni corrige por sí
solo una desviación que ya salga del motor en ese modo. La posición no forma
parte de este cambio y conserva su deuda anterior.
