# AGENTS.md — AutoAuto

## Objetivo
- Mejorar FaceSwap: eliminar el halo o blur elíptico fuera de la cara, seguir perfiles y oclusiones con estabilidad y aumentar nitidez y parecido con la fuente.
- Mantener Image Editor, Generate y Android funcionales al cambiar el procesamiento.

## Entorno y preferencias vigentes
- GPU RTX 3060 Ti de 8 GB, CUDA 12.4; proveedores CUDAExecutionProvider y CPUExecutionProvider.
- FaceSwap: inswapper_128_facefusion.onnx (128→256 px), XSeg Masker, GFPGAN Enhancer (blend 0,70), MediaPipe Face Mesh de 468 puntos y salida FFmpeg libx264 CRF 14.
- Caso representativo: selected_faces, hasta 655 facesets origen, un destino, vídeo de 1280×720, 850 fotogramas y 30 fps. El banco y el número de caras pueden variar según el lote.
- En Image Editor, el usuario prioriza cumplir la instrucción completa con rapidez, incluso si una edición compuesta cambia las caras. Esa preferencia no reduce la exigencia de identidad en FaceSwap.

## Criterios de trabajo
- Image Editor no se enruta con listas de palabras, géneros, prendas ni tipos de edición escritos a mano. La decisión sale de los conceptos ya medidos y de la región que el segmentador encuentre en la foto. Una instrucción que no pide una postura nueva no sustituye el resto de la imagen.
- Consultar MEMORY.md antes de tocar una ruta existente: contiene decisiones actuales, pruebas y trabajo pendiente. Para detalles históricos, usar docs/project_history/PROJECT_LOG_2026-09-29.md.
- Contrastar las notas con el código y los logs actuales; las entradas antiguas del historial incluyen experimentos revertidos y afirmaciones ya superadas.
- Distinguir pruebas de despacho o métricas de píxel de una validación visual real. No declarar resuelto un defecto con clips cortos o una sola foto.
- Al cambiar un flujo, ejecutar los probes pertinentes y comprobar visualmente muestras reales cuando la calidad de imagen sea el objetivo.
- No interrumpir renders ni reiniciar la UI activa sin comprobar primero su estado.

## Archivos principales
- FaceSwap: roop/ProcessMgr.py, roop/quality_enhancements.py, roop/face_util.py y ui/tabs/faceswap/.
- Image Editor: roop/img_editor/img_editor_manager.py, roop/img_editor/top_tier_finish.py y clientes ComfyUI en roop/img_editor/.
- Pruebas diagnósticas: diagnostics/; Android: android_app/autoapp/.
