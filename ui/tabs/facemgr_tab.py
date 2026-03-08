import os
import shutil
import threading
import time
from typing import List, Tuple, Optional

import cv2
import numpy as np
import gradio as gr
import roop.utilities as util
import roop.globals

from roop.types import Frame

selected_face_index = -1
thumbs = []
images = []

# Variables para control de timeouts
_processing_timeout = 30  # 30 segundos máximo por archivo
_processing_lock = threading.Lock()


def facemgr_tab() -> None:
    with gr.Tab("👨‍👩‍👧‍👦 Gestión facial"):
        with gr.Row():
            gr.Markdown("""
                        # Crear conjuntos de caras de fusión
                        Agregue varias imágenes de referencia a un  faceset.
                        """)
        with gr.Row():
            videoimagefst = gr.Image(label="Cortar cara del fotograma de vídeo", height=576, interactive=False, visible=True)
        with gr.Row():
            frame_num_fst = gr.Slider(1, 1, value=1, label="Número de frame", info='0:00:00', step=1.0, interactive=False)
            fb_cutfromframe = gr.Button("Use faces from this frame", variant='secondary', interactive=False)
        with gr.Row():
            fb_facesetfile = gr.Files(label='Faceset', file_count='single', file_types=['.fsz'], interactive=True)
            fb_files = gr.Files(label='Archivos a Insertar', file_count="multiple", file_types=["image", "video"], interactive=True)
        with gr.Row():
            with gr.Column():
                gr.Button("👀 Abrir carpeta de salida", size='sm').click(fn=lambda: util.open_folder(roop.globals.output_path))
            with gr.Column():
                gr.Markdown(' ')
        with gr.Row():
            faces = gr.Gallery(label="Caras en este Faceset", allow_preview=True, preview=True, height=128, object_fit="scale-down")
        with gr.Row():
            fb_remove = gr.Button("Eliminar Seleccionadas", variant='secondary')
            fb_update = gr.Button("Crea o Actualiza el Faceset file", variant='primary')
            fb_clear = gr.Button("Eliminar todo", variant='stop')

    fb_facesetfile.change(fn=on_faceset_changed, inputs=[fb_facesetfile], outputs=[faces])
    fb_files.change(fn=on_fb_files_changed, inputs=[fb_files], outputs=[faces, videoimagefst, frame_num_fst, fb_cutfromframe])
    fb_update.click(fn=on_update_clicked, outputs=[fb_facesetfile])
    fb_remove.click(fn=on_remove_clicked, outputs=[faces])
    fb_clear.click(fn=on_clear_clicked, outputs=[faces, fb_files, fb_facesetfile])
    fb_cutfromframe.click(fn=on_cutfromframe_clicked, inputs=[fb_files, frame_num_fst], outputs=[faces])
    frame_num_fst.release(fn=on_frame_num_fst_changed, inputs=[fb_files, frame_num_fst], outputs=[videoimagefst])
    faces.select(fn=on_face_selected)


def on_faceset_changed(faceset, progress=gr.Progress()) -> List[Frame]:
    global thumbs, images

    if faceset is None:
        return thumbs

    thumbs.clear()
    filename = faceset.name
    from roop.face_util import extract_face_images
    if filename.lower().endswith('fsz'):
        progress(0, desc="Recuperar caras de Faceset File", )
        unzipfolder = os.path.join(os.environ["TEMP"], 'faceset')
        if os.path.isdir(unzipfolder):
            try:
                shutil.rmtree(unzipfolder, ignore_errors=True)
            except OSError as e:
                gr.Warning(f'Error al limpiar carpeta temporal: {e}')
        util.mkdir_with_umask(unzipfolder)
        util.unzip(filename, unzipfolder)
        for file in os.listdir(unzipfolder):
            if file.endswith(".png"):
                SELECTION_FACES_DATA = extract_face_images(os.path.join(unzipfolder,file),  (False, 0), 0.1)
                if len(SELECTION_FACES_DATA) < 1:
                    gr.Warning(f"No se detectó ningún rostro en {file}!")
                for f in SELECTION_FACES_DATA:
                    image = f[1]
                    
                    # CORRECCIÓN: Preservar colores en la extracción de caras
                    if isinstance(image, np.ndarray):
                        if len(image.shape) == 3 and image.shape[2] == 3:
                            # Si está en formato BGR (OpenCV), convertir a RGB
                            if image.dtype == np.uint8:
                                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                print(f"[DEBUG] Cara extraída convertida de BGR->RGB: {image_rgb.shape}")
                            else:
                                image_rgb = image
                        else:
                            image_rgb = image
                    else:
                        image_rgb = image
                        
                    images.append(image_rgb)
                    thumbs.append(util.convert_to_gradio(image_rgb))
        
        return thumbs


def on_fb_files_changed(inputfiles, progress=gr.Progress()) -> Tuple[List[Frame], Optional[gr.Image], Optional[gr.Slider], Optional[gr.Button]]:
   global thumbs, images

   print("=== INICIO on_fb_files_changed ===")
   print(f"[DEBUG] on_fb_files_changed llamado con {len(inputfiles) if inputfiles else 0} archivos")

   if inputfiles is None or len(inputfiles) < 1:
       print("[DEBUG] No hay archivos para procesar")
       return thumbs, None, None, None

   from roop.face_util import extract_face_images
   from roop.capturer import get_video_frame_total

   progress(0, desc="Procesando archivos...", )
   print(f"[DEBUG] Procesando {len(inputfiles)} archivos")

   slider = None
   video_image = None
   cut_button = None

   # Procesar archivos de manera más eficiente - usar procesamiento por lotes para grandes volúmenes
   image_files = [f.name for f in inputfiles if util.has_image_extension(f.name)]

   if len(image_files) > 5:  # Si hay más de 5 imágenes, usar procesamiento por lotes
       print(f"[INFO] Procesando {len(image_files)} imágenes en lotes para mayor eficiencia")

       from roop.face_util import extract_face_images_batch, MAX_WORKERS

       print(f"[INFO] Iniciando procesamiento por lotes con {MAX_WORKERS} hilos para {len(image_files)} imágenes")

       # Procesar en lotes con máximo rendimiento
       batch_results = extract_face_images_batch(
           image_files,
           options_list=[(False, 0)] * len(image_files),
           face_detection_padding=0.1,
           target_face_detection=False,
           max_workers=MAX_WORKERS  # Usar máximo de hilos disponibles
       )

       print(f"[INFO] Procesamiento por lotes completado. Resultados obtenidos: {len(batch_results)}")

       # Procesar resultados
       for i, result in enumerate(batch_results):
           progress((i / len(batch_results)) * 0.8, desc=f"Procesando resultados lote {i+1}...")
           if result and len(result) > 0:
               for face_data in result:
                   image = face_data[1]
                   images.append(image)
                   thumbs.append(util.convert_to_gradio(image))

       # Configurar UI para imágenes
       slider = gr.Slider(interactive=False)
       video_image = gr.Image(interactive=False)
       cut_button = gr.Button(interactive=False)

   else:
       # Procesamiento individual para pocos archivos
       for i, f in enumerate(inputfiles):
           source_path = f.name
           print(f"[DEBUG] Procesando archivo {i+1}/{len(inputfiles)}: {source_path}")

           if util.has_image_extension(source_path):
               slider = gr.Slider(interactive=False)
               video_image = gr.Image(interactive=False)
               cut_button = gr.Button(interactive=False)
               roop.globals.source_path = source_path

               try:
                   progress((i / len(inputfiles)) * 0.8, desc=f"Detectando rostros en imagen {i+1}...")
                   print(f"[DEBUG] Llamando a extract_face_images para {source_path}")
                   SELECTION_FACES_DATA = extract_face_images(roop.globals.source_path, (False, 0), 0.1)
                   print(f"[DEBUG] extract_face_images retornó: {len(SELECTION_FACES_DATA) if SELECTION_FACES_DATA else 0} rostros")

                   if SELECTION_FACES_DATA and len(SELECTION_FACES_DATA) > 0:
                       print(f"[DEBUG] Procesando {len(SELECTION_FACES_DATA)} rostros detectados")
                       for face_data in SELECTION_FACES_DATA:
                           image = face_data[1]
                           images.append(image)
                           thumbs.append(util.convert_to_gradio(image))
                           print(f"[DEBUG] Rostro agregado - total thumbs: {len(thumbs)}")
                   else:
                       print(f"[WARNING] No se detectaron rostros en {source_path}")

               except Exception as e:
                   print(f"[ERROR] Error procesando imagen {source_path}: {str(e)}")
                   import traceback
                   traceback.print_exc()
                   gr.Warning(f"Error procesando {os.path.basename(source_path)}: {str(e)}")

   # Procesar videos por separado (no se incluyen en el procesamiento por lotes)
   for i, f in enumerate(inputfiles):
       source_path = f.name
       if util.is_video(source_path) or source_path.lower().endswith('gif'):
           try:
               progress((i / len(inputfiles)) * 0.8, desc=f"Procesando video {i+1}...")
               total_frames = get_video_frame_total(source_path)
               current_video_fps = util.detect_fps(source_path)
               cut_button = gr.Button(interactive=True)
               video_image, slider = display_video_frame(source_path, 1, total_frames)
               print(f"[DEBUG] Video procesado: {total_frames} frames, {current_video_fps} fps")

           except Exception as e:
               print(f"[ERROR] Error procesando video {source_path}: {str(e)}")
               gr.Warning(f"Error procesando video {os.path.basename(source_path)}: {str(e)}")

   progress(1.0, desc="Procesamiento completado")
   print(f"[DEBUG] Retornando {len(thumbs)} thumbs")
   print("=== FIN on_fb_files_changed ===")
   return thumbs, video_image, slider, cut_button
    

def display_video_frame(filename: str, frame_num: int, total: int=0) -> Tuple[gr.Image, gr.Slider]:
    global current_video_fps

    from roop.capturer import get_video_frame
    current_frame = get_video_frame(filename, frame_num)
    if current_video_fps == 0:
        current_video_fps = 1
    secs = (frame_num - 1) / current_video_fps
    minutes = secs / 60
    secs = secs % 60
    hours = minutes / 60
    minutes = minutes % 60
    milliseconds = (secs - int(secs)) * 1000
    timeinfo = f"{int(hours):0>2}:{int(minutes):0>2}:{int(secs):0>2}.{int(milliseconds):0>3}"
    if total > 0:
        return gr.Image(value=util.convert_to_gradio(current_frame), interactive=True), gr.Slider(info=timeinfo, minimum=1, maximum=total, interactive=True)  
    return gr.Image(value=util.convert_to_gradio(current_frame), interactive=True), gr.Slider(info=timeinfo, interactive=True)  


def on_face_selected(evt: gr.SelectData) -> None:
    global selected_face_index

    if evt is not None:
        selected_face_index = evt.index

def on_frame_num_fst_changed(inputfiles: List[gr.Files], frame_num: int) -> Frame:
    filename = inputfiles[0].name
    video_image, _ = display_video_frame(filename, frame_num, 0)
    return video_image


def on_cutfromframe_clicked(inputfiles: List[gr.Files], frame_num: int) -> List[Frame]:
    global thumbs
    from roop.face_util import extract_face_images
    import cv2
    import numpy as np
    
    filename = inputfiles[0].name
    SELECTION_FACES_DATA = extract_face_images(filename,  (True, frame_num), 0.1)
    for f in SELECTION_FACES_DATA:
        image = f[1]
        
        # CORRECCIÓN: Asegurar que la imagen extraída mantenga colores correctos
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Si está en formato BGR (OpenCV), convertir a RGB para preservar colores
                if image.dtype == np.uint8:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    print(f"[DEBUG] Cara extraída convertida de BGR->RGB: {image_rgb.shape}")
                else:
                    image_rgb = image
            else:
                image_rgb = image
        else:
            image_rgb = image
            
        images.append(image_rgb)
        thumbs.append(util.convert_to_gradio(image_rgb))
        
    return thumbs


def on_remove_clicked() -> List[Frame]:
    global thumbs, images, selected_face_index

    if len(thumbs) > selected_face_index:
        f = thumbs.pop(selected_face_index)
        del f
        f = images.pop(selected_face_index)
        del f
    return thumbs

def on_clear_clicked() -> Tuple[List[Frame], None, None]:
    global thumbs, images

    thumbs.clear()
    images.clear()
    return thumbs, None, None


def on_update_clicked() -> Optional[str]:
    if len(images) < 1:
        gr.Warning(f"¡No hay caras para crear faceset!")
        return None

    imgnames = []
    for index,img in enumerate(images):
        filename = os.path.join(roop.globals.output_path, f'{index}.png')
        cv2.imwrite(filename, img)
        imgnames.append(filename)

    finalzip = os.path.join(roop.globals.output_path, 'faceset.fsz')        
    util.zip(imgnames, finalzip)
    return finalzip
