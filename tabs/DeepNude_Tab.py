import subprocess

import cv2
import gradio as gr
from pathlib import Path
import shutil
import roop.utilities as util
import roop.globals

resultfiles = gr.Files()
resultimage = gr.Image()


def DeepNude_tab() -> None:
    global bt_srcfiles
    global resultfiles
    global resultimage
    with gr.Tab("🔞 DeepNude Basico"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### DeepNude Basico")
                bt_srcfiles = gr.Files(label='Archivos Origen', file_count="multiple", file_types=["image", ".fsz"],
                                       elem_id='filelist', height=300)
        with gr.Row(variant='panel'):
            with gr.Column():
                bt_start = gr.Button("▶ Start", variant='primary')
                gr.Button("👀 Open Output Folder", size='sm').click(
                    fn=lambda: util.open_folder(roop.globals.output_path))
            with gr.Column(scale=2):
                gr.Markdown(' ')

        with gr.Row(variant='panel'):
            with gr.Column():
                resultfiles = gr.Files(label='Archivo(s) procesado(s)', interactive=False)
            with gr.Column():
                resultimage = gr.Image(type='filepath', label='Imagen Final', interactive=False)

    start_event = bt_start.click(fn=nudity, inputs=bt_srcfiles, outputs=[bt_start, resultfiles], show_progress='full')
    after_swap_event = start_event.then(fn=on_resultfiles_finished, inputs=[resultfiles],
                                        outputs=[resultfiles, resultimage])
    resultfiles.select(fn=on_resultfiles_selected, inputs=[resultfiles], outputs=[resultimage])


def update_result_files(processed_files):
    global resultfiles
    resultfiles.value = processed_files


def on_resultfiles_selected(evt: gr.SelectData, files):
    selected_index = evt.index
    filename = files[selected_index].name
    return display_output(filename)


def on_resultfiles_finished(files):
    if files is None or len(files) < 1:
        return None, None
    filename = files[0].name
    return display_output(filename)


def display_output(filename):
    # Asegúrate de que filename sea una ruta de archivo válida como string
    if not isinstance(filename, str):
        raise ValueError("filename debe ser una cadena de texto que represente una ruta de archivo")

    from roop.capturer import get_image_frame
    current_frame = get_image_frame(filename)
    image_for_display = util.convert_to_gradio(current_frame)

    # Define la ruta de salida donde se guardará la imagen
    output_image_path = Path(roop.globals.output_path) / "output_image.png"

    # Guarda el ndarray como una imagen en la ruta especificada
    cv2.imwrite(str(output_image_path), image_for_display)

    # Retorna la ruta de la imagen guardada como string
    return str(output_image_path), None


def nudity(selected_files: gr.Files):
    global resultfiles
    global resultimage
    destination_dir = Path(roop.globals.output_path, "input_files")
    destination_dir.mkdir(parents=True, exist_ok=True)

    # Ensure selected_files is a list
    if not isinstance(selected_files, list):
        selected_files = [selected_files]

    processed_files_paths = []  # List to collect processed file paths

    for file_path in selected_files:
        file_name = Path(file_path).name
        destination_path = destination_dir / file_name
        shutil.copy(file_path, destination_path)
        output_file_path = str(destination_dir / f"output_{file_name}")
        call_main_py_with_correct_args(str(destination_path), output_file_path)
        processed_files_paths.append(output_file_path)  # Collect the output file path

    update_result_files(processed_files_paths)  # Update the result files with the list of processed file paths

    if processed_files_paths:
        first_processed_file = processed_files_paths[0]
        image_for_display, _ = display_output(first_processed_file)
        rr = Path(first_processed_file)
        resultimage.value = rr  # Ensure this is a valid file path
        return gr.Button(variant="primary"), processed_files_paths
    else:
        return gr.Button(variant="primary", disabled=True), []


import torch


def get_gpu_ids_and_names():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_info = []
        for gpu_id in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(gpu_id)
            gpu_info.append((gpu_id, gpu_name))
        return gpu_info
    else:
        return "CUDA is not available."


def call_main_py_with_correct_args(input_image_path, output_image_path):
    input_path = Path(input_image_path)
    gpu_ids_and_names = get_gpu_ids_and_names()
    print(gpu_ids_and_names)
    if not input_path.is_file():
        print(f"Error: El archivo de entrada {input_image_path} no existe.")
        return

    main_py_path = "D:/PROJECTS/AUTOPORN/DEP/main.py"
    command = [
        "python", main_py_path, "run", "-i", input_image_path, "-o", output_image_path,
        "--auto-resize-crop", "--experimental-artifacts-inpaint",
        "--n-cores", "12", "--gpu", "0"
    ]
    subprocess.run(command)
