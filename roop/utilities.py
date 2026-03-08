import glob
import mimetypes
import os
import platform
import shutil
import ssl
import subprocess
import sys
import urllib

import tempfile
import cv2
import numpy as np
import zipfile
import traceback

from pathlib import Path
from typing import List, Any

from scipy.spatial import distance
from tqdm import tqdm

import roop.template_parser as template_parser

import roop.globals

TEMP_FILE = "temp.mp4"
TEMP_DIRECTORY = "temp"

# monkey patch ssl for mac
if platform.system().lower() == "darwin":
    ssl._create_default_https_context = ssl._create_unverified_context()


# https://github.com/facefusion/facefusion/blob/master/facefusion
def detect_fps(target_path: str) -> float:
    fps = 24.0
    cap = cv2.VideoCapture(target_path)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


# Gradio wants Images in RGB
def convert_to_gradio(image, is_rgb=False):
    """
    Convert an image to a format compatible with Gradio.
    Handles numpy arrays and PIL Images, ensuring proper color conversion.
    Ensures no circular references for FastAPI serialization.

    Args:
        image: Input image (numpy array or PIL Image)
        is_rgb: If True, assumes the numpy array is already in RGB format (default: False, assumes BGR)

    Returns:
        RGB image ready for Gradio display (simple numpy array, no circular refs)
    """
    import numpy as np
    import cv2
    from PIL import Image

    if image is None:
        return None

    try:
        if isinstance(image, np.ndarray):
            # Manejar diferentes formatos de numpy arrays
            if len(image.shape) == 2:
                # Imagen en escala de grises -> convertir a RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3:
                if image.shape[2] == 4:
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                elif image.shape[2] == 3:
                    # Si ya está en RGB, devolver tal cual
                    if is_rgb:
                        rgb_image = image.copy()  # Crear copia para evitar referencias
                    else:
                        # Si está en BGR (formato OpenCV estándar), convertir a RGB
                        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    # Formato desconocido, asumir BGR
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                return None

            # Asegurar que es un array numpy simple sin referencias circulares
            if isinstance(rgb_image, np.ndarray):
                return rgb_image.copy()  # Retornar copia para evitar referencias
            return rgb_image

        elif isinstance(image, Image.Image):
            # Convertir PIL Image a numpy array RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # Convertir a numpy array y retornar copia
            rgb_array = np.array(image)
            return rgb_array.copy() if isinstance(rgb_array, np.ndarray) else rgb_array

        # Para cualquier otro tipo, intentar convertir a numpy array
        try:
            arr = np.array(image)
            if isinstance(arr, np.ndarray) and len(arr.shape) >= 2:
                return arr.copy()
        except:
            pass

    except Exception as e:
        print(f"[WARNING] Error converting image to Gradio format: {e}")
        return None

    return None


def create_error_image(message):
    """Crea una imagen de error con un mensaje"""
    height = 100
    width = 400
    error_img = np.zeros((height, width, 3), dtype=np.uint8)
    error_img[:, :, 0] = 50  # Fondo azul oscuro
    error_img[:, :, 2] = 50  # Fondo azul oscuro
    
    # Añadir texto del error
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    color = (255, 255, 255)  # Texto blanco
    thickness = 1
    
    # Dividir el mensaje en líneas
    words = message.split()
    lines = []
    current_line = []
    for word in words:
        test_line = ' '.join(current_line + [word])
        (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        if text_width < width - 20:  # Dejar margen
            current_line.append(word)
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
    if current_line:
        lines.append(' '.join(current_line))
    
    # Dibujar cada línea
    y = 30
    for i, line in enumerate(lines):
        if i >= 3:  # Máximo 3 líneas
            break
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        x = (width - text_size[0]) // 2
        cv2.putText(error_img, line, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
        y += 20
    
    return error_img


def sort_filenames_ignore_path(filenames):
    """Sorts a list of filenames containing a complete path by their filename,
    while retaining their original path.

    Args:
      filenames: A list of filenames containing a complete path.

    Returns:
      A sorted list of filenames containing a complete path.
    """
    filename_path_tuples = [
        (os.path.split(filename)[1], filename) for filename in filenames
    ]
    sorted_filename_path_tuples = sorted(filename_path_tuples, key=lambda x: x[0])
    return [
        filename_path_tuple[1] for filename_path_tuple in sorted_filename_path_tuples
    ]


def sort_rename_frames(path: str):
    filenames = os.listdir(path)
    filenames.sort()
    for i in range(len(filenames)):
        of = os.path.join(path, filenames[i])
        newidx = i + 1
        new_filename = os.path.join(
            path, f"{newidx:06d}." + roop.globals.CFG.output_image_format
        )
        os.rename(of, new_filename)


def get_temp_frame_paths(target_path: str) -> List[str]:
    temp_directory_path = get_temp_directory_path(target_path)
    return glob.glob(
        (
            os.path.join(
                glob.escape(temp_directory_path),
                f"*.{roop.globals.CFG.output_image_format}",
            )
        )
    )


def get_temp_directory_path(target_path: str) -> str:
    target_name, _ = os.path.splitext(os.path.basename(target_path))
    target_directory_path = os.path.dirname(target_path)
    return os.path.join(target_directory_path, TEMP_DIRECTORY, target_name)


def get_temp_output_path(target_path: str) -> str:
    temp_directory_path = get_temp_directory_path(target_path)
    return os.path.join(temp_directory_path, TEMP_FILE)


def normalize_output_path(source_path: str, target_path: str, output_path: str) -> Any:
    if source_path and target_path:
        source_name, _ = os.path.splitext(os.path.basename(source_path))
        target_name, target_extension = os.path.splitext(os.path.basename(target_path))
        if os.path.isdir(output_path):
            return os.path.join(
                output_path, source_name + "-" + target_name + target_extension
            )
    return output_path


def get_destfilename_from_path(
    srcfilepath: str, destfilepath: str, extension: str
) -> str:
    fn, ext = os.path.splitext(os.path.basename(srcfilepath))
    if "." in extension:
        return os.path.join(destfilepath, f"{fn}{extension}")
    return os.path.join(destfilepath, f"{fn}{extension}{ext}")


def replace_template(file_path: str, index: int = 0) -> str:
    fn, ext = os.path.splitext(os.path.basename(file_path))

    # Remove the "__temp" placeholder that was used as a temporary filename
    fn = fn.replace("__temp", "")

    template = roop.globals.CFG.output_template
    replaced_filename = template_parser.parse(
        template, {"index": str(index), "file": fn}
    )

    return os.path.join(roop.globals.output_path, f"{replaced_filename}{ext}")


def create_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    Path(temp_directory_path).mkdir(parents=True, exist_ok=True)


def move_temp(target_path: str, output_path: str) -> None:
    temp_output_path = get_temp_output_path(target_path)
    if os.path.isfile(temp_output_path):
        if os.path.isfile(output_path):
            os.remove(output_path)
        shutil.move(temp_output_path, output_path)


def clean_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    parent_directory_path = os.path.dirname(temp_directory_path)
    if not roop.globals.keep_frames and os.path.isdir(temp_directory_path):
        shutil.rmtree(temp_directory_path)
    if os.path.exists(parent_directory_path) and not os.listdir(parent_directory_path):
        os.rmdir(parent_directory_path)


def delete_temp_frames(filename: str) -> None:
    dir = os.path.dirname(os.path.dirname(filename))
    shutil.rmtree(dir)


def has_image_extension(image_path: str) -> bool:
    return image_path.lower().endswith(("png", "jpg", "jpeg", "webp"))


def has_extension(filepath: str, extensions: List[str]) -> bool:
    return filepath.lower().endswith(tuple(extensions))


def is_image(image_path: str) -> bool:
    if image_path and os.path.isfile(image_path):
        mimetype, _ = mimetypes.guess_type(image_path)
        return bool(mimetype and mimetype.startswith("image/"))
    return False


def is_video(video_path: str) -> bool:
    if video_path and os.path.isfile(video_path):
        mimetype, _ = mimetypes.guess_type(video_path)
        return bool(mimetype and mimetype.startswith("video/"))
    return False


def conditional_download(download_directory_path: str, urls: List[str]) -> None:
    if not os.path.exists(download_directory_path):
        os.makedirs(download_directory_path)
    for url in urls:
        download_file_path = os.path.join(
            download_directory_path, os.path.basename(url)
        )
        if not os.path.exists(download_file_path):
            request = urllib.request.open(url)  # type: ignore[attr-defined]
            total = int(request.headers.get("Content-Length", 0))
            with tqdm(
                total=total,
                desc=f"Downloading {url}",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress:
                urllib.request.urlretrieve(download_file_path, reporthook=lambda count, block_size, total_size: progress.update(block_size))  # type: ignore[attr-defined]


def get_local_files_from_folder(folder: str) -> List[str]:
    if not os.path.exists(folder) or not os.path.isdir(folder):
        return None
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
    ]
    return files


def resolve_relative_path(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))


def get_device() -> str:
    if len(roop.globals.execution_providers) < 1:
        roop.globals.execution_providers = ["CUDAExecutionProvider"]

    # Prioritize CUDAExecutionProvider
    if "CUDAExecutionProvider" in roop.globals.execution_providers:
        return "cuda"
    if "CoreMLExecutionProvider" in roop.globals.execution_providers:
        return "mps"
    if "ROCMExecutionProvider" in roop.globals.execution_providers:
        return "cuda"
    if "OpenVINOExecutionProvider" in roop.globals.execution_providers:
        return "mkl"
    return "cpu"


def str_to_class(module_name, class_name) -> Any:
    from importlib import import_module

    class_ = None
    try:
        print(f"[DEBUG] Attempting to import module: {module_name}")
        module_ = import_module(module_name)
        print(f"[DEBUG] Module imported successfully: {module_}")
        
        # Intentar obtener la clase directamente primero
        if hasattr(module_, class_name):
            print(f"[DEBUG] Found class {class_name} in module")
            class_type = getattr(module_, class_name)
            class_ = class_type()
        else:
            print(f"[DEBUG] Class {class_name} not found directly, looking for alternatives...")
            # Buscar clases que coincidan con patrones
            for attr_name in dir(module_):
                if 'FaceSwap' in attr_name and not attr_name.startswith('_'):
                    try:
                        class_type = getattr(module_, attr_name)
                        if callable(class_type):
                            print(f"[DEBUG] Found alternative class: {attr_name}")
                            class_ = class_type()
                            break
                    except Exception as e:
                        print(f"[DEBUG] Error trying to instantiate {attr_name}: {e}")
                        continue
            
            if class_ is None:
                print(f"Class {class_name} does not exist in module {module_name}")
        
    except ImportError as e:
        print(f"Module {module_name} does not exist: {e}")
        # Fallback: intentar importar solo el módulo base
        try:
            base_module = "roop.processors"
            print(f"[DEBUG] Trying base module: {base_module}")
            base = import_module(base_module)
            # Listar procesadores disponibles
            available = [x for x in dir(base) if 'FaceSwap' in x and not x.startswith('_')]
            print(f"[DEBUG] Available FaceSwap processors: {available}")
        except Exception as base_error:
            print(f"[DEBUG] Base module import failed: {base_error}")
    
    return class_

def is_installed(name:str) -> bool:
    return shutil.which(name);

# Taken from https://stackoverflow.com/a/68842705
def get_platform() -> str:
    if sys.platform == "linux":
        try:
            proc_version = open("/proc/version").read()
            if "Microsoft" in proc_version:
                return "wsl"
        except:
            pass
    return sys.platform

def open_with_default_app(filename:str):
    if filename == None:
        return
    platform = get_platform()
    if platform == "darwin":
        subprocess.call(("open", filename))
    elif platform in ["win64", "win32"]:        
        os.startfile(filename.replace("/", "\\"))
    elif platform == "wsl":
        subprocess.call("cmd.exe /C start".split() + [filename])
    else:  # linux variants
        subprocess.call(("xdg-open", filename))


def prepare_for_batch(target_files) -> str:
    print("Preparing temp files")
    tempfolder = os.path.join(tempfile.gettempdir(), "rooptmp")
    if os.exists(tempfolder):
        shutil.rmtree(tempfolder)
    Path(tempfolder).mkdir(parents=True, exist_ok=True)
    for f in target_files:
        newname = os.path.basename(f.name)
        shutil.move(f.name, os.path.join(tempfolder, newname))
    return tempfolder


def zip(files, zipname):
    with zipfile.ZipFile(zipname, "w") as zip_file:
        for f in files:
            zip_file.write(f, os.path.basename(f))


def unzip(zipfilename: str, target_path: str):
    with zipfile.ZipFile(zipfilename, "r") as zip_file:
        zip_file.extractall(target_path)


def mkdir_with_umask(directory):
    oldmask = os.umask(0)
    # mode needs octal
    os.makedirs(directory, mode=0o775, exist_ok=True)
    os.umask(oldmask)


def open_folder(path: str):
    platform = get_platform()
    try:
        if platform == "darwin":
            subprocess.call(("open", path))
        elif platform in ["win64", "win32"]:
            open_with_default_app(path)
        elif platform == "wsl":
            subprocess.call("cmd.exe /C start".split() + [filename])
        else:  # linux variants
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        traceback.print_exc()
        pass
        # import webbrowser
        # webbrowser.open(url)


def create_version_html() -> str:
    versions_html = f"""
<span>Version 1.0</span>
"""
    return versions_html


def compute_cosine_distance(emb1, emb2) -> float:
    return distance.cosine(emb1, emb2)
