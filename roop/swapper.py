import os  
from tqdm import tqdm  
import cv2  
import insightface  
import threading  
import roop.globals  
from roop.analyser import get_face_single, get_face_many  
  
FACE_SWAPPER = None  
THREAD_LOCK = threading.Lock()  
  
def get_face_swapper():  
    global FACE_SWAPPER  
    with THREAD_LOCK:  
        if FACE_SWAPPER is None:  
            # Check for higher resolution model first (256x256 > 128x128)
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            model_256 = os.path.join(base_dir, 'models', 'inswapper_256.onnx')
            model_128 = os.path.join(base_dir, 'models', 'inswapper_128.onnx')
            if os.path.exists(model_256):
                model_path = model_256
            else:
                model_path = model_128  
            # Detector GPU antes de inicializar
            try:
                import torch
                use_cuda = torch.cuda.is_available()
            except ImportError:
                use_cuda = False
            
            if use_cuda:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = roop.globals.providers
            
            try:
                loaded_model = insightface.model_zoo.get_model(model_path, providers=providers)
                if loaded_model is None or loaded_model.__class__.__name__ != 'INSwapper':
                    if 'inswapper' in model_path.lower():
                        from insightface.model_zoo.inswapper import INSwapper
                        print(f"[Swapper] Forcing manual INSwapper load for: {model_path}")
                        loaded_model = INSwapper(model_file=model_path)
                FACE_SWAPPER = loaded_model
            except Exception as e:
                print(f"[Swapper] Error loading model: {e}")
                try:
                    from insightface.model_zoo.inswapper import INSwapper
                    FACE_SWAPPER = INSwapper(model_file=model_path)
                    print(f"[Swapper] Manual fallback successful")
                except Exception as e2:
                    print(f"[Swapper] Manual fallback failed: {e2}")
    return FACE_SWAPPER  
  
def process_img(source_img, target_path, output_file):
    from roop.capturer import get_image_frame
    frame = get_image_frame(target_path)
    face = get_face_single(frame)
    source_face = get_face_single(get_image_frame(source_img))
    
    swapper = get_face_swapper()
    if swapper is None:
        print("Error: Face swapper model not loaded")
        return

    try:
        import inspect
        sig = inspect.signature(swapper.get)
        if 'paste_back' in sig.parameters:
            result = swapper.get(frame, face, source_face, paste_back=True)
        else:
            result = swapper.get(frame, face, source_face)
            
        cv2.imwrite(output_file, result)
        print('Image saved as:', output_file)
    except Exception as e:
        print(f"Error during face swap: {e}")
