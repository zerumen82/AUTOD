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
            model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../models/inswapper_128.onnx')  
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=roop.globals.providers)  
    return FACE_SWAPPER  
  
def process_img(source_img, target_path, output_file):  
    frame = cv2.imread(target_path)  
    face = get_face_single(frame)  
    source_face = get_face_single(cv2.imread(source_img))  
    result = get_face_swapper().get(frame, face, source_face, paste_back=True)  
    cv2.imwrite(output_file, result)  
    print('Image saved as:', output_file) 
